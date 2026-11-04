"""验证 Doclib 持久凭证、全部 HTTP 入口和 SDK 的认证边界。"""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from doclib_test_utils import DOCLIB_AUTH_HEADERS, TEST_DOCLIB_TOKEN, authenticated_doclib_config
from fastapi.testclient import TestClient
from pydantic import SecretStr

from mineru import config as configuration
from mineru.cli.commands import server as server_commands
from mineru.config import Config, load_config, load_doclib_server_config, update_config_file
from mineru.doclib import client as client_module
from mineru.doclib.client import DoclibClient
from mineru.doclib.endpoint import EndpointTransport, write_endpoint_file
from mineru.doclib.parse_server_state import ParseServerHealth
from mineru.doclib.server import DoclibServer
from mineru.errors import MineruError
from mineru.utils.private_file import create_private_tempfile, protect_private_file


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离凭证配置和进程缓存，测试不得写入开发者的实际配置目录。"""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    monkeypatch.delenv("MINERU_CONFIG", raising=False)
    monkeypatch.delenv("MINERU_DOCLIB_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(configuration, "_loaded_config", None)
    return tmp_path / "config.yaml"


@pytest.mark.parametrize("authorization", [None, "Bearer incorrect-token"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/server/status"),
        ("GET", "/api/v1/docs/by-path"),
        ("GET", "/api/v1/files/by-path"),
        ("GET", "/api/v1/search"),
        ("GET", "/api/v1/content"),
        ("POST", "/api/v1/scans"),
        ("POST", "/api/v1/docs/example/exports"),
        ("PUT", "/api/v1/configs/parse_server.local.mode"),
        ("POST", "/api/v1/server/shutdown"),
        ("GET", "/api/docs"),
        ("GET", "/api/redoc"),
        ("GET", "/api/openapi.json"),
    ],
)
def test_all_doclib_http_surfaces_require_auth_before_business_state(
    tmp_path: Path, method: str, path: str, authorization: str | None
) -> None:
    """没有业务依赖仍应直接返回 401，不能读文件、导出、改配置或关闭服务。"""
    server = DoclibServer(object(), config=authenticated_doclib_config(Config()), health=ParseServerHealth())
    output = tmp_path / "export.txt"
    headers = {"Authorization": authorization} if authorization is not None else {}
    with TestClient(server.app) as client:
        response = client.request(method, path, headers=headers, json={"output": str(output), "path": str(tmp_path)})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_doclib_token"
    assert TEST_DOCLIB_TOKEN not in response.text
    assert not output.exists()


def test_authenticated_shutdown_and_duplicate_authorization() -> None:
    """合法凭证保留 shutdown 能力，重复授权头即使都正确也拒绝执行。"""
    calls: list[bool] = []

    def shutdown() -> None:
        """记录安全的停止钩子，测试不得发送进程信号。"""
        calls.append(True)

    server = DoclibServer(
        SimpleNamespace(shutdown=shutdown), config=authenticated_doclib_config(Config()), health=ParseServerHealth()
    )
    assert TEST_DOCLIB_TOKEN not in repr(server.app.user_middleware)
    with TestClient(server.app) as client:
        duplicate = client.post("/api/v1/server/shutdown", headers=list(DOCLIB_AUTH_HEADERS.items()) * 2)
        assert duplicate.status_code == 401
        assert calls == []
        accepted = client.post("/api/v1/server/shutdown", headers=DOCLIB_AUTH_HEADERS)
        assert accepted.status_code == 200
        assert calls == [True]


def test_server_requires_explicit_token_for_embedded_configuration() -> None:
    """嵌入式服务没有凭证时拒绝构造，不提供隐式匿名模式。"""
    with pytest.raises(ValueError, match="requires doclib.auth.token"):
        DoclibServer(object(), config=Config(), health=ParseServerHealth())


def test_token_is_generated_once_and_preserves_existing_configuration(config_path: Path) -> None:
    """首次启动生成凭证，后续启动和其他配置更新保持凭证与原有设置。"""
    config_path.write_text("log:\n  level: debug\nmodel:\n  source: auto\n", encoding="utf-8")
    first = load_doclib_server_config()
    token = first.config.doclib.auth.token.get_secret_value()
    assert len(token) >= 32
    assert first.sources[("doclib", "auth", "token")] == "file"
    before = config_path.read_bytes()
    second = load_doclib_server_config()
    assert second.config.doclib.auth.token.get_secret_value() == token
    assert config_path.read_bytes() == before
    update_config_file({"model": {"source": "modelscope"}})
    raw = yaml.safe_load(config_path.read_text())
    assert raw["log"]["level"] == "debug"
    assert raw["model"]["source"] == "modelscope"
    assert raw["doclib"]["auth"]["token"] == token
    if os.name != "nt":
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_concurrent_initialization_uses_one_persisted_token(config_path: Path) -> None:
    """并发启动竞争同一个配置锁时只产生一个有效凭证。"""
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(lambda _: load_doclib_server_config(), range(8)))
    tokens = {item.config.doclib.auth.token.get_secret_value() for item in loaded}
    assert len(tokens) == 1
    assert yaml.safe_load(config_path.read_text())["doclib"]["auth"]["token"] in tokens


def test_first_start_creates_private_home_and_persistent_token(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """全新 home 的 0700 目录在 Windows 使用 OWNER RIGHTS，也必须正常生成凭证。"""
    home = config_path.parent / "new-home"
    monkeypatch.setenv("MINERU_HOME", str(home))
    assert not home.exists()
    first = load_doclib_server_config()
    path = home / "config.yaml"
    assert path.is_file()
    token = first.config.doclib.auth.token.get_secret_value()
    assert len(token) >= 32
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["doclib"]["auth"]["token"] == token
    assert load_doclib_server_config().config.doclib.auth.token.get_secret_value() == token
    assert not list(home.glob(".config.*.tmp"))


def test_configuration_reads_do_not_generate_credentials(config_path: Path) -> None:
    """普通 SDK 和配置读取没有生成凭证或配置文件的副作用。"""
    assert load_config().config.doclib.auth.token is None
    client = DoclibClient()
    client.close()
    assert not config_path.exists()


def test_credentials_are_redacted_and_survive_config_overrides(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """日志用的 repr/JSON 隐藏凭证，环境覆盖和配置复制不把掩码误当真实凭证。"""
    config_path.write_text(yaml.safe_dump({"doclib": {"auth": {"token": TEST_DOCLIB_TOKEN}}}))
    monkeypatch.setenv("MINERU_LOG_LEVEL", "debug")
    loaded = load_doclib_server_config()
    assert loaded.config.doclib.auth.token.get_secret_value() == TEST_DOCLIB_TOKEN
    assert TEST_DOCLIB_TOKEN not in repr(loaded)
    assert TEST_DOCLIB_TOKEN not in loaded.config.model_dump_json()
    assert configuration.PatchedConfig(log={"level": "warning"}).doclib.auth.token.get_secret_value() == TEST_DOCLIB_TOKEN


def test_environment_token_is_not_written_to_configuration(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """显式环境凭证覆盖文件值但不回写到磁盘，避免持久化部署注入的密钥。"""
    monkeypatch.setenv("MINERU_DOCLIB_AUTH_TOKEN", TEST_DOCLIB_TOKEN)
    loaded = load_doclib_server_config()
    assert loaded.config.doclib.auth.token.get_secret_value() == TEST_DOCLIB_TOKEN
    assert loaded.sources[("doclib", "auth", "token")] == "env"
    assert not config_path.exists()


def test_generation_fails_before_writing_secret_when_permissions_fail(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无法保护临时文件时拒绝启动，磁盘上不留下生成的凭证。"""

    def reject_permissions(directory: str) -> tuple[int, str]:
        """模拟文件系统无法创建私密临时文件，确认目标配置尚未写入。"""
        assert Path(directory) == config_path.parent
        assert not config_path.exists()
        raise PermissionError("private permissions unavailable")

    monkeypatch.setattr(configuration, "create_private_tempfile", reject_permissions)
    with pytest.raises(PermissionError, match="private permissions unavailable"):
        load_doclib_server_config()
    assert not config_path.exists()
    assert not list(config_path.parent.glob(".config.*.tmp"))


@pytest.mark.parametrize("field", ["base_url", "socket_path"])
def test_explicit_endpoint_never_implicitly_uses_local_token(config_path: Path, field: str) -> None:
    """指定连接地址必须同时提供凭证，不能把本机 token 自动发给其他目标。"""
    load_doclib_server_config()
    value = "http://example.test" if field == "base_url" else "/tmp/other.sock"
    with pytest.raises(ValueError, match="explicit token"):
        DoclibClient(**{field: value})


def test_sdk_loads_new_token_after_cached_configuration_and_authenticates_probe(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 启动前的旧配置缓存不遮蔽新凭证，状态身份校验和后续业务请求都带认证头。"""
    assert configuration.get_config().doclib.auth.token is None
    loaded = load_doclib_server_config()
    token = loaded.config.doclib.auth.token.get_secret_value()
    endpoint = config_path.parent / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint, pid=123, server_id="server-123", transports=[EndpointTransport(type="tcp", base_url="http://local")]
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """模拟会校验真实请求头的服务，身份探测和业务读取共用相同规则。"""
        requests.append(request)
        assert request.headers["authorization"] == f"Bearer {token}"
        assert token not in str(request.url)
        return httpx.Response(
            200,
            json={
                "running": True,
                "pid": 123,
                "server_id": "server-123",
                "socket_path": "",
                "data_dir": "",
                "sqlite_path": "",
                "log_path": "",
            },
        )

    transport = httpx.Client(transport=httpx.MockTransport(respond), base_url="http://local")
    monkeypatch.setattr(client_module, "_client_for_transport", lambda *args, **kwargs: transport)
    client = DoclibClient()
    try:
        assert client.get_server_status().server_id == "server-123"
    finally:
        client.close()
    assert len(requests) == 2


def test_sdk_missing_token_and_401_never_fall_back_to_anonymous(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """缺失凭证不发请求；认证失败不尝试其他 transport，更不能删除认证头重试。"""
    endpoint = config_path.parent / "doclib.endpoint.json"
    write_endpoint_file(
        endpoint,
        pid=123,
        server_id="server-123",
        transports=[
            EndpointTransport(type="tcp", base_url="http://first"),
            EndpointTransport(type="tcp", base_url="http://second"),
        ],
    )
    requests: list[httpx.Request] = []

    def reject(request: httpx.Request) -> httpx.Response:
        """两个候选服务均拒绝凭证，记录客户端是否错误地继续或降级。"""
        requests.append(request)
        return httpx.Response(401, json={"error": {"code": "invalid_doclib_token", "message": "invalid token"}})

    def transport(item: EndpointTransport, *, timeout: int) -> httpx.Client:
        """使用真实 httpx 请求组装与响应解码，仅替换网络传输。"""
        return httpx.Client(base_url=item.base_url, transport=httpx.MockTransport(reject), timeout=timeout)

    monkeypatch.setattr(client_module, "_client_for_transport", transport)
    for token in (None, TEST_DOCLIB_TOKEN):
        client = DoclibClient(token=token)
        try:
            with pytest.raises(MineruError) as error:
                client.get_server_status()
            assert error.value.code == "invalid_doclib_token"
        finally:
            client.close()
        if token is None:
            assert requests == []
    assert len(requests) == 1
    assert requests[0].url.host == "first"
    assert requests[0].headers["authorization"] == f"Bearer {TEST_DOCLIB_TOKEN}"


def test_token_rotation_takes_effect_for_new_server_only() -> None:
    """配置修改不改变已启动实例的认证快照，重启后的实例仅接受新凭证。"""
    cfg = authenticated_doclib_config(Config())
    first = DoclibServer(SimpleNamespace(shutdown=lambda: None), config=cfg, health=ParseServerHealth())
    new_token = "replacement-token-with-at-least-32-characters"
    cfg.doclib.auth.token = SecretStr(new_token)
    second = DoclibServer(SimpleNamespace(shutdown=lambda: None), config=cfg, health=ParseServerHealth())
    with TestClient(first.app) as old, TestClient(second.app) as new:
        assert old.post("/api/v1/server/shutdown", headers=DOCLIB_AUTH_HEADERS).status_code == 200
        assert new.post("/api/v1/server/shutdown", headers=DOCLIB_AUTH_HEADERS).status_code == 401
        assert new.post("/api/v1/server/shutdown", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200


def test_cli_does_not_report_authentication_failure_as_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 保留认证失败，不能误报停止或另启一个争抢目录的服务。"""

    def rejected_client(*args: object, **kwargs: object) -> None:
        """模拟 SDK 发现缺失凭证时的明确认证错误。"""
        raise MineruError("invalid_doclib_token", "missing token")

    monkeypatch.setattr(client_module, "DoclibClient", rejected_client)
    with pytest.raises(MineruError, match="missing token"):
        server_commands._server_running()


def test_private_file_permissions_are_enforced(tmp_path: Path) -> None:
    """实际文件系统落实私密权限；Windows 分支通过 DACL 设置后的读回验证。"""
    path = tmp_path / "config.yaml"
    path.write_text("private test value", encoding="utf-8")
    protect_private_file(path)
    assert path.read_text() == "private test value"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_private_tempfile_is_protected_before_any_content(tmp_path: Path) -> None:
    """真实临时文件在返回可写句柄前已经落实权限，正文随后才写入。"""
    fd, filename = create_private_tempfile(str(tmp_path))
    path = Path(filename)
    try:
        assert path.read_bytes() == b""
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("private test content")
        assert path.read_text() == "private test content"
    finally:
        path.unlink(missing_ok=True)


def test_custom_config_path_preserves_environment_placeholders(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自定义配置首次生成凭证时保留环境占位符，不误写默认配置或展开其他秘密。"""
    custom = config_path.parent / "custom.yaml"
    custom.write_text("log:\n  level: ${DOCLIB_TEST_LOG_LEVEL}\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_CONFIG", str(custom))
    monkeypatch.setenv("DOCLIB_TEST_LOG_LEVEL", "debug")
    loaded = load_doclib_server_config()
    assert loaded.config_file == str(custom)
    assert loaded.config.log.level == "debug"
    assert not config_path.exists()
    raw = yaml.safe_load(custom.read_text())
    assert raw["log"]["level"] == "${DOCLIB_TEST_LOG_LEVEL}"
    assert raw["doclib"]["auth"]["token"] == loaded.config.doclib.auth.token.get_secret_value()
