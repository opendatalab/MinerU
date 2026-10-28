"""离线文档资源、访问控制和代理路径回归。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mineru.parser import api_server


@pytest.mark.parametrize("root", ["", "/mineru"])
def test_offline_docs_resources_and_proxy_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """断言 Swagger/ReDoc 全部使用本地资源，授权回调遵守代理前缀。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), api_key="secret")
    with TestClient(app, root_path=root) as client:
        for page in ("/docs", "/redoc"):
            response = client.get(root + page)
            assert response.status_code == 200
            assert f"{root}/docs/assets/" in response.text
            assert "cdn.jsdelivr" not in response.text and "fonts.googleapis" not in response.text
        assert f"{root}/docs/oauth2-redirect" in client.get(root + "/docs").text
        assert client.get(root + "/docs/oauth2-redirect").status_code == 200
        script = client.get(root + "/docs/assets/redoc.standalone.js").text
        assert "https://cdn.redoc.ly/redoc/logo-mini.svg" not in script
        assert "data:image/png;base64," in script
        for asset in ("swagger-ui-bundle.js", "swagger-ui.css", "redoc.standalone.js", "favicon.png"):
            assert (
                client.get(
                    f"{root}/docs/assets/" + (asset if asset == "redoc.standalone.js" else f"vendor/{asset}")
                ).status_code
                == 200
            )
        schema = client.get(root + "/openapi.json").json()
        assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
        assert client.get(root + "/v1/usage").status_code == 401
        assert client.get(root + "/v1/usage", headers={"Authorization": "Bearer secret"}).status_code == 200
        for path, status_code in (("/v1/tasks", 405), ("/v1/files", 404)):
            assert client.get(root + path, headers={"Authorization": "Bearer secret"}).status_code == status_code


def test_disabled_docs_mount_no_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭文档时不挂载静态资源，三个既有文档入口全部不可访问。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    monkeypatch.setenv("MINERU_API_ENABLE_FASTAPI_DOCS", "false")
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",))
    with TestClient(app) as client:
        for path in ("/docs", "/redoc", "/openapi.json", "/docs/assets/vendor/favicon.png"):
            assert client.get(path).status_code == 404
