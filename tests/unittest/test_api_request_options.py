# Copyright (c) Opendatalab. All rights reserved.
"""mineru-api 请求参数解析与启动配置分拣的单元测试。

覆盖 issue #5433 的两个缺陷：
- Bug A: query string 参数被静默忽略（应作为 form 缺省时的回退来源）；
- Bug B: CLI 透传的请求级参数残留在 model_config，与请求参数合并时触发
  ``dict() got multiple values for keyword argument`` 一类的 TypeError。
"""

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mineru.cli.api_request import ParseRequestOptions, parse_request_form
from mineru.cli.vlm_preload import split_service_and_model_config


def _build_probe_app() -> FastAPI:
    """构造挂载 parse_request_form 依赖的探针应用。

    :return: 回显关键请求参数的 FastAPI 应用
    """
    app = FastAPI()

    @app.post("/file_parse")
    async def probe(
        request_options: Annotated[
            ParseRequestOptions, Depends(parse_request_form)
        ],
    ) -> dict:
        """回显服务端实际解析到的关键参数，供断言使用。

        :param request_options: parse_request_form 的解析产物
        :return: 关键参数快照
        """
        return {
            "backend": request_options.backend,
            "return_images": request_options.return_images,
            "response_format_zip": request_options.response_format_zip,
            "lang_list": request_options.lang_list,
            "start_page_id": request_options.start_page_id,
        }

    return app


def test_query_params_apply_when_body_omits_them() -> None:
    """query 参数在 body 未提供同名字段时应生效，而不是静默落回默认值。"""
    client = TestClient(_build_probe_app())
    resp = client.post(
        "/file_parse?backend=pipeline&return_images=true&response_format_zip=true"
        "&start_page_id=2",
        files={"files": ("sample.pdf", b"%PDF-1.4 minimal", "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend"] == "pipeline"
    assert body["return_images"] is True
    assert body["response_format_zip"] is True
    assert body["start_page_id"] == 2


def test_form_data_takes_precedence_over_query() -> None:
    """form 与 query 同时提供同名参数时，应保持 form 优先的向后兼容语义。"""
    client = TestClient(_build_probe_app())
    resp = client.post(
        "/file_parse?backend=pipeline",
        files={"files": ("sample.pdf", b"%PDF-1.4 minimal", "application/pdf")},
        data={"backend": "vlm-engine"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["backend"] == "vlm-engine"


def test_split_service_and_model_config_excludes_request_level_keys() -> None:
    """请求级 CLI 参数不应残留到 model_config，避免与请求参数合并冲突。"""
    service_config, model_config = split_service_and_model_config(
        {"backend": "vlm", "start_page_id": "3", "device": "cuda"}
    )
    assert service_config == {"enable_vlm_preload": False}
    assert "backend" not in model_config
    assert "start_page_id" not in model_config


def test_split_service_and_model_config_keeps_model_level_keys() -> None:
    """模型级 CLI 配置应原样保留在 model_config，供解析链路补缺使用。"""
    _, model_config = split_service_and_model_config(
        {"device": "cuda", "vrs": 2}
    )
    assert model_config == {"device": "cuda", "vrs": 2}
