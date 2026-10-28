"""通过真实 Tasks API 和微型 CPU ONNX 会话验证应用模型租约。"""

from __future__ import annotations

import gc
import time
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import pytest
from fastapi.testclient import TestClient

from mineru.config import get_config
from mineru.integrations.docvortex import build_metadata
from mineru.model.runtime import hybrid
from mineru.model.runtime.ownership import runtime_document_scope
from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson, PageInfo


def _identity_model() -> bytes:
    """生成无需下载权重的 ONNX 模型，让实际原生会话参与租约验证。"""
    value = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["value"], ["output"])], "identity", [value], [output])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model.SerializeToString()


def _parse_task(client: TestClient, content: bytes) -> None:
    """经过真实创建、上传、后台执行和查询流程，确认单文件产物成功发布。"""
    created = client.post(
        "/v1/tasks",
        json={"files": [{"file_name": "input.html", "bytes": len(content)}], "tier": "flash"},
    )
    assert created.status_code == 202, created.text
    receipt = created.json()
    upload = receipt["files"][0]["upload"]
    response = client.put(upload["url"], content=content, headers=upload["headers"])
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 5
    while True:
        response = client.get(f"/v1/tasks/{receipt['task_id']}")
        assert response.status_code == 200, response.text
        task = response.json()
        if task["status"] in {"completed", "failed", "canceled", "partial"}:
            assert task["status"] == "completed", task
            assert task["files"][0]["parse"]["tier"] == "flash"
            assert client.get(task["files"][0]["result_url"]).status_code == 200
            return
        assert time.monotonic() < deadline, task
        time.sleep(0.01)


def test_api_apps_share_native_sessions_until_last_owner_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一个应用关闭后其它应用仍可推理，最后应用退出后会话可回收且后续重新创建。"""
    monkeypatch.setattr(hybrid.AtomModelSingleton, "_models", {})
    monkeypatch.setattr(hybrid.HybridLocalModelContextSingleton, "_models", {})
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    model_bytes = _identity_model()
    sessions: list[weakref.ReferenceType[onnxruntime.InferenceSession]] = []

    def create_session(model_name: str, **_options: Any) -> onnxruntime.InferenceSession:
        """替换权重装载，每个真实缓存条目仍持有可执行的原生 ORT 会话。"""
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
        sessions.append(weakref.ref(session))
        return session

    async def parse_with_shared_models(_path: str, **_options: Any) -> ParseResult:
        """在真实执行所属应用中访问上下文和惰性模型，不保留模型或文档引用。"""
        with runtime_document_scope():
            context = hybrid.HybridLocalModelContextSingleton().get_model()
            output = context.mfr_model.run(None, {"value": np.array([7], dtype=np.float32)})
            np.testing.assert_array_equal(output[0], [7])
            return ParseResult(
                MiddleJson(
                    pages=[PageInfo(page_idx=0)],
                    is_full_document=True,
                    metadata={"file_suffix": "html", "producer": {"name": "mineru", "version": "test"}},
                    extensions=build_metadata(effort="flash", parse_mode="txt"),
                )
            )

    monkeypatch.setattr(hybrid, "atom_model_init", create_session)
    monkeypatch.setattr(api_server, "parse_async", parse_with_shared_models)

    with TestClient(api_server.create_app(upload_dir=str(tmp_path / "second"), tiers=("flash",))) as second:
        with TestClient(api_server.create_app(upload_dir=str(tmp_path / "first"), tiers=("flash",))) as first:
            _parse_task(first, b"<p>first</p>")
            _parse_task(second, b"<p>second</p>")
            assert len(sessions) == 3  # OCR、Layout 与惰性 MFR 只创建一份。
        gc.collect()
        assert all(reference() is not None for reference in sessions)
        _parse_task(second, b"<p>still active</p>")
        assert len(sessions) == 3
    gc.collect()
    assert all(reference() is None for reference in sessions)

    with TestClient(api_server.create_app(upload_dir=str(tmp_path / "replacement"), tiers=("flash",))) as replacement:
        _parse_task(replacement, b"<p>replacement</p>")
        assert len(sessions) == 6
    gc.collect()
    assert all(reference() is None for reference in sessions)
