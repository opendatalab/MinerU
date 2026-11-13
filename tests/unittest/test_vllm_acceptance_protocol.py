"""核对真机验收的观测条件，避免 FINAL_ONLY 终态被误认为活跃推理。"""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from scripts.acceptance.vllm_lifecycle import observe_generation


def test_active_observation_requires_unfinished_real_token_and_keeps_sampling() -> None:
    """改变副本的输出投递方式，真实非终态才计为活跃，退出及时清理。"""

    async def run() -> None:
        """逐帧检查 token 与 finished 状态，而不是只检查生成器已进入。"""
        active: set[str] = set()
        metrics = {"token_events": 0}
        sampling = SimpleNamespace(output_kind="FINAL_ONLY", temperature=0.2)

        async def generate(**kwargs: Any) -> AsyncIterator[Any]:
            """提供空帧、非终态 token 和终态帧三种真实协议形态。"""
            assert kwargs["sampling_params"] is not sampling
            assert kwargs["sampling_params"].output_kind == "CUMULATIVE"
            assert kwargs["sampling_params"].temperature == sampling.temperature
            yield SimpleNamespace(outputs=[], finished=False)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1])], finished=False)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])], finished=True)

        states = []
        async for _ in observe_generation(generate, active, metrics, "CUMULATIVE", sampling_params=sampling, request_id="one"):
            states.append(bool(active))
        assert states == [False, True, False]
        assert metrics["token_events"] == 1
        assert not active
        assert sampling.output_kind == "FINAL_ONLY"

    asyncio.run(run())
