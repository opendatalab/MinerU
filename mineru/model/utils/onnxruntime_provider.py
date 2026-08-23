import onnxruntime as ort

AVAILABLE_PROVIDERS: list[str] = ort.get_available_providers()


def ort_providers(device: str | None = None) -> list[tuple[str, dict[str, object]]]:
    """根据 device 选择 onnxruntime providers。"""
    norm = (device or "").lower().split(":", 1)[0]
    if norm != "cpu" and "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]
    # if "CoreMLExecutionProvider" in AVAILABLE_PROVIDERS:
    #     return [
    #         ("CoreMLExecutionProvider", {}),
    #         ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    #     ]
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def ort_session(model_path: str, device: str | None = None, intra_op_num_threads: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_num_threads > 0:
        opts.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers(device))
