from cellvision.runtime import detect_compute_runtime


def test_cpu_runtime_is_always_available():
    runtime = detect_compute_runtime("cpu")
    assert runtime.requested_device == "cpu"
    assert runtime.selected_device == "cpu"
    assert runtime.worker_kind == "cpu"
    assert runtime.cpu_count >= 1


def test_auto_runtime_selects_a_supported_backend():
    runtime = detect_compute_runtime("auto")
    assert runtime.selected_device in {"cpu", "cuda"}
    assert runtime.worker_kind == ("gpu" if runtime.selected_device == "cuda" else "cpu")
    assert runtime.as_dict()["label"]


def test_cuda_request_falls_back_when_cuda_is_unavailable():
    runtime = detect_compute_runtime("cuda")
    if not runtime.cuda_available:
        assert runtime.selected_device == "cpu"
        assert runtime.fallback_reason
