"""Opt-in kernel dispatch checks on a Blackwell GPU (SM 10.x), run on a GPU allocated to the test.

I2V_RUN_GPU_TESTS=1 python -m pytest tests/gpu -q -s
"""

import json
import os
import statistics
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

pytestmark = pytest.mark.skipif(os.environ.get("I2V_RUN_GPU_TESTS") != "1", reason="requires opt-in GPU allocation")


@pytest.fixture
def blackwell():
    assert torch.cuda.is_available(), "GPU tests explicitly requested, but CUDA is unavailable"
    assert torch.cuda.get_device_capability()[0] == 10, "These checks target Blackwell GPUs (SM 10.x)"
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    )
    expected = os.environ.get("I2V_EXPECT_CUDNN")
    if expected:
        assert torch.backends.cudnn.version() == int(expected), (
            "Loaded cuDNN differs from I2V_EXPECT_CUDNN"
        )


@torch.no_grad()
def test_qwen_patch_conv3d_does_not_fall_back_to_slow_kernel(blackwell):
    # The patch embedding shape of the teacher's Qwen vision encoder.
    conv = torch.nn.Conv3d(3, 1152, (2, 16, 16), stride=(2, 16, 16), bias=False).cuda().bfloat16().eval()
    x = torch.randn(1024, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        conv(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        result = conv(x)
        torch.cuda.synchronize()
    names = sorted(event.key for event in prof.key_averages() if "conv" in event.key)
    timings = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        conv(x)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    print(json.dumps({"conv_ops": names, "median_ms": statistics.median(timings)}))
    assert torch.isfinite(result).all()
    assert not any("slow_conv" in name for name in names), f"Qwen patch Conv3d fallback: {names}"
    assert any("cudnn" in name for name in names), f"Expected validated cuDNN dispatch, got {names}"
    # Other jobs on the GPU affect timings, so a time budget applies only when I2V_CONV_MAX_MS sets one.
    budget = os.environ.get("I2V_CONV_MAX_MS")
    if budget:
        assert statistics.median(timings) < float(budget)


@torch.no_grad()
def test_sdpa_really_dispatches_flash_for_video_attention(blackwell):
    q = torch.randn(1, 40, 4096, 128, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        output = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
    names = sorted(e.key for e in prof.key_averages() if "attention" in e.key)
    print(json.dumps({"sdpa_ops": names}))
    assert torch.isfinite(output).all()
    assert any("_scaled_dot_product_flash_attention" in name for name in names), names


def test_fa_wrapper_forwards_global_determinism_and_exact_lengths(blackwell, monkeypatch):
    from i2vt.vendor.wan.models import attention_utils as attention

    kernel = Mock(side_effect=lambda **kwargs: kwargs["q"])
    monkeypatch.setattr(attention, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(attention, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(attention, "flash_attn", SimpleNamespace(flash_attn_varlen_func=kernel), raising=False)
    q = torch.randn(1, 2323, 2, 128, device="cuda", dtype=torch.bfloat16)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        result = attention.flash_attention(q, q, q, version=2)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    assert result.shape == q.shape
    arguments = kernel.call_args.kwargs
    assert arguments["deterministic"] is True
    assert arguments["cu_seqlens_q"].dtype == torch.int32
    assert arguments["cu_seqlens_q"].tolist() == [0, 2323]
