"""CPU numerical regression against reference traces, with optional exact per-version hashes.

The reference traces come from an independent implementation. The run uses the real backend
packing, CFG, schedulers and the full Trainer; network weights, the teacher and the distributed
operations are small stand-ins.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from tests.helpers import load_config
from tests.training.backend_fixture import run_case

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = json.loads((FIXTURES / "backend_baseline.json").read_text())["traces"]


@pytest.fixture(scope="module")
def runtime_baseline():
    # Floating-point reductions and autograd need not be bitwise identical across PyTorch releases,
    # so exact hashes are kept per supported runtime. They come from the independent reference
    # implementation, never from the current code.
    if os.environ.get("I2V_EXACT_CPU_TRACES") != "1":
        return None
    version = torch.__version__.split("+")[0]
    filenames = {
        "2.14.0": "backend_baseline.json",
        "2.9.1": "backend_baseline_torch_2_9_1.json",
    }
    if version not in filenames:
        pytest.fail(
            f"No exact CPU trace reference for torch {version}. "
            f"Use a reference runtime ({', '.join(filenames)})."
        )
    reference = json.loads((FIXTURES / filenames[version]).read_text())
    assert reference["provenance"]["torch"].split("+")[0] == version
    assert reference["traces"].keys() == BASELINE.keys()
    return reference["traces"]


def serialize(value):
    if torch.is_tensor(value):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
    return value


NUMERIC = json.loads((FIXTURES / "backend_numeric_reference.json").read_text())["traces"]


def assert_trace_close(actual, expected, path="trace"):
    """CPU float32 tolerance; retain exact structure, strings and integer pixels."""
    assert type(actual) is type(expected), f"{path}: type {type(actual)} != {type(expected)}"
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            assert_trace_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length differs"
        for index, (value, reference) in enumerate(zip(actual, expected)):
            assert_trace_close(value, reference, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), f"{path}: non-finite value"
        limit = 1e-7 + 1e-6 * abs(expected)
        assert abs(actual - expected) <= limit, (
            f"{path}: actual={actual}, expected={expected}, delta={abs(actual - expected)}, limit={limit}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("case", BASELINE)
def test_backend_matches_reference_trace_exactly(case, tmp_path, runtime_baseline):
    backend, loss, chunk, accum, rollout_cfg, loss_cfg = case.split("/")
    config = load_config(backend)
    config.train.loss_type = loss
    config.train.gradient_accumulation_steps = int(accum)
    config.train.max_grad_norm = 1000
    config.train.rollout.num_inference_steps = 4
    config.train.rollout.train_steps = 0
    config.train.rollout.chunk_size = int(chunk)
    config.train.rollout.guidance_scale = float(rollout_cfg)
    config.train.rollout.student_guidance_scale = float(loss_cfg)
    trace = run_case(config, tmp_path)
    assert_trace_close(serialize(trace), NUMERIC[case], case)
    if runtime_baseline is not None:
        assert trace.keys() == runtime_baseline[case].keys()
        for name, expected in runtime_baseline[case].items():
            actual = hashlib.sha256(json.dumps(serialize(trace[name]), sort_keys=True).encode()).hexdigest()
            assert actual == expected, f"{case}: {name} differs from the independent reference"
    if backend == "videocof":
        for step in trace["steps"]:
            assert torch.equal(step["before"][:, :, :1], step["after"][:, :, :1])
            assert not torch.equal(step["before"][:, :, 1:2], step["after"][:, :, 1:2])
