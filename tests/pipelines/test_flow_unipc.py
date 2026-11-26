"""Golden trajectories of the original Wan (Alibaba) flow UniPC solver, which the vendored solver must reproduce."""

import json
from pathlib import Path

import pytest
import torch

from i2vt.vendor.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

CASES = json.loads((Path(__file__).parent / "fixtures/flow_unipc.json").read_text())


@pytest.mark.parametrize("case", CASES)
def test_diffusers_solver_preserves_original_flow_trajectory(case):
    scheduler = FlowUniPCMultistepScheduler(shift=1, predict_x0=case["predict_x0"])
    scheduler.set_timesteps(case["steps"], device="cpu", shift=case["shift"])
    sample = torch.tensor([[0.1, -0.3], [0.7, 1.1]])
    for i, timestep in enumerate(scheduler.timesteps):
        velocity = sample * 0.125 + (i + 1) * 0.01
        sample = scheduler.step(velocity, timestep, sample).prev_sample
        torch.testing.assert_close(sample, torch.tensor(case["trace"][i]), rtol=1e-6, atol=1e-7)
