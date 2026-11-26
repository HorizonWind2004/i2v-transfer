"""LoRA weights train in fp32 while the forward pass keeps the base model's dtype.

AdamW's per-step update is below bfloat16's spacing at typical LoRA-A magnitudes, so bfloat16
master weights would round most updates away.
"""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "backend", ["src/i2vt/models/bernini/modeling_bernini.py", "src/i2vt/models/videocof/modeling_videocof.py"],
)
def test_lora_trains_in_fp32_master_weights(backend):
    src = (ROOT / backend).read_text()
    assert "self.network.to(device=device, dtype=torch.float32)" in src
    assert "self.network.to(device=device, dtype=self.weight_dtype)" not in src


def test_bernini_low_noise_expert_also_uses_fp32_master_weights():
    src = (ROOT / "src/i2vt/models/bernini/modeling_bernini.py").read_text()
    assert "self.network_2.to(device=device, dtype=torch.float32)" in src
    assert "self.network_2.to(device=device, dtype=self.weight_dtype)" not in src


def test_fp32_lora_forward_keeps_the_input_dtype_and_learns_a():
    from i2vt.vendor.wan.utils.lora_utils import LoRAModule

    base = torch.nn.Linear(64, 64).to(torch.bfloat16)
    lora = LoRAModule("t", base, multiplier=1.0, lora_dim=8, alpha=4.0)
    lora.apply_to()
    lora.to(torch.float32)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    assert base(x).dtype == torch.bfloat16
    a0 = lora.lora_down.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=2e-5)
    for _ in range(5):
        opt.zero_grad()
        lora.lora_up.weight.data.normal_(std=1e-2)
        base(x).float().pow(2).mean().backward()
        opt.step()
    changed = (lora.lora_down.weight != a0).float().mean().item()
    assert changed > 0.9, f"only {changed:.0%} of lora_down moved; updates are being rounded away"
