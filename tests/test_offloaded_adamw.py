"""The CPU-offloaded AdamW must follow the same trajectory as torch.optim.AdamW."""
import torch

from i2vt.trainer import OffloadedAdamW


def test_matches_adamw():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(3))]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    ref = torch.optim.AdamW(a, lr=1e-2, weight_decay=0.01)
    off = OffloadedAdamW(b, lr=1e-2, weight_decay=0.01)
    for step in range(5):
        for params, opt in ((a, ref), (b, off)):
            opt.zero_grad(set_to_none=True)
            loss = sum(((p * (step + 1)) ** 2).sum() for p in params)
            loss.backward()
            opt.step()
        for p, q in zip(a, b):
            assert torch.allclose(p, q, atol=1e-6), step
    assert off.param_groups[0]["lr"] == 1e-2
