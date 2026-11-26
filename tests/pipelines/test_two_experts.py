"""Two-expert Bernini-R (14B): which expert runs each step, and which LoRA each expert gets."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from einops import rearrange

from i2vt.models.bernini.modeling_bernini import BerniniStudentBackend
from i2vt.pipelines.bernini import pipeline_bernini as bernini
from i2vt.vendor.bernini.models.wan_diffusion import GEN_Wanx22
from i2vt.vendor.wan.utils.lora_utils import create_network


def two_expert_checkpoint(root, boundary=0.875):
    (root / "transformer").mkdir(parents=True)
    (root / "transformer_2").mkdir()
    (root / "config.json").write_text(json.dumps({"switch_dit_boundary": boundary}))
    return root


class Expert(nn.Module):
    """Stands in for WanTransformer3DModel inside the sampler and records each call's timestep."""

    def __init__(self, name, calls):
        super().__init__()
        self.name, self.calls = name, calls
        self.config = SimpleNamespace(in_channels=4)
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def dtype(self):
        return torch.float32

    def patch_vae_latent(self, latent, source_id):
        tokens = rearrange(latent, "b c t (h ph) (w pw) -> b (t h w) (ph pw c)", ph=2, pw=2)
        return tokens, torch.zeros(1, 1, tokens.shape[1], 2)

    def forward(self, hidden_states, timestep, **_kwargs):
        self.calls.append((self.name, float(timestep)))
        return SimpleNamespace(sample=torch.zeros_like(hidden_states))


def sampled_experts(boundary, steps):
    calls = []
    sampler = GEN_Wanx22.__new__(GEN_Wanx22)
    nn.Module.__init__(sampler)
    sampler.config = SimpleNamespace(interpolate_src_id=True, max_trained_src_id=5)
    sampler.switch_dit_boundary = boundary
    sampler.transformer = Expert("transformer", calls)
    sampler.transformer_2 = Expert("transformer_2", calls)
    sampler.use_unipc = True
    sampler.scheduler = bernini._make_unipc_scheduler(5.0)
    sampler.vae_scale_factor_temporal, sampler.vae_scale_factor_spatial = 4, 8
    text = torch.zeros(1, 3, 8)
    sampler.sample(
        prompt_embeds=text,
        uncond_prompt_embeds=text,
        num_frames=5,
        width=32,
        height=32,
        multi_video_vae_latents=[torch.zeros(1, 4, 2, 4, 4)],
        num_inference_steps=steps,
        guidance_mode="v2v",
        device="cpu",
    )
    return calls


def test_expert_boundary_comes_from_the_checkpoint_config(tmp_path):
    assert bernini.expert_boundary(two_expert_checkpoint(tmp_path / "a", 0.9)) == 0.9
    (tmp_path / "one" / "transformer").mkdir(parents=True)
    assert bernini.expert_boundary(tmp_path / "one") is None


@pytest.mark.parametrize("steps", [20, 40])
def test_sampler_switches_experts_where_training_does(tmp_path, steps):
    boundary = bernini.expert_boundary(two_expert_checkpoint(tmp_path / "ckpt"))
    calls = sampled_experts(boundary, steps)
    student = BerniniStudentBackend.__new__(BerniniStudentBackend)
    student.transformer, student.transformer_2 = "transformer", "transformer_2"
    student.switch_boundary = boundary * 1000.0
    # v2v runs two forwards per step, both on the step's expert.
    assert len(calls) == 2 * steps
    assert all(name == student._expert_for(torch.tensor(t)) for name, t in calls)
    high = [t for name, t in calls if name == "transformer"]
    low = [t for name, t in calls if name == "transformer_2"]
    assert high and low and min(high) >= 875 > max(low)
    # Once switched, the sampler never returns to the high-noise expert.
    names = [name for name, _ in calls]
    assert names == sorted(names, key=lambda n: n == "transformer_2")


class WanTransformer3DModel(nn.Module):
    """Named like the real model so the trainer's LoRA network targets its linear layers."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.ModuleDict({"to_q": nn.Linear(8, 8), "ffn": nn.Linear(8, 8)}) for _ in range(2))
        self.proj_out = nn.Linear(8, 8)

    def forward(self, x):
        for block in self.blocks:
            x = block["ffn"](torch.tanh(block["to_q"](x)))
        return self.proj_out(x)


def trained_lora(model, path, seed):
    """A LoRA in the trainer's format (rank 4, alpha 2), returned with the model it was applied to."""
    torch.manual_seed(seed)
    hooked = copy.deepcopy(model)
    network = create_network(1.0, 4, 2.0, None, hooked)
    network.apply_to(None, hooked, False, True)
    for param in network.parameters():
        nn.init.normal_(param, std=0.2)
    network.save_weights(str(path), torch.float32, {})
    return hooked


def test_merged_lora_matches_the_trainer_lora(tmp_path):
    base = WanTransformer3DModel()
    hooked = trained_lora(base, tmp_path / "bernini_lora_step_200.safetensors", seed=0)
    merged = copy.deepcopy(base)
    assert bernini.merge_lora(merged, tmp_path / "bernini_lora_step_200.safetensors") == 5
    x = torch.randn(3, 8)
    torch.testing.assert_close(merged(x), hooked(x), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(merged(x), base(x))


def test_each_expert_gets_its_own_lora(tmp_path):
    base = WanTransformer3DModel()
    high = trained_lora(base, tmp_path / "bernini_lora_step_200.safetensors", seed=1)
    low = trained_lora(base, tmp_path / "bernini_lora_low_step_200.safetensors", seed=2)
    diff_dec = SimpleNamespace(transformer=copy.deepcopy(base), transformer_2=copy.deepcopy(base))
    paths = bernini.merge_expert_loras(diff_dec, tmp_path / "bernini_lora_step_200.safetensors")
    assert paths == {
        "transformer": tmp_path / "bernini_lora_step_200.safetensors",
        "transformer_2": tmp_path / "bernini_lora_low_step_200.safetensors",
    }
    x = torch.randn(3, 8)
    torch.testing.assert_close(diff_dec.transformer(x), high(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(diff_dec.transformer_2(x), low(x), rtol=1e-5, atol=1e-5)


def test_missing_low_noise_lora_is_refused_before_anything_is_merged(tmp_path):
    base = WanTransformer3DModel()
    trained_lora(base, tmp_path / "bernini_lora_step_200.safetensors", seed=1)
    diff_dec = SimpleNamespace(transformer=copy.deepcopy(base), transformer_2=copy.deepcopy(base))
    with pytest.raises(FileNotFoundError, match="transformer_2"):
        bernini.merge_expert_loras(diff_dec, tmp_path / "bernini_lora_step_200.safetensors")
    x = torch.randn(3, 8)
    torch.testing.assert_close(diff_dec.transformer(x), base(x))
    with pytest.raises(ValueError, match="high-noise"):
        bernini.low_noise_lora_path(tmp_path / "bernini_lora_low_step_200.safetensors")


def test_a_lora_for_another_architecture_is_refused(tmp_path):
    trained_lora(WanTransformer3DModel(), tmp_path / "lora.safetensors", seed=0)
    other = nn.Sequential(nn.Linear(8, 8))
    with pytest.raises(ValueError, match="matches no weight"):
        bernini.merge_lora(other, tmp_path / "lora.safetensors")
