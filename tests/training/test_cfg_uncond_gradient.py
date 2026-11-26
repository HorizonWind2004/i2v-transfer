"""The unconditional branch of CFG: its gradient, and the no_grad shortcut at student CFG 1.

With rollout_cfg > 1 and student_cfg == 1.0 the unconditional prediction enters the loss with
coefficient 0, so it runs without a graph.
"""

import types

import torch

from i2vt.models.videocof.modeling_videocof import VideoCoFStudentBackend


class Tiny(torch.nn.Module):
    """The prediction depends on both x and context, so that cond and uncond differ."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, *, x, context, t, seq_len, frame_split_indices, ground_frame_indices):
        # context has shape (batch, tokens, dim), and its mean serves as the conditioning strength
        scale = context.float().mean()
        return self.weight * x * (1.0 + scale)


def make_backend(rollout_cfg, student_cfg):
    model = VideoCoFStudentBackend.__new__(VideoCoFStudentBackend)
    model.transformer = Tiny()
    model.transformer.config = types.SimpleNamespace(patch_size=(1, 1, 1))
    model.vae = types.SimpleNamespace(latent_channels=2)
    model.scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
    model.weight_dtype = torch.float32
    model.config = types.SimpleNamespace(
        rollout_guidance_scale=rollout_cfg, student_guidance_scale=student_cfg
    )
    return model


def call(model, latents, prompt, rollout_cfg, student_cfg):
    context = types.SimpleNamespace(source_count=0, edit_start=0)
    return model._velocity(
        latents,
        torch.tensor(500.0),
        prompt,
        context,
        rollout_guidance_scale=rollout_cfg,
        loss_guidance_scale=student_cfg,
    )


def fixtures():
    latents = torch.ones(1, 2, 2, 2, 2)
    # index 0 = unconditional (negative), index 1 = conditional (positive), matching the
    # [negative, positive] order that the pipeline's encode_prompt returns
    prompt = torch.stack([torch.zeros(3, 2), torch.ones(3, 2)])
    return latents, prompt


def test_returns_tuple_when_scales_differ():
    model = make_backend(2.0, 1.0)
    latents, prompt = fixtures()
    out = call(model, latents, prompt, 2.0, 1.0)
    assert isinstance(out, tuple) and len(out) == 2


def test_loss_prediction_equals_cond_when_student_cfg_is_one():
    """At student_cfg = 1 the loss prediction is (1-1)*uncond + 1*cond, which is cond."""
    model = make_backend(2.0, 1.0)
    latents, prompt = fixtures()
    _, loss_pred = call(model, latents, prompt, 2.0, 1.0)
    cond_only = model.transformer(
        x=latents, context=prompt[1:], t=torch.tensor(500.0),
        seq_len=1, frame_split_indices=[0], ground_frame_indices=None,
    )
    assert torch.allclose(loss_pred, cond_only, atol=1e-6)


def test_uncond_gradient_is_exactly_zero_at_student_cfg_one():
    """Why the no_grad optimisation is valid: uncond enters the loss gradient with coefficient 1 - 1.0 = 0.

    Compare with the weight gradient from a cond-only forward pass. The two must match exactly,
    otherwise skipping uncond under no_grad has changed the gradient.
    """
    latents, prompt = fixtures()

    model = make_backend(2.0, 1.0)
    _, loss_pred = call(model, latents, prompt, 2.0, 1.0)
    loss_pred.sum().backward()
    got = model.transformer.weight.grad.clone()

    ref_net = Tiny()
    cond = ref_net(
        x=latents, context=prompt[1:], t=torch.tensor(500.0),
        seq_len=1, frame_split_indices=[0], ground_frame_indices=None,
    )
    cond.sum().backward()

    assert torch.allclose(got, ref_net.weight.grad, atol=1e-6), (
        f"the uncond branch leaked gradient: {got} vs cond only {ref_net.weight.grad}"
    )


def test_uncond_keeps_gradient_when_student_cfg_not_one():
    """At student_cfg ≠ 1 the uncond coefficient is 1 - student_cfg ≠ 0, so it must build a graph."""
    latents, prompt = fixtures()
    model = make_backend(2.5, 1.5)
    _, loss_pred = call(model, latents, prompt, 2.5, 1.5)
    loss_pred.sum().backward()
    got = model.transformer.weight.grad.clone()

    ref_net = Tiny()
    u = ref_net(x=latents, context=prompt[:1], t=torch.tensor(500.0),
                seq_len=1, frame_split_indices=[0], ground_frame_indices=None)
    c = ref_net(x=latents, context=prompt[1:], t=torch.tensor(500.0),
                seq_len=1, frame_split_indices=[0], ground_frame_indices=None)
    (u + 1.5 * (c - u)).sum().backward()
    assert torch.allclose(got, ref_net.weight.grad, atol=1e-6)


def test_no_cfg_path_unchanged():
    model = make_backend(1.0, 1.0)
    latents = torch.ones(1, 2, 2, 2, 2)
    out = call(model, latents, torch.ones(1, 3, 2), 1.0, 1.0)
    assert not isinstance(out, tuple)


def _grad_states(rollout_cfg, student_cfg):
    """Record whether grad is enabled at each transformer forward pass.

    The zero-gradient test above passes with or without no_grad (the gradient is mathematically zero),
    so it shows that the optimisation is safe, not that it is in effect. This one checks the latter.
    """
    latents, prompt = fixtures()
    model = make_backend(rollout_cfg, student_cfg)
    states = []
    inner = model.transformer.forward

    def spy(**kwargs):
        states.append(torch.is_grad_enabled())
        return inner(**kwargs)

    model.transformer.forward = spy
    call(model, latents, prompt, rollout_cfg, student_cfg)
    return states


def test_uncond_forward_runs_under_no_grad_at_student_cfg_one():
    # Order: uncond first, then cond
    assert _grad_states(2.0, 1.0) == [False, True]


def test_both_forwards_keep_grad_when_student_cfg_not_one():
    assert _grad_states(2.5, 1.5) == [True, True]
