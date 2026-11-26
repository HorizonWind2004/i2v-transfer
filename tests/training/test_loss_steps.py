"""Loss on only some rollout steps (the high-noise or the low-noise end).

Three properties are checked:
  1. With "last:N" the first chunks hold only unsupervised steps, and training still reaches the
     supervised ones.
  2. The loss is normalised by the number of supervised steps, not the rollout length, so supervising
     half the steps does not halve the learning rate.
  3. The trajectory always runs all num_inference_steps steps, so "first:N" and "last:N" see the
     same states.
"""
from concurrent.futures import Future

import pytest

from i2vt.config import resolve_loss_steps


def test_all_is_every_step_and_is_the_default_behaviour():
    assert resolve_loss_steps("all", 10) == list(range(10))
    assert resolve_loss_steps("", 10) == list(range(10))
    assert resolve_loss_steps(None, 10) == list(range(10))


def test_first_takes_the_high_noise_end():
    # The scheduler's timesteps descend (1000 → 24), so a smaller index means more noise
    assert resolve_loss_steps("first:5", 10) == [0, 1, 2, 3, 4]


def test_last_takes_the_low_noise_end():
    assert resolve_loss_steps("last:5", 10) == [5, 6, 7, 8, 9]


def test_first_and_last_partition_the_trajectory():
    a, b = resolve_loss_steps("first:5", 10), resolve_loss_steps("last:5", 10)
    assert set(a) | set(b) == set(range(10))
    assert not set(a) & set(b)
    assert len(a) == len(b)          # same step count, so gradient scales match after normalisation


def test_bad_specs_are_rejected_at_config_time():
    for bad in ("middle:3", "first", "first:0", "last:11", "first:x"):
        with pytest.raises(ValueError):
            resolve_loss_steps(bad, 10)


def _drain(chunk_size, num_steps, spec):
    """Mirror the trainer's chunk loop and return the indices of the steps that are backpropagated.

    With last:5 at chunk_size=2 the first chunk (steps 0 and 1) holds no supervised step, and the loop
    must go on rather than stop there.
    """
    train_idx = set(resolve_loss_steps(spec, num_steps))
    traj = iter(range(num_steps))
    done, step_idx = [], 0
    while True:
        pending, consumed = [], 0
        for _ in range(chunk_size):
            try:
                next(traj)
            except StopIteration:
                break
            consumed += 1
            if step_idx not in train_idx:
                step_idx += 1
                continue
            pending.append(step_idx)
            step_idx += 1
        if not consumed:
            break
        if not pending:
            continue
        done += pending
    return done


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 10])
def test_last_half_still_trains_every_selected_step(chunk):
    assert _drain(chunk, 10, "last:5") == [5, 6, 7, 8, 9]


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 10])
def test_first_half_still_trains_every_selected_step(chunk):
    assert _drain(chunk, 10, "first:5") == [0, 1, 2, 3, 4]


def test_all_still_trains_everything():
    assert _drain(2, 10, "all") == list(range(10))


def test_shift_one_splits_the_noise_axis_in_half():
    """Only shift=1 spreads 10 steps evenly, so that first:5 / last:5 are exactly t>500 / t<=500.

    The default shift=5 puts 8 of 10 steps above t=500.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    s = FlowMatchEulerDiscreteScheduler(shift=1.0)
    s.set_timesteps(10)
    t = [float(x) for x in s.timesteps]
    hi = [t[i] for i in resolve_loss_steps("first:5", 10)]
    lo = [t[i] for i in resolve_loss_steps("last:5", 10)]
    assert all(x > 500 for x in hi), hi
    assert all(x <= 500 for x in lo), lo

    s5 = FlowMatchEulerDiscreteScheduler(shift=5.0)
    s5.set_timesteps(10)
    t5 = [float(x) for x in s5.timesteps]
    assert sum(x > 500 for x in t5) == 8


@pytest.mark.parametrize(
    "num_steps,train_steps,loss_steps", [(4, 0, "all"), (4, 2, "all"), (4, 0, "last:1"), (4, 2, "first:1")]
)
def test_loss_is_normalised_by_the_supervised_steps(monkeypatch, tmp_path, num_steps, train_steps, loss_steps):
    """Each supervised step has gradient 2 and the steps are averaged, so the total gradient is 2."""
    import torch
    from PIL import Image

    from i2vt import trainer
    from i2vt.models.modeling_utils import DistillationStep
    from tests.helpers import load_config

    weight = torch.nn.Parameter(torch.tensor(1.0))
    rollout_lengths = []

    class Student:
        def params(self):
            return [weight]

        def rollout(self, data, rollout):
            steps = rollout.train_steps or rollout.num_inference_steps
            rollout_lengths.append(steps)
            for i in range(steps):
                x = torch.zeros(1, 1, 1, 1, 1)
                yield DistillationStep(x, x + weight, torch.tensor(1000.0 - i), torch.tensor(1.0 - i / 10))

        def save(self, *args):
            pass

    class Teacher:
        prediction_type = "velocity"

        def submit(self, step, conditioning):
            future = Future()
            future.set_result(torch.zeros_like(step.noisy_latents))
            return future

    config = load_config()
    config.run.output_dir = str(tmp_path)
    config.train.max_steps = 1
    config.train.gradient_accumulation_steps = 1
    config.train.loss_type = "v"
    config.train.validation.steps = 0
    config.train.logging.mode = "disabled"
    config.train.check_vae_compat = False
    config.train.rollout.num_inference_steps = num_steps
    config.train.rollout.train_steps = train_steps
    config.train.rollout.loss_steps = loss_steps
    config.train.rollout.chunk_size = 1
    config.data.num_workers = 0
    data = [{"source": Image.new("RGB", (2, 2)), "video_edit_prompt": "edit", "joy_edit_prompt": "edit"}]
    gradients = []

    def clip(params, max_norm):
        gradients.append(float(weight.grad))
        return torch.tensor(0.0)

    for name in ("broadcast_params", "average_gradients"):
        monkeypatch.setattr(trainer, name, lambda *args, **kwargs: None)
    for name in ("average_scalar", "average_vector", "sum_vector"):
        monkeypatch.setattr(trainer, name, lambda value, *args: value)
    monkeypatch.setattr(trainer, "build_image_edit_dataset", lambda *args: data)
    monkeypatch.setattr(trainer.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer.torch.nn.utils, "clip_grad_norm_", clip)
    trainer.Trainer(config, Student(), teacher=Teacher()).train(0, 1, torch.device("cpu"), None)
    assert rollout_lengths == [train_steps or num_steps]
    assert gradients == [pytest.approx(2.0)]
