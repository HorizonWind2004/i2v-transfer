"""Deterministic small-network training run for the numeric regression tests.

The real backend packing, guidance, scheduler and Trainer execute; the pretrained networks, the
dataset, the teacher and the distributed collectives are replaced by small stand-ins.
"""

import contextlib
import copy
import importlib
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image


def run_case(config, output_dir):
    from i2vt import trainer
    from i2vt.utils import InMemoryDiskCache

    backend = config.student.backend
    mod = importlib.import_module(f"i2vt.models.{backend}.modeling_{backend}")
    classname = "BerniniStudentBackend" if backend == "bernini" else "VideoCoFStudentBackend"
    model = getattr(mod, classname).__new__(getattr(mod, classname))
    model.device = torch.device("cpu")
    model.weight_dtype = torch.float32
    model.config = SimpleNamespace(
        width=2,
        height=2,
        # The reference traces use frames squashed to width x height; area-preserving resize is
        # covered by its own tests.
        resize_mode="squash",
        component_root="test",
        component_config_path="test",
        reasoning_latent_count=1,
        rollout_guidance_scale=config.train.rollout.guidance_scale,
        student_guidance_scale=config.train.rollout.student_guidance_scale,
    )
    trace = {"steps": [], "teachers": [], "gradients": [], "weights": [], "losses": [], "prompts": []}

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.11, -0.07]))
            self.config = SimpleNamespace(patch_size=(1, 1, 1))

        def patch_vae_latent(self, x, source_id=None):
            tokens = x.flatten(2).transpose(1, 2)
            return tokens, torch.zeros(1, 1, tokens.shape[1], 2)

        def forward(self, **kw):
            if "x" in kw:
                x = kw["x"]
                cond = kw["context"].mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
                return x * self.weight.view(1, 2, 1, 1, 1) + cond * 0.01 + kw["t"].view(-1, 1, 1, 1, 1) * 0.0001
            x = kw["hidden_states"]
            return SimpleNamespace(
                sample=x * self.weight
                + kw["encoder_hidden_states"].mean() * 0.01
                + kw["timestep"].reshape(1, 1, 1) * 0.0001
            )

    class Scheduler(FlowMatchEulerDiscreteScheduler):
        def step(self, prediction, timestep, sample, **kw):
            sigma = self.sigmas[
                self.step_index if self.step_index is not None else self.index_for_timestep(timestep)
            ].clone()
            result = super().step(prediction, timestep, sample, **kw)
            after = result[0]
            trace["steps"].append(
                {
                    "sigma": sigma,
                    "t": timestep.detach().clone(),
                    "before": sample.detach().clone(),
                    "rollout_prediction": prediction.detach().clone(),
                    "after": after.detach().clone(),
                }
            )
            return result

    model.network = model.transformer = Network()
    model.network.weight.register_hook(
        lambda grad: trace.setdefault("backward_gradients", []).append(grad.detach().clone())
    )
    model.vae = SimpleNamespace(latent_channels=2)
    model.scheduler = Scheduler(num_train_timesteps=1000, shift=3)

    def prompt(text):
        trace["prompts"].append(text)
        if backend == "bernini":
            return (torch.ones(1, 3, 2), torch.zeros(1, 3, 2))
        use_cfg = config.train.rollout.guidance_scale > 1 or config.train.rollout.student_guidance_scale > 1
        return torch.tensor([0.0, 1.0]).view(2, 1, 1).expand(2, 3, 2) if use_cfg else torch.ones(1, 3, 2)

    model._encode_prompt = prompt
    source = torch.arange(8).reshape(1, 2, 1, 2, 2).float() / 10
    model._source_cache = model._cache = InMemoryDiskCache(
        namespace="test", root=None, cache_size=2, device=model.device, dtype=model.weight_dtype
    )
    pipeline_class = mod.BerniniVideoEditPipeline if backend == "bernini" else mod.VideoCoFVideoEditPipeline
    student = model
    original_velocity = student._velocity

    def velocity(*args, **kwargs):
        result = original_velocity(*args, **kwargs)
        loss_prediction = result[1] if isinstance(result, tuple) else result
        trace.setdefault("loss_predictions", []).append(loss_prediction.detach().clone())
        return result

    student._velocity = velocity
    model.network.save_weights = lambda path, dtype, metadata: trace.setdefault("checkpoints", []).append(
        {"filename": Path(path).name, "dtype": str(dtype), "metadata": metadata}
    )

    class Teacher:
        prediction_type = "velocity"

        def submit(self, step, conditioning):
            latent = step.noisy_latents
            timestep = step.timestep.float().reshape(1)
            prompt = conditioning["joy_edit_prompt"]
            target = latent.detach() * 0.23 - 0.17
            trace["teachers"].append(
                {
                    "latent": latent.detach().clone(),
                    "t": timestep.detach().clone(),
                    "target": target.clone(),
                    "prompt": prompt,
                }
            )
            f = Future()
            f.set_result(target)
            return f

        def close(self):
            pass

    original_optimizer = torch.optim.AdamW

    class Optimizer(original_optimizer):
        def step(self, *args, **kwargs):
            trace["gradients"].append(model.network.weight.grad.detach().clone())
            result = super().step(*args, **kwargs)
            trace["weights"].append(model.network.weight.detach().clone())
            return result

    config.run.output_dir = str(output_dir)
    config.train.max_steps = 1
    config.train.validation.steps = 0
    config.train.logging.mode = "disabled"
    config.train.check_vae_compat = False
    config.data.num_workers = 0
    config.data.resize_mode = "squash"
    data = [
        {
            "source": Image.new("RGB", (2, 2)),
            "video_edit_prompt": "student instruction",
            "joy_edit_prompt": "teacher instruction",
        }
    ]

    def scalar(value, *args):
        trace["losses"].append(value.detach().clone())
        return value

    def vector(value, *args):
        if "timestep_losses" not in trace:
            trace["timestep_losses"] = value.detach().clone()
        return value

    # average_scalar/average_vector record the trace. The other reductions only have to stay out of
    # NCCL; scanning by suffix covers any new *_scalar or *_vector reduction too.
    traced = {"average_scalar", "average_vector"}
    extra = [
        patch.object(trainer, name, side_effect=lambda v, *a, **k: v)
        for name in dir(trainer)
        if name.endswith(("_scalar", "_vector")) and name not in traced
    ]
    with (
        contextlib.ExitStack() as stack,
        patch.object(pipeline_class, "image_to_latents", return_value=source),
        patch.object(trainer, "build_image_edit_dataset", return_value=data),
        patch.object(trainer, "broadcast_params"),
        patch.object(trainer, "average_gradients"),
        patch.object(trainer, "average_scalar", side_effect=scalar),
        patch.object(trainer, "average_vector", side_effect=vector),
        patch.object(trainer.dist, "barrier"),
        patch.object(torch.optim, "AdamW", Optimizer),
    ):
        for cm in extra:
            stack.enter_context(cm)
        torch.manual_seed(1234)
        trainer.Trainer(config, student, teacher=Teacher()).train(0, 1, torch.device("cpu"), None)
    # Also compare decoded preview bytes for the default equal-guidance path.
    frozen = copy.deepcopy(trace)
    if config.train.rollout.guidance_scale == config.train.rollout.student_guidance_scale:
        from i2vt.pipelines import training_preview

        model.config.width = model.config.height = 2
        model.config.rollout_steps = config.train.rollout.num_inference_steps
        model.config.rollout_train_steps = 0
        model.config.rollout_timestep_mode = config.train.rollout.timestep_mode
        model.config.rollout_timestep_min = config.train.rollout.timestep_min
        model.config.rollout_timestep_max = config.train.rollout.timestep_max
        model.vae.decode = lambda x, **k: (torch.cat([x, torch.zeros_like(x[:, :1])], dim=1),)
        with patch.object(pipeline_class, "image_to_latents", return_value=source), torch.no_grad():
            images = training_preview(student, data[0]["source"], "preview", seed=5678)
        frozen["preview_pixels"] = [list(image.tobytes()) for image in images]
    return frozen
