from __future__ import annotations

from dataclasses import dataclass

from i2vt.config import ConfigMixin


@dataclass
class JoyEditConfig(ConfigMixin):
    model_path: str
    config_path: str | None
    cache_dir: str
    rollout_steps: int
    seed: int
    cache_size: int
    guidance_scale: float
    guidance_scale_by_type: dict[str, float] | None = None
    guidance_scale_by_task: dict[str, float] | None = None


def model_config(c) -> JoyEditConfig:
    return JoyEditConfig(
        model_path=c.teacher.model_path,
        config_path=c.teacher.model_kwargs["config_path"],
        cache_dir=c.run.cache_dir,
        rollout_steps=c.train.rollout.num_inference_steps,
        seed=c.train.seed,
        cache_size=c.teacher.cache_size,
        guidance_scale=c.teacher.guidance_scale,
        # Without these two tables the teacher's getattr returns None and per-type CFG falls back to the scalar.
        guidance_scale_by_type=c.teacher.guidance_scale_by_type,
        guidance_scale_by_task=c.teacher.guidance_scale_by_task,
    )
