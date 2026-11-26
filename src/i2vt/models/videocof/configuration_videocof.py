from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from i2vt.config import ConfigMixin


@dataclass
class VideoCoFConfig(ConfigMixin):
    gradient_checkpointing: bool
    height: int
    lora_alpha: float
    lora_rank: int
    lora_target: str
    max_sequence_length: int
    cache_dir: str
    reasoning_latent_count: int
    rollout_guidance_scale: float
    rollout_steps: int
    rollout_timestep_max: int
    rollout_timestep_min: int
    rollout_timestep_mode: str
    rollout_train_steps: int
    # Shift of the training rollout schedule; None keeps the component config (5.0 in wan_civitai.yaml).
    rollout_shift: float | None
    cache_size: int
    student_guidance_scale: float
    use_cot_prompt: bool
    use_official_negative_prompt: bool
    component_config_path: str
    lora_path: str | None
    merge_lora_path: str | None
    component_root: str
    weight_dtype: str
    width: int
    resize_mode: str = "area"


def model_config(c) -> VideoCoFConfig:
    return VideoCoFConfig(
        gradient_checkpointing=c.student.gradient_checkpointing,
        height=c.data.height,
        lora_alpha=c.student.lora_alpha,
        lora_rank=c.student.lora_rank,
        lora_target=c.student.lora_target,
        max_sequence_length=c.student.max_sequence_length,
        cache_dir=c.run.cache_dir,
        reasoning_latent_count=c.student.model_kwargs["reasoning_latent_count"],
        rollout_guidance_scale=c.train.rollout.guidance_scale,
        rollout_steps=c.train.rollout.num_inference_steps,
        rollout_timestep_max=c.train.rollout.timestep_max,
        rollout_timestep_min=c.train.rollout.timestep_min,
        rollout_timestep_mode=c.train.rollout.timestep_mode,
        rollout_train_steps=c.train.rollout.train_steps,
        rollout_shift=c.train.rollout.kwargs["rollout_shift"],
        cache_size=c.student.cache_size,
        student_guidance_scale=c.train.rollout.student_guidance_scale,
        use_cot_prompt=c.student.model_kwargs["use_cot_prompt"],
        use_official_negative_prompt=c.train.rollout.kwargs["use_official_negative_prompt"],
        component_config_path=str(files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")),
        lora_path=c.student.lora_path,
        merge_lora_path=c.student.model_kwargs["merge_lora_path"],
        component_root=c.student.model_path,
        weight_dtype=c.student.weight_dtype,
        width=c.data.width,
        resize_mode=c.data.resize_mode,
    )
