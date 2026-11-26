from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from i2vt.config import ConfigMixin


@dataclass
class BerniniConfig(ConfigMixin):
    component_root: str
    flow_shift: float
    lora_path: str | None
    tune_mode: str
    model_path: str
    gradient_checkpointing: bool
    height: int
    lora_alpha: float
    lora_rank: int
    lora_target: str
    max_sequence_length: int
    rollout_guidance_scale: float
    rollout_steps: int
    rollout_timestep_max: int
    rollout_timestep_min: int
    rollout_timestep_mode: str
    rollout_train_steps: int
    cache_size: int
    component_config_path: str
    weight_dtype: str
    width: int
    lora_path_2: str | None = None
    resize_mode: str = "area"


def model_config(c) -> BerniniConfig:
    return BerniniConfig(
        component_root=c.student.model_kwargs["component_root"],
        flow_shift=c.train.rollout.kwargs["flow_shift"],
        lora_path=c.student.lora_path,
        tune_mode=c.student.tune_mode,
        model_path=c.student.model_path,
        gradient_checkpointing=c.student.gradient_checkpointing,
        height=c.data.height,
        lora_alpha=c.student.lora_alpha,
        lora_rank=c.student.lora_rank,
        lora_target=c.student.lora_target,
        max_sequence_length=c.student.max_sequence_length,
        rollout_guidance_scale=c.train.rollout.guidance_scale,
        rollout_steps=c.train.rollout.num_inference_steps,
        rollout_timestep_max=c.train.rollout.timestep_max,
        rollout_timestep_min=c.train.rollout.timestep_min,
        rollout_timestep_mode=c.train.rollout.timestep_mode,
        rollout_train_steps=c.train.rollout.train_steps,
        cache_size=c.student.cache_size,
        component_config_path=str(files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")),
        weight_dtype=c.student.weight_dtype,
        width=c.data.width,
        lora_path_2=c.student.lora_path_2,
        resize_mode=c.data.resize_mode,
    )
