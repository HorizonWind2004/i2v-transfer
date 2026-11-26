from dataclasses import dataclass
from importlib.resources import files

from i2vt.config import ConfigMixin


@dataclass
class VideoCoFPipelineConfig(ConfigMixin):
    acceleration_lora_path: str
    component_config_path: str
    cot: bool
    weight_dtype: str
    device: str
    enable_acceleration_lora: bool
    fps: int
    guidance_scale: float
    max_video_side: int
    memory_mode: str
    component_root: str
    num_frames: int
    reasoning_frames: int
    seed: int
    shift: float
    source_frames: int
    num_inference_steps: int
    lora_path: str


def pipeline_config(c, device="cuda:0"):
    return VideoCoFPipelineConfig(
        acceleration_lora_path=c.eval.inference.kwargs["acceleration_lora_path"],
        component_config_path=str(files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")),
        cot=c.eval.inference.kwargs["cot"],
        device=device,
        weight_dtype=c.student.weight_dtype,
        enable_acceleration_lora=c.eval.inference.kwargs["enable_acceleration_lora"],
        fps=c.eval.inference.fps,
        guidance_scale=c.eval.inference.guidance_scale,
        max_video_side=c.eval.inference.kwargs["max_video_side"],
        memory_mode=c.eval.inference.memory_mode,
        component_root=c.student.model_path,
        num_frames=c.eval.inference.num_frames,
        reasoning_frames=c.eval.inference.kwargs["reasoning_frames"],
        seed=c.eval.inference.seed,
        shift=c.eval.inference.kwargs["shift"],
        source_frames=c.eval.inference.source_frames,
        num_inference_steps=c.eval.inference.num_inference_steps,
        lora_path=str(c.checkpoint_path) if c.checkpoint_path else "",
    )
