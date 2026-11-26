from dataclasses import dataclass
from importlib.resources import files

from i2vt.config import ConfigMixin


@dataclass
class BerniniPipelineConfig(ConfigMixin):
    component_source: str
    model_path: str
    frame_sampling: str
    lora_path: str
    component_config_path: str
    weight_dtype: str
    device: str
    eta: float
    fps: int
    guidance_mode: str
    max_image_size: int
    component_root: str
    momentum: float
    norm_threshold: list[float]
    num_frames: int
    omega_img: float
    omega_scale: float
    omega_txt: float
    omega_vid: float
    seed: int
    shift: float
    source_frames: int
    num_inference_steps: int


def pipeline_config(c, device="cuda:0"):
    return BerniniPipelineConfig(
        component_source=c.student.model_kwargs["component_source"],
        model_path=c.student.model_path,
        frame_sampling=c.eval.inference.kwargs["frame_sampling"],
        lora_path=str(c.checkpoint_path) if c.checkpoint_path else "",
        component_config_path=str(files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")),
        device=device,
        weight_dtype=c.student.weight_dtype,
        eta=c.eval.inference.kwargs["eta"],
        fps=c.eval.inference.fps,
        guidance_mode=c.eval.inference.kwargs["guidance_mode"],
        max_image_size=c.eval.inference.kwargs["max_image_size"],
        component_root=c.student.model_kwargs["component_root"],
        momentum=c.eval.inference.kwargs["momentum"],
        norm_threshold=c.eval.inference.kwargs["norm_threshold"],
        num_frames=c.eval.inference.num_frames,
        omega_img=c.eval.inference.kwargs["omega_img"],
        omega_scale=c.eval.inference.kwargs["omega_scale"],
        omega_txt=c.eval.inference.kwargs["omega_txt"],
        omega_vid=c.eval.inference.kwargs["omega_vid"],
        seed=c.eval.inference.seed,
        shift=c.eval.inference.kwargs["shift"],
        source_frames=c.eval.inference.source_frames,
        num_inference_steps=c.eval.inference.num_inference_steps,
    )
