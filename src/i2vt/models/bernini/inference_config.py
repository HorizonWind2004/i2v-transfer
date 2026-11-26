import html
import re
from dataclasses import asdict, dataclass, field

from i2vt.config import _convert

# Official bytedance/Bernini: SYSTEM_PROMPTS["v2v"] in bernini/prompt_enhancer.py,
# picked by resolve_system_prompt() for task_type v2v (the CLI default).
V2V_SYSTEM_PROMPT = "You are a helpful assistant specialized in video editing."


def clean_prompt(text: str) -> str:
    """bernini/pipeline.py::_prompt_clean, byte for byte."""
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def v2v_model_prompt(instruction: str) -> str:
    """The text Bernini's T5 reads for a v2v edit.

    Official inference builds it as ``system_prompt + _prompt_clean(prompt)``
    (bernini/pipeline.py). Training and evaluation both call this, so the
    student is never trained on a condition it is not evaluated under.
    """
    return V2V_SYSTEM_PROMPT + clean_prompt(instruction)


# Wan's official negative prompt (model input, kept verbatim).
OFFICIAL_WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


INFERENCE_STEPS = 40
NUM_FRAMES = 81


@dataclass
class ModelKwargs:
    component_root: str
    """Wan2.1-T2V-14B directory; the T5 text encoder and the VAE are read from it."""
    component_source: str = "official"


@dataclass
class RolloutKwargs:
    flow_shift: float = 5.0


@dataclass
class InferenceKwargs:
    """Official Bernini-R v2v settings."""

    guidance_mode: str = "v2v_apg"
    omega_vid: float = 1.25
    omega_img: float = 4.5
    omega_txt: float = 4.0
    omega_scale: float = 0.8
    eta: float = 0.5
    momentum: float = 0.0
    norm_threshold: list[float] = field(default_factory=lambda: [50.0, 50.0, 50.0])
    shift: float = 5.0
    max_image_size: int = 848
    frame_sampling: str = "official"


def validate(c):
    c.student.model_kwargs = asdict(_convert(c.student.model_kwargs, ModelKwargs, "student.model_kwargs"))
    c.train.rollout.kwargs = asdict(_convert(c.train.rollout.kwargs, RolloutKwargs, "train.rollout.kwargs"))
    c.eval.inference.kwargs = asdict(_convert(c.eval.inference.kwargs, InferenceKwargs, "eval.inference.kwargs"))
    inference = c.eval.inference
    if inference.num_inference_steps is None:
        inference.num_inference_steps = INFERENCE_STEPS
    if inference.num_frames is None:
        inference.num_frames = NUM_FRAMES
    if inference.source_frames is None:
        inference.source_frames = inference.num_frames
    if set(c.teacher.model_kwargs) != {"config_path"}:
        raise ValueError("teacher.model_kwargs: expected config_path")
    if c.teacher.model_kwargs["config_path"] is not None and not isinstance(
        c.teacher.model_kwargs["config_path"], str
    ):
        raise ValueError("teacher.model_kwargs.config_path: expected string or null")

    if c.student.model_kwargs["component_source"] not in {"official", "reused"}:
        raise ValueError("student.model_kwargs.component_source: unsupported source")
    if c.eval.inference.kwargs["frame_sampling"] not in {"official", "exact"}:
        raise ValueError("eval.inference.kwargs.frame_sampling: unsupported sampler")
    if c.eval.inference.num_frames != c.eval.inference.source_frames or c.eval.inference.num_frames % 4 != 1:
        raise ValueError("Bernini frame caps must match and be 4n+1")
    if c.eval.inference.guidance_scale != 1.0:
        raise ValueError("Bernini APG uses inference.kwargs omega parameters; guidance_scale must remain 1.0")
    if c.eval.inference.memory_mode != "full":
        raise ValueError(
            "Bernini pipeline manages staged device transfers internally; other memory modes are unsupported"
        )
    if len(c.eval.inference.kwargs["norm_threshold"]) != 3:
        raise ValueError("Bernini norm_threshold must have three values")
