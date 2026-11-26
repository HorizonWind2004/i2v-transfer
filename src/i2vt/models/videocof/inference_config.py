from dataclasses import asdict, dataclass

from i2vt.config import _convert

INFERENCE_STEPS = 4
NUM_FRAMES = 33

# Verbatim from the official VideoCoF fast_infer.py (knightyxp/VideoCoF).
OFFICIAL_VIDEOCOF_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)


@dataclass
class ModelKwargs:
    merge_lora_path: str | None = None
    use_cot_prompt: bool = True
    reasoning_latent_count: int = 1


@dataclass
class RolloutKwargs:
    # Prompt of the unconditional CFG branch, as in official inference. It matters only when the
    # rollout or student CFG is above 1.
    use_official_negative_prompt: bool = True
    # Shift of the training rollout schedule; None keeps the component config (5.0 in wan_civitai.yaml).
    # 5.0 puts 6 of 10 steps above t=800, and 1.0 gives the uniform [1000, 889, ..., 1].
    rollout_shift: float | None = None
    # Shift of offline training's sigma draws. It must equal eval.inference.kwargs.shift, so training
    # covers the noise range inference visits.
    flow_shift: float = 5.0


@dataclass
class InferenceKwargs:
    enable_acceleration_lora: bool = True
    acceleration_lora_path: str | None = None
    """FusionX LoRA that VideoCoF uses for 4-step inference."""
    reasoning_frames: int = 1
    cot: bool = True
    shift: float = 5.0
    max_video_side: int = 0


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
    if inference.kwargs["enable_acceleration_lora"] and not inference.kwargs["acceleration_lora_path"]:
        raise ValueError("eval.inference.kwargs.acceleration_lora_path: required with enable_acceleration_lora")
    if c.train.objective == "offline":
        train_shift = float(c.train.rollout.kwargs["flow_shift"])
        if train_shift != float(inference.kwargs["shift"]):
            raise ValueError(
                f"train.rollout.kwargs.flow_shift={train_shift} must match eval.inference.kwargs.shift="
                f"{inference.kwargs['shift']}: offline training has to cover the noise levels inference visits"
            )
    if set(c.teacher.model_kwargs) != {"config_path"}:
        raise ValueError("teacher.model_kwargs: expected config_path")
    if c.teacher.model_kwargs["config_path"] is not None and not isinstance(
        c.teacher.model_kwargs["config_path"], str
    ):
        raise ValueError("teacher.model_kwargs.config_path: expected string or null")
