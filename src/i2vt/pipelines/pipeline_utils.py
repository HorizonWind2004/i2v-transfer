from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from i2vt.utils import dtype_from_name, resolve_model_path


def training_size(image_size, *, height: int, width: int, mode: str, multiple: int = 16) -> tuple[int, int]:
    """(width, height) a training frame is resized to.

    "area" keeps the source's aspect ratio at about height*width pixels, as evaluation does
    (sides are multiples of 16). "squash" resizes every frame to width x height.

    The student and both teachers call this with the source's size, so they see the same pixels.
    """
    if mode == "squash":
        return width, height
    if mode != "area":
        raise ValueError(f"unknown resize mode {mode!r}")
    src_w, src_h = image_size
    scale = (width * height / (src_w * src_h)) ** 0.5
    return (max(multiple, round(src_w * scale / multiple) * multiple),
            max(multiple, round(src_h * scale / multiple) * multiple))


def frame_to_image(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().float().cpu()
    if float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
        frame = (frame + 1.0) / 2.0
    frame = frame.clamp(0.0, 1.0)
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array)


def image_to_tensor(
    image: Image.Image, *, width: int, height: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = torch.from_numpy(np.asarray(image.convert("RGB"), dtype="float32"))
    return array.permute(2, 0, 1).div(255).mul(2).sub(1).unsqueeze(0).unsqueeze(2).to(device, dtype=dtype)


def load_video_frames(video_path: str, source_frames: int, max_video_side: int = 0):
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)

    stride = max(1, total_frames // source_frames)
    start_frame = (
        0
        if total_frames <= stride * source_frames
        else torch.randint(0, total_frames - stride * source_frames, (1,)).item()
    )
    frames = []
    original_width, original_height = None, None
    for i in range(source_frames):
        idx = start_frame + i * stride
        if idx >= total_frames:
            break
        frame = reader.get_data(idx)
        pil_frame = Image.fromarray(frame).convert("RGB")
        if original_width is None:
            original_width, original_height = pil_frame.size
        frames.append(pil_frame)
    reader.close()

    while len(frames) < source_frames:
        frames.append(frames[-1].copy() if frames else Image.new("RGB", (832, 480), (0, 0, 0)))

    if max_video_side > 0 and max(original_width, original_height) > max_video_side:
        scale = max_video_side / max(original_width, original_height)
        original_width = max(16, round(original_width * scale / 16) * 16)
        original_height = max(16, round(original_height * scale / 16) * 16)
        frames = [frame.resize((original_width, original_height), Image.Resampling.LANCZOS) for frame in frames]

    input_video = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).unsqueeze(0).float()
    input_video = input_video * (2.0 / 255.0) - 1.0
    return input_video, original_height, original_width


def load_wan_components(
    *,
    component_root: str,
    component_config_path: str,
    weight_dtype: str,
):
    """Load the Wan tokenizer, text encoder, and VAE shared by Wan pipelines."""
    from transformers import AutoTokenizer

    from i2vt.vendor.wan.models.wan_text_encoder import WanT5EncoderModel
    from i2vt.vendor.wan.models.wan_vae import AutoencoderKLWan

    config = OmegaConf.load(component_config_path)
    dtype = dtype_from_name(weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(
            component_root,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            "tokenizer",
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        resolve_model_path(
            component_root,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            "text_encoder",
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        resolve_model_path(component_root, config["vae_kwargs"].get("vae_subpath", "vae"), "vae"),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )
    return config, tokenizer, text_encoder, vae


def normalize_to_01(video: torch.Tensor) -> torch.Tensor:
    if float(video.min()) < 0.0 or float(video.max()) > 1.0:
        video = (video + 1.0) / 2.0
    return video.clamp(0.0, 1.0)


@dataclass
class VideoEditPipelineOutput:
    frames: Any
    fps: int
    metadata: dict
    source_frames: Any = None
    reasoning_frames: Any = None
