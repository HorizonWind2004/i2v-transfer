from __future__ import annotations

import torch
from omegaconf import OmegaConf

# The negative prompt of the official VideoCoF fast_infer.py; it takes effect only with CFG above 1.
from i2vt.models.videocof.inference_config import OFFICIAL_VIDEOCOF_NEGATIVE_PROMPT as NEGATIVE_PROMPT
from i2vt.pipelines.pipeline_utils import (
    VideoEditPipelineOutput,
    image_to_tensor,
    load_video_frames,
    load_wan_components,
)
from i2vt.utils import dtype_from_name, filter_kwargs, resolve_model_path

from .configuration_videocof import VideoCoFPipelineConfig, pipeline_config


def load_videocof_pipeline(
    *,
    component_root: str,
    component_config_path: str,
    weight_dtype: str,
    memory_mode: str,
    enable_acceleration_lora: bool,
    acceleration_lora_path: str | None,
    lora_path: str | None,
    seed: int,
    device: str,
):
    """Load shared VideoCoF components and merge existing-format LoRA weights once."""
    from i2vt.vendor.wan.models.wan_transformer3d import WanTransformer3DModel
    from i2vt.vendor.wan.pipeline.pipeline_wan import WanPipeline
    from i2vt.vendor.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from i2vt.vendor.wan.utils.lora_utils import merge_lora

    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config, tokenizer, text_encoder, vae = load_wan_components(
        component_root=component_root,
        component_config_path=component_config_path,
        weight_dtype=weight_dtype,
    )
    dtype = dtype_from_name(weight_dtype)

    transformer = WanTransformer3DModel.from_pretrained(
        resolve_model_path(
            component_root,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            "transformer",
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        # Building on meta and filling from the checkpoint leaves anything the checkpoint does not
        # carry (freqs among them) on meta, and the move to the device then fails with "cannot copy
        # out of meta tensor". A 14B model in bf16 takes about 28 GB of host memory this way.
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    )
    vae = vae.to(dtype)

    scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    scheduler_kwargs["shift"] = 1
    scheduler = FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler, scheduler_kwargs))
    pipeline = WanPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler
    )

    if memory_mode == "sequential_cpu_offload":
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if enable_acceleration_lora:
        if not acceleration_lora_path:
            raise ValueError("Set --acceleration-lora-path or use --no-enable-acceleration-lora.")
        pipeline = merge_lora(pipeline, acceleration_lora_path, multiplier=1.0, device=device)
        print(f"Loaded acceleration LoRA: {acceleration_lora_path}", flush=True)
    if lora_path:
        pipeline = merge_lora(pipeline, lora_path, multiplier=1.0, device=device)
        print(f"Loaded VideoCoF LoRA: {lora_path}", flush=True)
    return pipeline, device


class VideoCoFVideoEditPipeline:
    @staticmethod
    def format_prompt(prompt: str, *, use_cot: bool) -> str:
        if not use_cot:
            return f"A video sequence showing the same scene but {prompt}"
        ground = prompt.strip().rstrip(".")
        for prefix in ("remove ", "delete ", "erase ", "replace ", "change "):
            if ground.lower().startswith(prefix):
                ground = ground[len(prefix) :]
                break
        return (
            "A video sequence showing three parts: first the original scene, "
            f"then grounded {ground}, and finally the same scene but {prompt}"
        )

    @staticmethod
    def encode_prompt(pipeline, prompt: str, *, max_length: int, do_cfg: bool, device, negative_prompt: str = ""):
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_length,
            device=device,
        )
        return negative_prompt_embeds + prompt_embeds if do_cfg else prompt_embeds

    @staticmethod
    def image_to_latents(vae, image, *, width: int, height: int, device, dtype):
        image_tensor = image_to_tensor(image, width=width, height=height, device=device, dtype=dtype)
        return vae.encode(image_tensor)[0].mode().detach().to(device=device, dtype=dtype)

    @staticmethod
    def initial_latents(source_latents, reasoning_latent_count: int):
        reasoning = torch.randn(
            source_latents.shape[0],
            source_latents.shape[1],
            reasoning_latent_count,
            source_latents.shape[3],
            source_latents.shape[4],
            device=source_latents.device,
            dtype=source_latents.dtype,
        )
        edit_start = source_latents.shape[2] + reasoning_latent_count
        return torch.cat([source_latents, reasoning, torch.randn_like(source_latents)], dim=2), edit_start

    def __init__(self, config: VideoCoFPipelineConfig):
        self.config = config
        self.pipeline, self.device = load_videocof_pipeline(
            component_config_path=config.component_config_path,
            weight_dtype=config.weight_dtype,
            component_root=config.component_root,
            memory_mode=config.memory_mode,
            enable_acceleration_lora=config.enable_acceleration_lora,
            acceleration_lora_path=config.acceleration_lora_path,
            lora_path=config.lora_path,
            seed=config.seed,
            device=config.device,
        )

    def __call__(self, video: str, prompt: str) -> VideoEditPipelineOutput:
        video_path = video
        config = self.config
        input_video, video_height, video_width = load_video_frames(
            video_path, config.source_frames, config.max_video_side
        )
        formatted_prompt = self.format_prompt(prompt, use_cot=config.cot)

        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        with torch.no_grad():
            sample = self.pipeline(
                video=input_video,
                prompt=formatted_prompt,
                num_frames=config.num_frames,
                source_frames=config.source_frames,
                reasoning_frames=config.reasoning_frames,
                negative_prompt=NEGATIVE_PROMPT,
                height=video_height,
                width=video_width,
                generator=generator,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                shift=config.shift,
                repeat_rope=True,
                cot=config.cot,
                callback_on_step_end=None,
            ).videos

        return VideoEditPipelineOutput(
            frames=sample[:, :, -config.source_frames :, :, :].detach().cpu(),
            fps=config.fps,
            metadata={
                "prompt": prompt,
                "model_prompt": formatted_prompt,
                "backend": "videocof",
                "lora_path": config.lora_path,
            },
            source_frames=input_video.detach().cpu(),
            reasoning_frames=sample.detach().cpu(),
        )

    @classmethod
    def from_config(cls, config, device="cuda:0"):
        return cls(pipeline_config(config, device))
