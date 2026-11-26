from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from i2vt.models.bernini.inference_config import (
    OFFICIAL_WAN_NEGATIVE_PROMPT as NEGATIVE_PROMPT,
)
from i2vt.models.bernini.inference_config import (
    V2V_SYSTEM_PROMPT,
    v2v_model_prompt,
)
from i2vt.pipelines.pipeline_utils import VideoEditPipelineOutput, image_to_tensor, load_wan_components
from i2vt.utils import dtype_from_name

from .configuration_bernini import BerniniPipelineConfig, pipeline_config


class _WanT5Adapter(nn.Module):
    """Expose the original VideoCoF UMT5 weights through Bernini's HF-style API."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        for module in self.modules():
            module._is_hf_initialized = True

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return SimpleNamespace(last_hidden_state=hidden_states)


def _make_unipc_scheduler(flow_shift: float):
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    return UniPCMultistepScheduler(
        beta_end=0.02,
        beta_schedule="linear",
        beta_start=0.0001,
        disable_corrector=[],
        dynamic_thresholding_ratio=0.995,
        final_sigmas_type="zero",
        flow_shift=flow_shift,
        lower_order_final=True,
        num_train_timesteps=1000,
        predict_x0=True,
        prediction_type="flow_prediction",
        rescale_betas_zero_snr=False,
        sample_max_value=1.0,
        solver_order=2,
        solver_p=None,
        solver_type="bh2",
        steps_offset=0,
        thresholding=False,
        time_shift_type="exponential",
        timestep_spacing="linspace",
        trained_betas=None,
        use_beta_sigmas=False,
        use_dynamic_shifting=False,
        use_exponential_sigmas=False,
        use_flow_sigmas=True,
        use_karras_sigmas=False,
    )


def bernini_frame_indices(
    total_frames: int,
    source_fps: float,
    target_fps: int,
    max_frames: int,
) -> list[int]:
    """Match Bernini's official fps-aware, 4n+1 frame sampling."""
    sampled_frames = total_frames / source_fps * target_fps
    sampled_frames = max(5, int(np.floor(sampled_frames / 4)) * 4 + 1)
    effective_total_frames = total_frames
    if source_fps == target_fps:
        effective_total_frames = int(np.floor(total_frames / 4)) * 4 + 1
    indices = torch.linspace(0, effective_total_frames - 1, sampled_frames).round().long().tolist()
    frame_cap = int(np.floor(max_frames / 4)) * 4 + 1
    return indices[:frame_cap]


def bernini_frame_size(width: int, height: int, max_image_size: int, stride: int = 16) -> tuple[int, int]:
    """Official MaxLongEdgeMinShortEdgeResize (min size 1): long edge at most ``max_image_size``, sides on ``stride``."""

    def snap(value):
        return max(stride, int(round(value / stride) * stride))

    scale = max(min(max_image_size / max(width, height), 1.0), 1 / min(width, height))
    new_width, new_height = snap(round(width * scale)), snap(round(height * scale))
    if max(new_width, new_height) > max_image_size:
        scale = max_image_size / max(new_width, new_height)
        new_width, new_height = snap(round(new_width * scale)), snap(round(new_height * scale))
    return new_width, new_height


def load_bernini_video(
    video_path: str,
    source_frames: int,
    max_image_size: int,
    fps: int,
    frame_sampling: str,
):
    """Load Bernini source frames using exact or official fps-aware sampling."""
    import decord

    reader = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Source video has no frames: {video_path}")

    if frame_sampling == "official":
        source_fps = float(reader.get_avg_fps() or fps)
        indices = bernini_frame_indices(
            total_frames=total_frames,
            source_fps=source_fps,
            target_fps=fps,
            max_frames=source_frames,
        )
    else:
        indices = torch.linspace(0, total_frames - 1, source_frames).round().long().tolist()
    frame_arrays = reader.get_batch(indices).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in frame_arrays]

    width, height = frames[0].size
    resized_width, resized_height = bernini_frame_size(width, height, max_image_size)
    if (resized_width, resized_height) != (width, height):
        frames = [frame.resize((resized_width, resized_height), Image.Resampling.BICUBIC) for frame in frames]

    video = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames]))
    video = video.permute(3, 0, 1, 2).unsqueeze(0).float()
    # ToTensor then Normalize(0.5, 0.5), in the official order of operations.
    return video.div(255.0).sub(0.5).div(0.5)


def expert_boundary(model_path) -> float | None:
    """Noise level (0-1) where a two-expert Bernini-R hands over to transformer_2; None for one expert.

    t >= boundary * 1000 runs the high-noise expert ``transformer``, lower t the low-noise ``transformer_2``.
    The value is ``switch_dit_boundary`` of the checkpoint's config.json (0.875 for Bernini-R-14B).
    """
    root = Path(model_path)
    if not (root / "transformer_2").is_dir():
        return None
    top = root / "config.json"
    boundary = json.loads(top.read_text()).get("switch_dit_boundary", 0.875) if top.is_file() else 0.875
    return float(boundary)


def low_noise_lora_path(lora_path) -> Path:
    """The low-noise expert's LoRA saved next to the high-noise one: bernini_lora_low_step_N.safetensors."""
    path = Path(lora_path)
    if not path.name.startswith("bernini_lora_step_"):
        raise ValueError(
            f"{path.name}: a two-expert checkpoint is passed as its high-noise file bernini_lora_step_<N>.safetensors"
        )
    return path.with_name(path.name.replace("bernini_lora_step_", "bernini_lora_low_step_", 1))


def merge_lora(module: nn.Module, lora_path, scale: float = 1.0) -> int:
    """Add ``scale * alpha / rank * up @ down`` to the weights of ``module`` in place; returns the layer count.

    Keys are the trainer's (``lora_unet_<module path with dots as underscores>``). The update is computed in
    float32 on the weight's device and added in the weight's dtype. Every LoRA module must match a weight.
    """
    from safetensors.torch import load_file

    state = load_file(str(lora_path))
    if not state:
        raise ValueError(f"{lora_path}: no LoRA weights")
    pairs = {}
    for key, value in state.items():
        for suffix, slot in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[slot] = value
                break
        else:
            raise ValueError(f"{lora_path}: unexpected key {key}")
    weights = {
        name[: -len(".weight")].replace(".", "_"): param
        for name, param in module.named_parameters()
        if name.endswith(".weight")
    }
    for stem, pair in pairs.items():
        if "down" not in pair or "up" not in pair:
            raise ValueError(f"{lora_path}: {stem} lacks lora_down or lora_up")
        weight = weights.get(stem.removeprefix("lora_unet_").lstrip("_"))
        if weight is None:
            raise ValueError(f"{lora_path}: {stem} matches no weight of {type(module).__name__}")
        rank = pair["down"].shape[0]
        alpha = float(pair["alpha"]) if "alpha" in pair else float(rank)
        delta = (pair["up"].to(torch.float32) @ pair["down"].to(torch.float32)) * (scale * alpha / rank)
        if delta.shape != weight.shape:
            raise ValueError(f"{lora_path}: {stem} has shape {tuple(delta.shape)}, weight {tuple(weight.shape)}")
        with torch.no_grad():
            weight.add_(delta.to(weight.device, weight.dtype))
    return len(pairs)


def merge_expert_loras(diff_dec, lora_path) -> dict[str, Path]:
    """Merge each expert's own LoRA into its weights and return {expert: file}.

    ``transformer`` takes ``lora_path`` (bernini_lora_step_N) and ``transformer_2`` the low-noise
    file saved beside it (bernini_lora_low_step_N).
    """
    paths = {"transformer": Path(lora_path), "transformer_2": low_noise_lora_path(lora_path)}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Bernini LoRA of {name} is missing: {path}")
    for name, path in paths.items():
        layers = merge_lora(getattr(diff_dec, name), path)
        print(f"Merged Bernini LoRA into {name}: {layers} layers from {path}", flush=True)
    return paths


class BerniniVideoEditPipeline:
    """Official Bernini-R v2v inference for the one-expert 1.3B and the two-expert 14B checkpoints."""

    model_name = "Bernini-R-1.3B"
    two_experts = False
    lora_paths: dict = {}

    @staticmethod
    def encode_text(tokenizer, text_encoder, prompt: str, *, max_length: int, device, dtype):
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(
                input_ids=tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
            )[0]
        # Official renderer.encode_prompt keeps the first seq_len rows and pads the rest with
        # zeros back to max_length, then tells the DiT all max_length positions are text. T5's
        # outputs at pad positions are not zero, so they are zeroed here as well.
        seq_len = int(tokens.attention_mask.sum().item())
        embeds = embeds.clone()
        embeds[:, seq_len:] = 0
        return embeds.to(dtype=dtype).detach()

    @staticmethod
    def spatial_prediction(transformer, source_latents, target_latents, timestep, text_embeds, *, dtype, device):
        source = source_latents.to(dtype=dtype)
        target = target_latents.to(dtype=dtype)
        source_tokens, source_rope = transformer.patch_vae_latent(source, source_id=1)
        target_tokens, target_rope = transformer.patch_vae_latent(target, source_id=0)
        hidden_states = torch.cat([source_tokens, target_tokens], dim=1)
        rotary = torch.cat([source_rope, target_rope], dim=2)
        autocast = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
        with autocast:
            prediction = transformer(
                hidden_states=hidden_states,
                timestep=timestep.to(device=device, dtype=torch.float32).reshape(-1)[:1],
                encoder_hidden_states=text_embeds,
                rotary_emb=rotary,
                batch_image_vae_seqlen=[hidden_states.shape[1]],
                text_features_length=[text_embeds.shape[1]],
            ).sample
        prediction = prediction[:, source_tokens.shape[1] :]
        patch_t, patch_h, patch_w = transformer.config.patch_size
        channels = target.shape[1]
        latent_t, latent_h, latent_w = target.shape[-3:]
        expected = math.ceil(latent_t / patch_t) * math.ceil(latent_h / patch_h) * math.ceil(latent_w / patch_w)
        if prediction.shape[1] != expected:
            raise RuntimeError(
                f"Bernini target token count mismatch: prediction={prediction.shape[1]} expected={expected}"
            )
        return (
            prediction.view(
                target.shape[0],
                latent_t // patch_t,
                latent_h // patch_h,
                latent_w // patch_w,
                patch_t,
                patch_h,
                patch_w,
                channels,
            )
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape_as(target)
        )

    @staticmethod
    def encode_prompt(tokenizer, prompt: str, negative_prompt: str, *, max_length: int, device):
        return tuple(
            tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            for text in (prompt, negative_prompt)
        )

    @staticmethod
    def image_to_latents(vae, image, *, width: int, height: int, device, dtype):
        image_tensor = image_to_tensor(image, width=width, height=height, device=device, dtype=dtype)
        return vae.encode(image_tensor)[0].mode().detach().to(device=device, dtype=dtype)

    def __init__(self, config: BerniniPipelineConfig):
        if config.num_frames != config.source_frames:
            raise ValueError("Bernini v2v uses --num-frames/--source-frames as the same frame cap.")
        if config.num_frames % 4 != 1:
            raise ValueError("Bernini/Wan video frame cap must be 4n+1 (for example, 33 or 81).")

        self.config = config
        dit_root = Path(config.model_path)
        if not (dit_root / "transformer" / "config.json").is_file():
            raise FileNotFoundError(f"Bernini transformer config is missing: {dit_root / 'transformer/config.json'}")

        from transformers import AutoTokenizer

        from i2vt.vendor.bernini.attention import get_attention_backend
        from i2vt.vendor.bernini.models import BerniniRendererConfig, BerniniRendererModel, wan_diffusion
        from i2vt.vendor.bernini.models import renderer as renderer_module

        torch.manual_seed(config.seed)
        device = torch.device(config.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self.device = device

        dtype = dtype_from_name(config.weight_dtype)
        vae_config_path = dit_root / "vae" / "config.json"
        if not vae_config_path.is_file():
            raise FileNotFoundError(f"Bernini VAE normalization config is missing: {vae_config_path}")
        vae_config = json.loads(vae_config_path.read_text(encoding="utf-8"))
        latent_channels = int(vae_config.get("z_dim", len(vae_config["latents_mean"])))
        checkpoint_mean = torch.tensor(vae_config["latents_mean"])
        checkpoint_std = torch.tensor(vae_config["latents_std"])
        if checkpoint_mean.numel() != latent_channels or checkpoint_std.numel() != latent_channels:
            raise ValueError(f"Bernini latent normalization shape mismatch: {vae_config_path}")
        if torch.any(checkpoint_std <= 0):
            raise ValueError(f"Bernini latent std must be positive: {vae_config_path}")
        if config.component_source == "official":
            from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
            from transformers import UMT5EncoderModel

            required_dirs = ("tokenizer", "text_encoder", "vae")
            missing_dirs = [name for name in required_dirs if not (dit_root / name).is_dir()]
            if missing_dirs:
                raise FileNotFoundError(
                    f"Official Bernini components are incomplete under {dit_root}: missing {missing_dirs}"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                dit_root,
                subfolder="tokenizer",
                local_files_only=True,
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                dit_root,
                subfolder="text_encoder",
                torch_dtype=dtype,
                local_files_only=True,
            )
            original_vae = DiffusersAutoencoderKLWan.from_pretrained(
                dit_root,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae_needs_external_normalization = True
            base_components_root = dit_root
        else:
            videocof_config, tokenizer, original_text_encoder, original_vae = load_wan_components(
                component_root=config.component_root,
                component_config_path=config.component_config_path,
                weight_dtype=config.weight_dtype,
            )
            original_vae = original_vae.to(torch.float32)
            if not torch.allclose(original_vae.mean.cpu(), checkpoint_mean, atol=1e-6):
                raise ValueError("Reused Wan VAE latent mean differs from Bernini.")
            if not torch.allclose(original_vae.std.cpu(), checkpoint_std, atol=1e-6):
                raise ValueError("Reused Wan VAE latent std differs from Bernini.")
            text_encoder = _WanT5Adapter(original_text_encoder)
            vae_needs_external_normalization = False
            base_components_root = Path(config.component_root)
        boundary = expert_boundary(dit_root)
        two_experts = boundary is not None
        renderer_config = BerniniRendererConfig(
            wan22_base=str(dit_root),
            skip_transformer_1=False,
            skip_transformer_2=not two_experts,
            switch_dit_boundary=boundary if two_experts else 0,
            max_sequence_length=512,
            shift=config.shift,
            use_unipc=True,
            use_src_id_rotary_emb=True,
        )
        renderer_config.dtype = dtype

        original_t5_loader = renderer_module.UMT5EncoderModel.from_pretrained
        original_scheduler_loader = wan_diffusion.UniPCMultistepScheduler.from_pretrained
        try:
            renderer_module.UMT5EncoderModel.from_pretrained = lambda *_args, **_kwargs: text_encoder
            wan_diffusion.UniPCMultistepScheduler.from_pretrained = lambda *_args, **_kwargs: _make_unipc_scheduler(
                config.shift
            )
            model = BerniniRendererModel(renderer_config)
        finally:
            renderer_module.UMT5EncoderModel.from_pretrained = original_t5_loader
            wan_diffusion.UniPCMultistepScheduler.from_pretrained = original_scheduler_loader

        model.eval()
        original_vae.eval()
        bernini_lora_network = None
        lora_paths = {}
        if config.lora_path and two_experts:
            lora_paths = merge_expert_loras(model.diff_dec, config.lora_path)
        elif config.lora_path:
            lora_paths = {"transformer": Path(config.lora_path)}
            from safetensors import safe_open

            checkpoint = Path(config.lora_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Bernini checkpoint is missing: {checkpoint}")
            with safe_open(checkpoint, framework="pt", device="cpu") as handle:
                is_lora = any(key.endswith(".lora_down.weight") for key in handle.keys())
        if config.lora_path and not two_experts and not is_lora:
            # A full-finetune checkpoint is the transformer's own state dict, so
            # it replaces the base weights rather than being applied on top of
            # them. Told apart by content, not by filename, so a renamed file
            # cannot be loaded as the wrong kind.
            from safetensors.torch import load_file

            target = model.diff_dec.transformer
            target_dtype = next(target.parameters()).dtype
            state = load_file(str(config.lora_path), device="cpu")
            target.load_state_dict({k: v.to(target_dtype) for k, v in state.items()}, strict=True)
            print(
                f"Loaded Bernini full-finetune weights from {config.lora_path}: {len(state)} tensors",
                flush=True,
            )
            del state
        elif config.lora_path and not two_experts:
            from safetensors import safe_open

            from i2vt.vendor.wan.utils.lora_utils import create_network

            lora_path = Path(config.lora_path)
            if not lora_path.is_file():
                raise FileNotFoundError(f"Bernini LoRA is missing: {lora_path}")
            with safe_open(lora_path, framework="pt", device="cpu") as handle:
                checkpoint_keys = set(handle.keys())
                down_keys = sorted(key for key in checkpoint_keys if key.endswith(".lora_down.weight"))
                if not down_keys:
                    raise ValueError(f"Bernini LoRA has no lora_down weights: {lora_path}")
                first_down = down_keys[0]
                lora_rank = int(handle.get_tensor(first_down).shape[0])
                alpha_key = first_down.removesuffix(".lora_down.weight") + ".alpha"
                lora_alpha = (
                    float(handle.get_tensor(alpha_key).item()) if alpha_key in checkpoint_keys else float(lora_rank)
                )
                checkpoint_modules = {key.removesuffix(".lora_down.weight") for key in down_keys}
            bernini_lora_network = create_network(
                1.0,
                lora_rank,
                lora_alpha,
                None,
                model.diff_dec.transformer,
            )
            available_modules = {lora.lora_name for lora in bernini_lora_network.unet_loras}
            unknown_modules = sorted(checkpoint_modules - available_modules)
            if unknown_modules:
                raise ValueError(f"Bernini LoRA contains unknown modules: {unknown_modules[:5]}")
            before = len(bernini_lora_network.unet_loras)
            bernini_lora_network.unet_loras = [
                lora for lora in bernini_lora_network.unet_loras if lora.lora_name in checkpoint_modules
            ]
            print(
                "Bernini inference LoRA modules: "
                f"loaded {len(bernini_lora_network.unet_loras)}, "
                f"omitted {before - len(bernini_lora_network.unet_loras)}",
                flush=True,
            )
            bernini_lora_network.apply_to(None, model.diff_dec.transformer, False, True)
            load_info = bernini_lora_network.load_weights(str(lora_path))
            if load_info.missing_keys or load_info.unexpected_keys:
                raise ValueError(
                    "Bernini LoRA key mismatch: "
                    f"missing={load_info.missing_keys[:5]} "
                    f"unexpected={load_info.unexpected_keys[:5]}"
                )
            bernini_lora_network.to(device=device, dtype=dtype).eval()
        self.model = model
        self.two_experts = two_experts
        self.model_name = "Bernini-R-14B" if two_experts else "Bernini-R-1.3B"
        self.lora_paths = lora_paths
        self.vae = original_vae
        self.tokenizer = tokenizer
        self.bernini_lora_network = bernini_lora_network
        self.dit_root = dit_root
        self.vae_latent_mean = checkpoint_mean.view(1, latent_channels, 1, 1, 1)
        self.vae_latent_std = checkpoint_std.view(1, latent_channels, 1, 1, 1)
        self.vae_needs_external_normalization = vae_needs_external_normalization
        self.base_components_root = base_components_root
        print(
            f"Loaded {self.model_name}: transformer={dit_root} "
            f"components={config.component_source}:{base_components_root} "
            f"lora={', '.join(f'{k}={v}' for k, v in lora_paths.items()) or 'none'} "
            f"attention={get_attention_backend()}",
            flush=True,
        )

    @torch.no_grad()
    def __call__(self, video: str, prompt: str) -> VideoEditPipelineOutput:
        video_path = video
        config = self.config
        input_video = load_bernini_video(
            video_path,
            config.source_frames,
            config.max_image_size,
            config.fps,
            config.frame_sampling,
        )
        sampled_source_frames = input_video.shape[2]
        model_prompt = v2v_model_prompt(prompt)
        prompt_tokens, negative_tokens = self.encode_prompt(
            self.tokenizer, model_prompt, NEGATIVE_PROMPT, max_length=512, device=self.device
        )

        self.vae.to(self.device)
        input_video = input_video.to(device=self.device, dtype=self.vae.dtype)
        source_latents = self.vae.encode(input_video).latent_dist.mode()
        if self.vae_needs_external_normalization:
            mean = self.vae_latent_mean.to(device=source_latents.device, dtype=source_latents.dtype)
            std = self.vae_latent_std.to(device=source_latents.device, dtype=source_latents.dtype)
            source_latents = (source_latents - mean) / std
        self.vae.to("cpu")
        del input_video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not self.two_experts:
            # With two experts the sampler keeps one on the GPU at a time and moves it in when its steps begin.
            self.model.to(self.device)
        latents = self.model.sample(
            input_ids=prompt_tokens.input_ids.to(self.device),
            attention_mask=prompt_tokens.attention_mask.to(self.device),
            uncond_input_ids=negative_tokens.input_ids.to(self.device),
            uncond_attention_mask=negative_tokens.attention_mask.to(self.device),
            multi_video_vae_latents=[source_latents],
            num_frames=sampled_source_frames,
            width=source_latents.shape[-1] * 8,
            height=source_latents.shape[-2] * 8,
            num_inference_steps=config.num_inference_steps,
            guidance_mode=config.guidance_mode,
            omega_vid=config.omega_vid,
            omega_img=config.omega_img,
            omega_txt=config.omega_txt,
            omega_scale=config.omega_scale,
            flow_shift=config.shift,
            seed=config.seed,
            device=self.device,
            eta=config.eta,
            norm_threshold=tuple(config.norm_threshold),
            momentum=config.momentum,
        )
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.vae.to(self.device)
        decode_latents = latents.to(self.vae.dtype)
        if self.vae_needs_external_normalization:
            mean = self.vae_latent_mean.to(device=decode_latents.device, dtype=decode_latents.dtype)
            std = self.vae_latent_std.to(device=decode_latents.device, dtype=decode_latents.dtype)
            decode_latents = decode_latents * std + mean
        decoded = self.vae.decode(decode_latents, return_dict=False)[0]
        self.vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metadata = {
            "prompt": prompt,
            "model_prompt": model_prompt,
            "system_prompt": V2V_SYSTEM_PROMPT,
            "backend": "bernini",
            "model": self.model_name,
            "transformer_path": str(self.dit_root),
            "component_source": config.component_source,
            "base_components_root": str(self.base_components_root),
            "task_type": "v2v",
            "guidance": {
                "mode": config.guidance_mode,
                "omega_vid": config.omega_vid,
                "omega_img": config.omega_img,
                "omega_txt": config.omega_txt,
                "omega_scale": config.omega_scale,
                "eta": config.eta,
                "momentum": config.momentum,
                "norm_threshold": config.norm_threshold,
            },
            "generation": {
                "frame_cap": config.num_frames,
                "frame_sampling": config.frame_sampling,
                "sampled_source_frames": sampled_source_frames,
                "output_frames": decoded.shape[2],
                "fps": config.fps,
                "steps": config.num_inference_steps,
                "shift": config.shift,
                "seed": config.seed,
                "max_image_size": config.max_image_size,
            },
            "prompt_enhancement": False,
            "loras": [{"path": str(path), "expert": name, "scale": 1.0} for name, path in self.lora_paths.items()],
        }
        return VideoEditPipelineOutput(frames=decoded.detach().cpu(), fps=config.fps, metadata=metadata)

    @classmethod
    def from_config(cls, config, device="cuda:0"):
        return cls(pipeline_config(config, device))
