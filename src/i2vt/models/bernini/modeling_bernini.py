from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from i2vt.models.bernini.inference_config import (
    OFFICIAL_WAN_NEGATIVE_PROMPT,
    v2v_model_prompt,
)
from i2vt.models.modeling_utils import DistillationStep, StudentBackend
from i2vt.pipelines.bernini.pipeline_bernini import BerniniVideoEditPipeline, expert_boundary
from i2vt.pipelines.pipeline_utils import training_size
from i2vt.utils import (
    InMemoryDiskCache,
    configure_rollout_timesteps,
    dtype_from_name,
    filter_kwargs,
    resolve_model_path,
)

from .configuration_bernini import BerniniConfig


class BerniniStudentBackend(StudentBackend):
    """Bernini-R student (1.3B, or 14B with two experts) using the original Wan VAE and UMT5."""

    _lora_target_exclusions = ("_attn1_", "_self_attn_")
    # Class-level defaults, so objects the tests build without __init__ behave like a one-expert
    # LoRA student.
    full_finetune = False
    transformer_2 = None
    network_2 = None
    switch_boundary = None

    def __init__(self, config: BerniniConfig, device: torch.device):
        self.config = config
        self.device = device
        self.weight_dtype = dtype_from_name(config.weight_dtype)

        from diffusers import FlowMatchEulerDiscreteScheduler

        from i2vt.vendor.bernini.models.transformer_wan import WanTransformer3DModel
        from i2vt.vendor.wan.models.wan_text_encoder import WanT5EncoderModel
        from i2vt.vendor.wan.models.wan_vae import AutoencoderKLWan
        from i2vt.vendor.wan.utils.lora_utils import create_network

        config_path = config.component_config_path
        self.component_config = OmegaConf.load(config_path)
        component_root = config.component_root

        self.vae = AutoencoderKLWan.from_pretrained(
            resolve_model_path(
                component_root,
                self.component_config["vae_kwargs"].get("vae_subpath", "vae"),
                "vae",
            ),
            additional_kwargs=OmegaConf.to_container(self.component_config["vae_kwargs"]),
        ).to(device=device, dtype=self.weight_dtype)
        self.vae.eval().requires_grad_(False)

        tokenizer_path = resolve_model_path(
            component_root,
            self.component_config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            "tokenizer",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            resolve_model_path(
                component_root,
                self.component_config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
                "text_encoder",
            ),
            additional_kwargs=OmegaConf.to_container(self.component_config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).to(device=device, dtype=self.weight_dtype)
        self.text_encoder.eval().requires_grad_(False)

        self.transformer = WanTransformer3DModel.from_pretrained(
            config.model_path,
            subfolder="transformer",
            use_src_id_rotary_emb=True,
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=True,
        ).to(device=device, dtype=self.weight_dtype)
        self.full_finetune = config.tune_mode == "full"
        if self.full_finetune:
            self.transformer.train().requires_grad_(True)
        else:
            self.transformer.eval().requires_grad_(False)
        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            print("Enabled Bernini transformer gradient checkpointing", flush=True)

        # Bernini 14B uses the two-expert layout of Wan2.2-A14B: transformer handles high noise and
        # transformer_2 low noise, split at switch_dit_boundary (0.875 for 14B). 1.3B has one transformer.
        # The check is whether the directory exists, so the same code runs both sizes.
        self.transformer_2 = None
        self.switch_boundary = None
        second = Path(config.model_path) / "transformer_2"
        if second.is_dir():
            self.transformer_2 = WanTransformer3DModel.from_pretrained(
                config.model_path,
                subfolder="transformer_2",
                use_src_id_rotary_emb=True,
                torch_dtype=self.weight_dtype,
                low_cpu_mem_usage=True,
            ).to(device=device, dtype=self.weight_dtype)
            self.transformer_2.eval().requires_grad_(False)
            if config.gradient_checkpointing:
                self.transformer_2.enable_gradient_checkpointing()
            self.switch_boundary = expert_boundary(config.model_path) * 1000.0
            print(f"Bernini: two experts loaded, switch boundary t={self.switch_boundary:.0f}", flush=True)

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(
                FlowMatchEulerDiscreteScheduler,
                {
                    "num_train_timesteps": 1000,
                    "shift": config.flow_shift,
                    "use_dynamic_shifting": False,
                },
            )
        )

        if self.full_finetune:
            # Nothing to wrap: the transformer's own parameters are the ones
            # the optimizer sees, so lora_rank/alpha/target are all ignored.
            self.network = None
            trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            print(f"Bernini full finetune: {trainable/1e6:.1f}M trainable parameters", flush=True)
        else:
            self.network = create_network(1.0, config.lora_rank, config.lora_alpha, None, self.transformer)
            self._filter_lora_targets()
            self.network.apply_to(None, self.transformer, False, True)
            if config.lora_path:
                expected_rank = self.read_lora_rank(config.lora_path)
                if expected_rank is not None and expected_rank != config.lora_rank:
                    raise ValueError(
                        f"Bernini LoRA rank mismatch: checkpoint rank={expected_rank}, --lora-rank={config.lora_rank}."
                    )
                info = self.network.load_weights(config.lora_path)
                print(
                    f"Loaded Bernini LoRA from {config.lora_path}: {info}",
                    flush=True,
                )
            # fp32 master weights: AdamW's per-step update (~lr) is below bf16's spacing at
            # LoRA-A magnitudes (~1e-4 at |w|~0.01), so bf16 weights would round most A updates away.
            # The forward still runs in bf16 under autocast; checkpoints are saved in weight_dtype.
            self.network.to(device=device, dtype=torch.float32)
            self.network.train()

        # Each expert gets its own LoRA, since their weights differ. The low-noise expert runs only on
        # steps with t < boundary, so its gradient comes from those steps alone.
        self.network_2 = None
        if self.transformer_2 is not None:
            self.network_2 = create_network(
                1.0, config.lora_rank, config.lora_alpha, None, self.transformer_2
            )
            self._filter_lora_targets(self.network_2)
            self.network_2.apply_to(None, self.transformer_2, False, True)
            if config.lora_path_2:
                info = self.network_2.load_weights(config.lora_path_2)
                print(f"Loaded Bernini low-noise LoRA from {config.lora_path_2}: {info}", flush=True)
            self.network_2.to(device=device, dtype=torch.float32)  # fp32 master weights, see network above
            self.network_2.train()

        stats_path = Path(config.model_path) / "vae" / "config.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Bernini VAE normalization config not found: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        latent_channels = int(stats.get("z_dim", len(stats["latents_mean"])))
        checkpoint_mean = torch.tensor(stats["latents_mean"])
        checkpoint_std = torch.tensor(stats["latents_std"])
        if checkpoint_mean.numel() != latent_channels or checkpoint_std.numel() != latent_channels:
            raise ValueError(f"Bernini latent normalization shape mismatch: {stats_path}")
        if torch.any(checkpoint_std <= 0):
            raise ValueError(f"Bernini latent std must be positive: {stats_path}")
        # The reused AutoencoderKLWan applies (z_raw - mean) / std internally. JoyEdit's WanxVAE uses
        # the same statistics, so student and teacher share one normalized latent space.
        if not torch.allclose(self.vae.mean.cpu(), checkpoint_mean, atol=1e-6):
            raise ValueError("Reused Wan VAE latent mean differs from Bernini.")
        if not torch.allclose(self.vae.std.cpu(), checkpoint_std, atol=1e-6):
            raise ValueError("Reused Wan VAE latent std differs from Bernini.")

        self.cache_size = max(0, int(getattr(config, "cache_size", 8)))
        cache_root = getattr(config, "cache_dir", None)
        self._prompt_cache = InMemoryDiskCache(
            namespace="bernini_prompt",
            cache_size=self.cache_size,
            root=cache_root,
            device=self.device,
            dtype=self.weight_dtype,
        )
        self._source_cache = InMemoryDiskCache(
            namespace="bernini_source",
            cache_size=self.cache_size,
            root=cache_root,
            device=self.device,
            dtype=self.weight_dtype,
        )

    @property
    def trainable_module(self):
        return self.transformer if self.full_finetune else self.network

    def params(self):
        if self.full_finetune:
            return [p for p in self.transformer.parameters() if p.requires_grad]
        nets = [self.network] + ([self.network_2] if self.network_2 is not None else [])
        return [p for net in nets for p in net.parameters() if p.requires_grad]

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the edit instruction and configured rollout-negative prompt."""

        # Both halves are fixed, not configurable: pipeline_bernini.sample_video
        # always prepends the system prompt and always uses the official
        # negative prompt, so anything else here trains the student on a
        # condition it is never evaluated under.
        conditional_prompt = v2v_model_prompt(prompt)
        unconditional_prompt = OFFICIAL_WAN_NEGATIVE_PROMPT
        key = self.hash_key(
            {
                "conditional_prompt": conditional_prompt,
                "unconditional_prompt": unconditional_prompt,
                "max_sequence_length": self.config.max_sequence_length,
                "component_root": str(self.config.component_root or self.config.component_root),
            }
        )
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        prompt_embeds = (
            BerniniVideoEditPipeline.encode_text(
                self.tokenizer,
                self.text_encoder,
                conditional_prompt,
                max_length=self.config.max_sequence_length,
                device=self.device,
                dtype=self.weight_dtype,
            ),
            BerniniVideoEditPipeline.encode_text(
                self.tokenizer,
                self.text_encoder,
                unconditional_prompt,
                max_length=self.config.max_sequence_length,
                device=self.device,
                dtype=self.weight_dtype,
            ),
        )
        self._prompt_cache.put(key, prompt_embeds)
        return prompt_embeds

    def _expert_for(self, timestep):
        """High noise uses transformer and low noise transformer_2. 1.3B has one and always returns it."""
        if self.transformer_2 is None or self.switch_boundary is None:
            return self.transformer
        t = timestep
        if torch.is_tensor(t):
            t = t.flatten()[0].item()
        return self.transformer if float(t) >= self.switch_boundary else self.transformer_2

    def _velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        source_latents: torch.Tensor,
        *,
        rollout_guidance_scale: float,
        loss_guidance_scale: float,
    ):
        do_cfg = rollout_guidance_scale > 1.0 or loss_guidance_scale > 1.0
        expert = self._expert_for(timestep)
        cond = BerniniVideoEditPipeline.spatial_prediction(
            expert,
            source_latents,
            latents,
            timestep,
            prompt_embeds,
            dtype=self.weight_dtype,
            device=self.device,
        )
        if not do_cfg:
            return cond
        # With loss CFG 1 the unconditional branch contributes exactly zero gradient:
        # loss = (1 - loss_cfg) * uncond + loss_cfg * cond, so its coefficient is 0,
        # and the rollout branch only feeds scheduler.step inside no_grad.
        # No graph is built for it in that case. Forward values are unchanged and one backward pass is saved.
        uncond_needs_grad = loss_guidance_scale != 1.0
        with torch.set_grad_enabled(torch.is_grad_enabled() and uncond_needs_grad):
            uncond = BerniniVideoEditPipeline.spatial_prediction(
                expert,
                source_latents,
                latents,
                timestep,
                negative_prompt_embeds,
                dtype=self.weight_dtype,
                device=self.device,
            )
        delta = cond - uncond
        rollout_velocity = uncond + rollout_guidance_scale * delta
        if rollout_guidance_scale == loss_guidance_scale:
            return rollout_velocity
        loss_velocity = uncond + loss_guidance_scale * delta
        return rollout_velocity, loss_velocity

    def _train_size(self, image):
        """(width, height) for this sample, from the SOURCE frame's aspect ratio."""
        return training_size(image.size, height=self.config.height, width=self.config.width,
                             mode=getattr(self.config, "resize_mode", "area"))

    def _image_latents(self, image, size=None):
        """Encode one image to the student's normalized latent space, cached.

        Shared by the rollout and offline training. ``size`` is the source's training size; the
        stored edit is encoded at the source's size, so both land on the same grid even if the
        edit's own aspect ratio differs.
        """
        width, height = size or self._train_size(image)
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        cache_key = self.image_digest(resized)
        latents = self._source_cache.get(cache_key)
        if latents is None:
            latents = BerniniVideoEditPipeline.image_to_latents(
                self.vae,
                resized,
                width=width,
                height=height,
                device=self.device,
                dtype=self.weight_dtype,
            )
            self._source_cache.put(cache_key, latents)
        return latents

    def offline_steps(self, batch, rollout_config, offline_config):
        """Yield supervised steps against a stored edit instead of a rollout.

        The student never denoises here. Each step draws one noise level, builds the noisy latent
        from the stored edit rather than from the student's own trajectory, and predicts the
        velocity there; the offline teacher answers with the exact velocity to that edit. The
        trainer treats the resulting DistillationSteps like rollout steps.

        Sigma comes from a continuous density pushed through the same shift the sampler uses, so
        training covers every noise level inference visits, including the low-noise band
        (0.024, 0.395) that the rollout's ten discrete timesteps skip.
        """
        if batch.get("target") is None:
            raise ValueError("offline training needs a 'target' image on every row")
        started = time.perf_counter()
        prompt_embeds, _ = self._encode_prompt(batch["video_edit_prompt"])
        timings = {"encode_prompt": time.perf_counter() - started}

        started = time.perf_counter()
        size = self._train_size(batch["source"])
        source_latents = self._image_latents(batch["source"], size)
        target_latents = self._image_latents(batch["target"], size)
        timings["initial_latents"] = time.perf_counter() - started

        shift = float(self.config.flow_shift)
        num_train = float(self.scheduler.config.num_train_timesteps)

        for _ in range(max(1, int(offline_config.draws_per_sample))):
            if offline_config.sigma_sampling == "uniform":
                u = torch.rand((), device="cpu")
            else:
                u = torch.sigmoid(
                    torch.randn((), device="cpu") * offline_config.logit_std + offline_config.logit_mean
                )
            # The scheduler defines its grid as shift*u / (1 + (shift-1)*u) and
            # then t = 1000*sigma, so pushing the draw through the same map puts
            # the training density on the schedule inference actually walks.
            sigma_value = shift * u / (1.0 + (shift - 1.0) * u)
            sigma_value = float(torch.clamp(sigma_value, min=offline_config.sigma_min, max=1.0))
            sigma = torch.tensor(sigma_value, device=self.device, dtype=torch.float32)
            t = sigma * num_train

            noise = torch.randn_like(target_latents)
            # x_t = (1 - sigma) * x0 + sigma * eps, the path the sampler integrates.
            noisy = ((1.0 - sigma) * target_latents.float() + sigma * noise.float()).to(self.weight_dtype)
            noisy = noisy.detach().requires_grad_(True)
            started = time.perf_counter()
            output = self._velocity(
                noisy,
                t,
                prompt_embeds,
                prompt_embeds,
                source_latents,
                rollout_guidance_scale=1.0,
                loss_guidance_scale=1.0,
            )
            prediction = output[1] if isinstance(output, tuple) else output
            timings["student_velocity"] = time.perf_counter() - started
            yield DistillationStep(
                noisy.detach().clone(),
                prediction,
                t.detach().clone(),
                sigma.detach().clone(),
                num_steps=max(1, int(offline_config.draws_per_sample)),
                timings=timings,
            )
            timings = {}

    def rollout(self, batch, rollout_config):
        started = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(batch["video_edit_prompt"])
        timings = {"encode_prompt": time.perf_counter() - started}
        started = time.perf_counter()
        source_latents = self._image_latents(batch["source"])
        latents = torch.randn_like(source_latents)
        timings["initial_latents"] = time.perf_counter() - started
        configure_rollout_timesteps(
            self.scheduler,
            steps=rollout_config.num_inference_steps,
            device=self.device,
            mode=rollout_config.timestep_mode,
            min_timestep=rollout_config.timestep_min,
            max_timestep=rollout_config.timestep_max,
            mu=1,
        )
        timesteps = list(self.scheduler.timesteps)
        if rollout_config.train_steps:
            if rollout_config.train_steps > len(timesteps):
                raise ValueError("rollout train_steps exceeds scheduler steps")
            timesteps = timesteps[: rollout_config.train_steps]
        if not timesteps:
            raise ValueError("Rollout must contain at least one timestep")
        for timestep in timesteps:
            t = timestep.to(self.device)
            latents = latents.detach().requires_grad_(True)
            started = time.perf_counter()
            output = self._velocity(
                latents,
                t,
                prompt_embeds,
                negative_prompt_embeds,
                source_latents,
                rollout_guidance_scale=rollout_config.guidance_scale,
                loss_guidance_scale=rollout_config.student_guidance_scale,
            )
            rollout_prediction, loss_prediction = output if isinstance(output, tuple) else (output, output)
            timings["student_velocity"] = time.perf_counter() - started
            target = latents.detach().clone()
            prediction = loss_prediction
            sigma = (t.float() / float(self.scheduler.config.num_train_timesteps)).detach().clone()
            started = time.perf_counter()
            with torch.no_grad():
                latents = self.scheduler.step(rollout_prediction, t, latents, return_dict=False)[0]
            timings["scheduler_step"] = time.perf_counter() - started
            yield DistillationStep(
                target, prediction, t.detach().clone(), sigma, num_steps=len(timesteps), timings=timings
            )
            timings = {}
        return latents.detach()

    @torch.no_grad()
    def save(self, out_dir, step: int):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.full_finetune:
            from safetensors.torch import save_file

            # Another filename than the LoRA's: evaluation tells the two kinds apart by content,
            # and a mislabelled file would load as the wrong kind.
            state = {k: v.detach().to(self.weight_dtype).cpu() for k, v in self.transformer.state_dict().items()}
            save_file(
                state,
                str(out_dir / f"bernini_full_step_{step}.safetensors"),
                {"student_backend": "bernini", "step": str(step), "tune_mode": "full"},
            )
            return
        self.network.save_weights(
            str(out_dir / f"bernini_lora_step_{step}.safetensors"),
            self.weight_dtype,
            {"student_backend": "bernini", "step": str(step)},
        )
        if self.network_2 is not None:
            # One file per expert, so merge_lora can read each directly.
            self.network_2.save_weights(
                str(out_dir / f"bernini_lora_low_step_{step}.safetensors"),
                self.weight_dtype,
                {"student_backend": "bernini", "step": str(step), "expert": "low_noise"},
            )
