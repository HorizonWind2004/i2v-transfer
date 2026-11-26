from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from i2vt.models.modeling_utils import (
    DistillationStep,
    StudentBackend,
)
from i2vt.models.videocof.inference_config import OFFICIAL_VIDEOCOF_NEGATIVE_PROMPT
from i2vt.pipelines.videocof.pipeline_videocof import VideoCoFVideoEditPipeline
from i2vt.utils import (
    InMemoryDiskCache,
    configure_rollout_timesteps,
    dtype_from_name,
    filter_kwargs,
    resolve_model_path,
)

from .configuration_videocof import VideoCoFConfig


@dataclass(frozen=True)
class VideoCoFSampleContext:
    source_count: int
    edit_start: int


class VideoCoFStudentBackend(StudentBackend):
    def __init__(self, config: VideoCoFConfig, device: torch.device):
        self.config = config
        self.device = device

        from diffusers import FlowMatchEulerDiscreteScheduler

        from i2vt.vendor.wan.models.wan_text_encoder import WanT5EncoderModel
        from i2vt.vendor.wan.models.wan_transformer3d import WanTransformer3DModel
        from i2vt.vendor.wan.models.wan_vae import AutoencoderKLWan
        from i2vt.vendor.wan.pipeline.pipeline_wan import WanPipeline
        from i2vt.vendor.wan.utils.lora_utils import create_network, merge_lora

        config_path = config.component_config_path
        self.component_config = OmegaConf.load(config_path)
        self.weight_dtype = dtype_from_name(config.weight_dtype)
        model_root = config.component_root

        self.vae = AutoencoderKLWan.from_pretrained(
            resolve_model_path(model_root, self.component_config["vae_kwargs"].get("vae_subpath", "vae"), "vae"),
            additional_kwargs=OmegaConf.to_container(self.component_config["vae_kwargs"]),
        ).to(device=device, dtype=self.weight_dtype)
        self.vae.eval().requires_grad_(False)

        self.transformer = WanTransformer3DModel.from_pretrained(
            resolve_model_path(
                model_root,
                self.component_config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
                "transformer",
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                self.component_config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).to(device=device, dtype=self.weight_dtype)
        self.transformer.eval().requires_grad_(False)
        if config.gradient_checkpointing:
            self.transformer._set_gradient_checkpointing(enable=True)
            print("Enabled VideoCoF transformer gradient checkpointing", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolve_model_path(
                model_root,
                self.component_config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
                "tokenizer",
            )
        )
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            resolve_model_path(
                model_root,
                self.component_config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
                "text_encoder",
            ),
            additional_kwargs=OmegaConf.to_container(self.component_config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).to(device=device, dtype=self.weight_dtype)
        self.text_encoder.eval().requires_grad_(False)

        scheduler_kwargs = OmegaConf.to_container(self.component_config["scheduler_kwargs"])
        # The training rollout shift decides where the 10 steps fall on the noise axis: the component config's
        # 5.0 puts 6 of 10 steps above t=800, and shift=1 gives the uniform [1000, 889, ..., 1].
        if self.config.rollout_shift is not None:
            scheduler_kwargs["shift"] = self.config.rollout_shift
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
        )
        self.pipeline = WanPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
        ).to(device)

        if (
            config.merge_lora_path
            and config.lora_path
            and os.path.abspath(config.merge_lora_path) == os.path.abspath(config.lora_path)
        ):
            raise ValueError(
                "--videocof-merge-lora-path and --videocof-lora-path point to the same file. "
                "Merge the pretrained base LoRA or continue a trainable LoRA, but do not do both with the same weights."
            )

        if config.merge_lora_path:
            self.pipeline = merge_lora(
                self.pipeline,
                config.merge_lora_path,
                multiplier=1.0,
                device=device,
                dtype=self.weight_dtype,
                transformer_only=True,
            )
            print(f"Merged VideoCoF LoRA into base transformer: {config.merge_lora_path}", flush=True)

        self.network = create_network(1.0, config.lora_rank, config.lora_alpha, None, self.transformer)
        self._filter_lora_targets()
        self.network.apply_to(self.text_encoder, self.transformer, False, True)
        if config.lora_path:
            expected_rank = self.read_lora_rank(config.lora_path)
            if expected_rank is not None and expected_rank != config.lora_rank:
                raise ValueError(
                    f"VideoCoF LoRA rank mismatch: checkpoint rank={expected_rank}, "
                    f"--lora-rank={config.lora_rank}. Set LORA_RANK={expected_rank}."
                )
            info = self.network.load_weights(config.lora_path)
            print(f"Loaded VideoCoF LoRA from {config.lora_path}: {info}", flush=True)
        # fp32 master weights, as for Bernini: one AdamW update (~lr) is smaller than the bf16 spacing at the
        # magnitude of LoRA-A, so bf16 parameters would round most updates of A away. The forward pass still runs
        # under the bf16 autocast in _velocity, the LoRA module casts its output back, and checkpoints keep
        # weight_dtype.
        self.network.to(device=device, dtype=torch.float32)
        self.network.train()
        self.cache_size = max(0, int(getattr(config, "cache_size", 8)))
        cache_root = getattr(config, "cache_dir", None)
        self._cache = InMemoryDiskCache(
            namespace="videocof_student",
            cache_size=self.cache_size,
            root=cache_root,
            device=self.device,
            dtype=self.weight_dtype,
        )

    def params(self):
        return [p for p in self.network.parameters() if p.requires_grad]

    def _encode_prompt(self, prompt: str):
        formatted_prompt = VideoCoFVideoEditPipeline.format_prompt(prompt, use_cot=self.config.use_cot_prompt)
        do_cfg = self.config.rollout_guidance_scale > 1.0 or self.config.student_guidance_scale > 1.0
        # An empty string makes CFG push away from nothing. The official prompt makes it push away from gray,
        # low-quality or static outputs (it contains "overall gray"). It is encoded only when do_cfg is set.
        negative_prompt = OFFICIAL_VIDEOCOF_NEGATIVE_PROMPT if self.config.use_official_negative_prompt else ""
        prompt_key = self.hash_key(
            {
                "kind": "videocof_prompt_embeds_v2",
                "prompt": formatted_prompt,
                # Part of the key, so another negative prompt never reuses a cached embedding.
                "negative_prompt": negative_prompt,
                "rollout_guidance_scale": float(self.config.rollout_guidance_scale),
                "student_guidance_scale": float(self.config.student_guidance_scale),
                "do_cfg": do_cfg,
                "max_sequence_length": self.config.max_sequence_length,
                "model_root": str(self.config.component_root),
                "config": str(self.config.component_config_path),
                "dtype": str(self.weight_dtype),
            }
        )
        memory_key = f"prompt:{prompt_key}"
        prompt_embeds = self._cache.get(memory_key)
        if prompt_embeds is not None:
            return prompt_embeds
        prompt_embeds = VideoCoFVideoEditPipeline.encode_prompt(
            self.pipeline,
            formatted_prompt,
            max_length=self.config.max_sequence_length,
            do_cfg=do_cfg,
            device=self.device,
            negative_prompt=negative_prompt,
        )
        # encode_prompt returns a list of variable-length tensors; the cache detaches and
        # moves to CPU itself, matching how the bernini backend stores its prompt tuple.
        self._cache.put(memory_key, prompt_embeds)
        return prompt_embeds

    @torch.no_grad()
    def save(self, out_dir, step: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.network.save_weights(str(out_dir / f"videocof_lora_step_{step}.safetensors"), self.weight_dtype, {})

    def _velocity(self, latents, timestep, prompt, context, *, rollout_guidance_scale, loss_guidance_scale):
        rollout_guidance_scale = (
            rollout_guidance_scale if rollout_guidance_scale is not None else self.config.rollout_guidance_scale
        )
        loss_guidance_scale = (
            loss_guidance_scale if loss_guidance_scale is not None else self.config.student_guidance_scale
        )
        do_cfg = rollout_guidance_scale > 1.0 or loss_guidance_scale > 1.0
        source_count = context.source_count
        edit_start = context.edit_start

        target_shape = (self.vae.latent_channels, latents.shape[2], latents.shape[3], latents.shape[4])
        patch = self.transformer.config.patch_size
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (patch[1] * patch[2]) * target_shape[1])

        def forward(latent_input, context_embeds):
            if hasattr(self.scheduler, "scale_model_input"):
                latent_input = self.scheduler.scale_model_input(latent_input, timestep)
            batch = latent_input.shape[0]
            with torch.cuda.amp.autocast(dtype=self.weight_dtype):
                return self.transformer(
                    x=latent_input,
                    context=context_embeds,
                    t=timestep.expand(batch),
                    seq_len=seq_len,
                    frame_split_indices=[source_count] * batch,
                    ground_frame_indices=([(source_count, edit_start)] * batch if edit_start > source_count else None),
                )

        if not do_cfg:
            prediction = forward(latents, prompt)
            prediction[:, :, :source_count] = 0
            return prediction

        # The prompt is the list [negative, positive] (see the pipeline's encode_prompt), so index 0 is the
        # unconditional branch. The two branches run as separate forwards:
        #
        #  1. With loss CFG 1 the unconditional branch contributes exactly zero gradient:
        #     loss = (1 - loss_cfg) * uncond + loss_cfg * cond, so its coefficient is 0,
        #     and the rollout branch only feeds scheduler.step, which runs inside no_grad.
        #     It then runs under no_grad with unchanged values.
        #  2. Two forwards need half the peak activation memory of one batch-2 forward.
        #
        # With loss CFG other than 1 the unconditional gradient coefficient is 1 - loss_cfg != 0, so it needs a graph.
        uncond_needs_grad = loss_guidance_scale != 1.0
        with torch.set_grad_enabled(torch.is_grad_enabled() and uncond_needs_grad):
            uncond = forward(latents, prompt[:1])
        cond = forward(latents, prompt[1:])

        delta = cond - uncond
        rollout_prediction = uncond + rollout_guidance_scale * delta
        rollout_prediction[:, :, :source_count] = 0
        if loss_guidance_scale == rollout_guidance_scale:
            return rollout_prediction
        loss_prediction = uncond + loss_guidance_scale * delta
        loss_prediction[:, :, :source_count] = 0
        return rollout_prediction, loss_prediction

    def _train_size(self, image):
        """(width, height) for this sample, from the SOURCE frame's aspect ratio."""
        from i2vt.pipelines.pipeline_utils import training_size

        return training_size(image.size, height=self.config.height, width=self.config.width,
                             mode=getattr(self.config, "resize_mode", "area"))

    def _image_latents(self, image, size=None):
        width, height = size or self._train_size(image)
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        source_key = self.hash_key(
            {
                "kind": "videocof_source_latents_v1",
                "image_sha1": self.image_digest(resized),
                "height": height,
                "width": width,
                "model_root": str(self.config.component_root),
                "config": str(self.config.component_config_path),
                "dtype": str(self.weight_dtype),
            }
        )
        memory_key = f"source_latents:{source_key}"
        latents = self._cache.get(memory_key)
        if latents is None:
            latents = VideoCoFVideoEditPipeline.image_to_latents(
                self.vae,
                resized,
                width=width,
                height=height,
                device=self.device,
                dtype=self.weight_dtype,
            )
            self._cache.put(memory_key, latents)
        return latents

    def offline_steps(self, batch, rollout_config, offline_config):
        """Yield supervised steps against a stored edit, the VideoCoF twin of Bernini's.

        Same contract as `BerniniStudentBackend.offline_steps`: one continuous sigma
        per draw, pushed through the shift inference samples with, the noisy
        latent built from the stored edit, and the offline teacher answering with
        (x_t - x0) / sigma on the edit frames.

        VideoCoF also denoises reasoning latents between the source and the edit,
        and the dataset has no clean value for them, only the edited image. They
        get the source latent as their clean value at the same sigma: the official
        reasoning frame is the source with the edit region marked, so that is the
        nearest thing to it, and every non-source frame then sits at the noise
        level the timestep claims. The loss covers only the edit frames, so this
        choice shapes the context the student sees, not what it is scored on.
        """
        if batch.get("target") is None:
            raise ValueError("offline training needs a 'target' image on every row")
        started = time.perf_counter()
        prompt = self._encode_prompt(batch["video_edit_prompt"])
        timings = {"encode_prompt": time.perf_counter() - started}

        started = time.perf_counter()
        size = self._train_size(batch["source"])
        source_latents = self._image_latents(batch["source"], size)
        target_latents = self._image_latents(batch["target"], size)
        if target_latents.shape != source_latents.shape:
            raise ValueError(f"target latents {tuple(target_latents.shape)} != source {tuple(source_latents.shape)}")
        source_count = source_latents.shape[2]
        reasoning_count = self.config.reasoning_latent_count
        edit_start = source_count + reasoning_count
        context = VideoCoFSampleContext(source_count, edit_start)
        clean_tail = torch.cat([source_latents[:, :, :1].expand(-1, -1, reasoning_count, -1, -1), target_latents], dim=2)
        timings["initial_latents"] = time.perf_counter() - started

        # The config checks that this equals the shift inference samples with.
        shift = float(rollout_config.kwargs["flow_shift"])
        num_train = float(self.scheduler.config.num_train_timesteps)
        draws = max(1, int(offline_config.draws_per_sample))

        for _ in range(draws):
            if offline_config.sigma_sampling == "uniform":
                u = torch.rand((), device="cpu")
            else:
                u = torch.sigmoid(torch.randn((), device="cpu") * offline_config.logit_std + offline_config.logit_mean)
            sigma_value = shift * u / (1.0 + (shift - 1.0) * u)
            sigma_value = float(torch.clamp(sigma_value, min=offline_config.sigma_min, max=1.0))
            sigma = torch.tensor(sigma_value, device=self.device, dtype=torch.float32)
            t = sigma * num_train

            noise = torch.randn_like(clean_tail)
            noisy_tail = ((1.0 - sigma) * clean_tail.float() + sigma * noise.float()).to(self.weight_dtype)
            latents = torch.cat([source_latents, noisy_tail], dim=2).detach().requires_grad_(True)
            started = time.perf_counter()
            output = self._velocity(
                latents, t, prompt, context, rollout_guidance_scale=1.0, loss_guidance_scale=1.0
            )
            prediction = output[1] if isinstance(output, tuple) else output
            timings["student_velocity"] = time.perf_counter() - started
            yield DistillationStep(
                latents[:, :, edit_start:].detach().clone(),
                prediction[:, :, edit_start:],
                t.detach().clone(),
                sigma.detach().clone(),
                num_steps=draws,
                timings=timings,
            )
            timings = {}

    def rollout(self, batch, rollout_config):
        started = time.perf_counter()
        prompt = self._encode_prompt(batch["video_edit_prompt"])
        timings = {"encode_prompt": time.perf_counter() - started}
        started = time.perf_counter()
        source_latents = self._image_latents(batch["source"])
        source_count = source_latents.shape[2]
        latents, edit_start = VideoCoFVideoEditPipeline.initial_latents(
            source_latents, self.config.reasoning_latent_count
        )
        context = VideoCoFSampleContext(source_count, edit_start)
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
                prompt,
                context,
                rollout_guidance_scale=rollout_config.guidance_scale,
                loss_guidance_scale=rollout_config.student_guidance_scale,
            )
            rollout_prediction, loss_prediction = output if isinstance(output, tuple) else (output, output)
            timings["student_velocity"] = time.perf_counter() - started
            target = latents[:, :, edit_start:].detach().clone()
            prediction = loss_prediction[:, :, edit_start:]
            sigma = (t.float() / float(self.scheduler.config.num_train_timesteps)).detach().clone()
            started = time.perf_counter()
            with torch.no_grad():
                latents = self.scheduler.step(rollout_prediction, t, latents, return_dict=False)[0]
            timings["scheduler_step"] = time.perf_counter() - started
            yield DistillationStep(
                target, prediction, t.detach().clone(), sigma, num_steps=len(timesteps), timings=timings
            )
            timings = {}
        return latents[:, :, edit_start:].detach()
