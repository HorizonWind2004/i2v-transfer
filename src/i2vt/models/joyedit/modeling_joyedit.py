from __future__ import annotations

import torch
from PIL import Image

from i2vt.models.modeling_utils import TeacherBackend
from i2vt.utils import InMemoryDiskCache, vae_signature

from .configuration_joyedit import JoyEditConfig


def joyedit_guidance(uncond, cond, guidance_scale):
    """CFG exactly as JoyEdit's own sampler does it.

    vendor/joyedit/modules/models/pipeline.py ("perform guidance") takes the
    usual uncond + g * (cond - uncond) and then rescales it back to the norm of
    the conditional velocity, per token along the channel axis (dim 2 of
    [B, items, C, ...]). Without the rescale a CFG>1 teacher hands the student a
    velocity larger than any JoyEdit ever integrates with; with g == 1 the two
    are identical.
    """
    guided = uncond + guidance_scale * (cond - uncond)
    cond_norm = torch.norm(cond, dim=2, keepdim=True)
    guided_norm = torch.norm(guided, dim=2, keepdim=True)
    return guided * (cond_norm / guided_norm)

class JoyEditTeacherBackend(TeacherBackend):
    def __init__(self, config: JoyEditConfig, device: torch.device):
        self.config = config
        self.device = device
        from i2vt.vendor.joyedit.infer_runtime.model import build_model
        from i2vt.vendor.joyedit.infer_runtime.settings import load_settings

        settings = load_settings(
            ckpt_root=config.model_path,
            config_path=config.config_path,
            rewrite_model="gpt-5",
            default_seed=config.seed,
        )
        runtime = build_model(settings, device=device)
        self.pipeline = runtime.pipeline
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.transformer.eval().requires_grad_(False)
        self.pipeline.text_encoder.eval().requires_grad_(False)
        self.pipeline.vae.eval().requires_grad_(False)
        self.weight_dtype = next(self.pipeline.transformer.parameters()).dtype
        self.cache_size = max(0, int(getattr(config, "cache_size", 8)))
        self._cache = InMemoryDiskCache(
            namespace="joy_teacher",
            cache_size=self.cache_size,
            root=getattr(config, "cache_dir", None),
            device=self.device,
            dtype=self.weight_dtype,
        )

    def vae_signature(self):
        return vae_signature(self.pipeline.vae)

    def _resized_source(self, source, width, height):
        """Resize the source frame and hash it once per sample.

        Every rollout step of a sample conditions on the same frame, and the
        resize plus the SHA-1 over 786 KB of pixels cost ~11 ms each time. The
        memo keeps a reference to the image so `is` cannot alias a freed image
        onto a new one at the same address.
        """
        key = (source, width, height)
        cached_key, cached = getattr(self, "_source_memo", (None, None))
        if cached_key is None or cached_key[0] is not source or cached_key[1:] != key[1:]:
            image = source.resize((width, height), Image.Resampling.BICUBIC)
            cached = (image, self.image_digest(image))
            self._source_memo = (key, cached)
        return cached

    def _scale_for(self, conditioning):
        """Teacher CFG for this sample: looked up by edit_type, falling back to the scalar default.

        edit_type comes from the dataset row through the trainer's conditioning unchanged (the trainer
        merges the whole row into conditioning), so this is the type of the sample being trained on.
        """
        # getattr: config stand-ins in tests carry only required fields and should not need the optional ones.
        by_task = getattr(self.config, "guidance_scale_by_task", None)
        if by_task and conditioning.get("task") in by_task:
            return float(by_task[conditioning["task"]])
        by_type = getattr(self.config, "guidance_scale_by_type", None)
        if not by_type:
            return float(self.config.guidance_scale)
        edit_type = conditioning.get("edit_type")
        if isinstance(edit_type, (list, tuple)):
            edit_type = edit_type[0] if edit_type else None
        return float(by_type.get(str(edit_type), self.config.guidance_scale))

    @torch.no_grad()
    def predict(self, step, conditioning):
        pipeline = self.pipeline
        guidance_scale = self._scale_for(conditioning)
        pipeline._guidance_scale = guidance_scale
        prompt = conditioning["joy_edit_prompt"]
        source = conditioning.get("joyedit_source") or conditioning["source"]
        height = conditioning["height"]
        width = conditioning["width"]
        timestep = step.timestep.to(self.device).float().reshape(1)
        latent = step.noisy_latents.to(self.device)

        image, image_sha1 = self._resized_source(source, width, height)
        cache_key = self.hash_key(
            {
                "kind": "joy_teacher_conditioning_v1",
                "prompt": prompt,
                "height": height,
                "width": width,
                "teacher_guidance_scale": guidance_scale,
                "do_cfg": pipeline.do_classifier_free_guidance,
                "image_sha1": image_sha1,
                "joy_ckpt_root": str(self.config.model_path),
                "joy_config": str(self.config.config_path),
                "dtype": str(self.weight_dtype),
            }
        )

        entry = self._cache.get(cache_key)
        if entry is None:
            images = [image]
            image_tokens = "<image>\n"
            prompt_list = [f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"]
            neg_list = [f"<|im_start|>user\n{image_tokens}<|im_end|>\n"]

            prompt_embeds, prompt_attention_mask = pipeline.encode_prompt(
                prompt=prompt_list,
                images=images,
                device=self.device,
                num_videos_per_prompt=1,
                max_sequence_length=4096,
                template_type="image",
            )
            if pipeline.do_classifier_free_guidance:
                negative_prompt_embeds, negative_prompt_attention_mask = pipeline.encode_prompt(
                    prompt=neg_list,
                    images=images,
                    device=self.device,
                    num_videos_per_prompt=1,
                    max_sequence_length=4096,
                    template_type="image",
                )
                max_length = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
                prompt_embeds = torch.cat(
                    [
                        pipeline.pad_sequence(negative_prompt_embeds, max_length),
                        pipeline.pad_sequence(prompt_embeds, max_length),
                    ]
                )
                prompt_attention_mask = torch.cat(
                    [
                        pipeline.pad_sequence(negative_prompt_attention_mask, max_length),
                        pipeline.pad_sequence(prompt_attention_mask, max_length),
                    ]
                )

            full_latent, condition = pipeline.prepare_latents(
                batch_size=1,
                num_items=2,
                num_channels_latents=pipeline.vae.config.latent_channels,
                height=height,
                width=width,
                video_length=1,
                dtype=self.weight_dtype,
                device=self.device,
                generator=None,
                reference_images=[image],
            )
            prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype).detach()
            prompt_attention_mask = prompt_attention_mask.detach()
            reference_latent = full_latent[:, :1].detach()
            condition = condition.detach() if condition is not None else None
            entry = {
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "reference_latent": reference_latent,
                "condition": condition,
            }
            self._cache.put(cache_key, entry)

        prompt_embeds = entry["prompt_embeds"]
        prompt_attention_mask = entry["prompt_attention_mask"]
        reference_latent = entry["reference_latent"]
        condition = entry["condition"]

        pipeline.scheduler.set_timesteps(self.config.rollout_steps, device=self.device)
        latent = latent.unsqueeze(1) if latent.ndim == 5 else latent
        full_latent = torch.cat([reference_latent, latent.to(device=self.device, dtype=self.weight_dtype)], dim=1)
        model_latent = torch.cat([full_latent, condition], dim=2) if condition is not None else full_latent
        model_input = torch.cat([model_latent] * 2) if pipeline.do_classifier_free_guidance else model_latent
        model_input = model_input.to(dtype=self.weight_dtype)
        t = timestep.to(self.device, dtype=torch.float32).repeat(model_input.shape[0])
        prediction = pipeline.transformer(
            hidden_states=model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_attention_mask,
            return_dict=False,
        )[0]
        if pipeline.do_classifier_free_guidance:
            uncond, cond = prediction.chunk(2)
            prediction = joyedit_guidance(uncond, cond, guidance_scale)
        prediction = prediction[:, -1:]
        return prediction.squeeze(1).detach() if step.noisy_latents.ndim == 5 else prediction.detach()
