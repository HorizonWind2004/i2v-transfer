"""A teacher that answers from a stored edit instead of running a model.

Online training asks the image editor for the velocity at the noisy latent the student reached.
Offline training regresses onto the teacher's stored edit instead. Expressing the difference as
another teacher keeps the loss types, gradient accumulation, logging, checkpointing and validation
previews the same for both.

The velocity needs no stored noise. Flow matching defines

    x_t = (1 - sigma) * x0 + sigma * eps
    v   = eps - x0

and subtracting gives x_t - x0 = sigma * (eps - x0), so

    v = (x_t - x0) / sigma

is recoverable from the student's own noisy latent, its sigma, and the target
latent alone, the same algebra `distillation_loss` uses to turn a sample-space prediction
into a velocity.
"""

from __future__ import annotations

import torch
from PIL import Image

from i2vt.models.modeling_utils import TeacherBackend
from i2vt.utils import InMemoryDiskCache, resolve_model_path


class OfflineTeacherBackend(TeacherBackend):
    """Return the exact velocity that carries a noisy latent to a stored edit."""

    prediction_type = "velocity"

    def __init__(self, config, device):
        from omegaconf import OmegaConf

        from i2vt.vendor.wan.models.wan_vae import AutoencoderKLWan

        self.config = config
        self.device = device
        self.weight_dtype = config.weight_dtype

        # The same VAE the Bernini student builds, from the same component root,
        # so the target lands in the one normalized latent space the student's
        # own latents live in. assert_compatible_vae checks this at startup.
        component_config = OmegaConf.load(config.component_config_path)
        self.vae = AutoencoderKLWan.from_pretrained(
            resolve_model_path(
                config.component_root,
                component_config["vae_kwargs"].get("vae_subpath", "vae"),
                "vae",
            ),
            additional_kwargs=OmegaConf.to_container(component_config["vae_kwargs"]),
        ).to(device=device, dtype=self.weight_dtype)
        self.vae.eval().requires_grad_(False)

        self._latent_cache = InMemoryDiskCache(
            namespace="offline_target_latent",
            root=config.cache_dir,
            cache_size=max(1, int(config.cache_size)),
            device=device,
            dtype=self.weight_dtype,
        )

    def params(self):
        return []

    def save(self, out_dir, step: int):
        # Nothing here is trained; the student owns every parameter that moves.
        return None

    def vae_signature(self):
        from i2vt.utils import vae_signature

        return vae_signature(self.vae)

    @torch.no_grad()
    def _target_latents(self, target: Image.Image, height: int, width: int):
        from i2vt.pipelines.bernini.pipeline_bernini import BerniniVideoEditPipeline

        # Resized to the source's training size, so the target has the student's geometry.
        resized = target.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        key = self.image_digest(resized)
        cached = self._latent_cache.get(key)
        if cached is not None:
            return cached
        latents = BerniniVideoEditPipeline.image_to_latents(
            self.vae, resized, width=width, height=height, device=self.device, dtype=self.weight_dtype
        )
        self._latent_cache.put(key, latents)
        return latents

    @torch.no_grad()
    def predict(self, step, conditioning) -> torch.Tensor:
        target = conditioning.get("target")
        if target is None:
            raise ValueError(
                "the offline teacher needs a 'target' image on every row; "
                "the configured dataset has no target column"
            )
        latents = self._target_latents(target, conditioning["height"], conditioning["width"])
        noisy = step.noisy_latents.to(device=self.device, dtype=torch.float32)
        target_latents = latents.to(device=self.device, dtype=torch.float32)
        if target_latents.shape != noisy.shape:
            raise RuntimeError(
                f"offline target shape {tuple(target_latents.shape)} does not match "
                f"the student latent {tuple(noisy.shape)}"
            )
        sigma = step.sigma.to(device=self.device, dtype=torch.float32)
        if torch.any(sigma <= 0):
            # v = (x_t - x0) / sigma is undefined at sigma 0.
            raise ValueError("the offline teacher requires sigma > 0")
        velocity = (noisy - target_latents) / sigma
        return velocity.to(dtype=step.prediction.dtype).detach()
