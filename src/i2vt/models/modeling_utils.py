from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from PIL import Image
from safetensors.torch import safe_open


@dataclass(frozen=True)
class PreparedSample:
    """Backend-private initial state and model-specific conditioning."""

    latents: torch.Tensor
    context: Any


@dataclass(frozen=True)
class DistillationStep:
    """Normalized target latent [B,C,T,H,W] and its differentiable prediction.

    noisy_latents is a detached snapshot BEFORE the scheduler update. Source and
    reasoning states are private. timestep retains the model's original time
    encoding; sigma is the flow noise level, not an interchangeable timestep.
    """

    noisy_latents: torch.Tensor
    prediction: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    prediction_type: Literal["velocity", "sample"] = "velocity"
    num_steps: int = 1
    timings: dict[str, float] = field(default_factory=dict)


class Backend(ABC):
    _lora_target_exclusions = ("_self_attn_",)

    @staticmethod
    def image_digest(image: Image.Image) -> str:
        return hashlib.sha1(image.tobytes()).hexdigest()

    @staticmethod
    def hash_key(parts: dict) -> str:
        payload = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _filter_lora_targets(self, network=None):
        # network may be passed explicitly: each expert of a two-expert model has its own LoRA.
        network = self.network if network is None else network
        target = getattr(self.config, "lora_target", "all")
        if target == "all":
            return
        if target == "no_self_attn":
            before = len(network.unet_loras)
            exclusions = self._lora_target_exclusions
            network.unet_loras = [
                lora
                for lora in network.unet_loras
                if not any(exclusion in lora.lora_name for exclusion in exclusions)
            ]
            removed = before - len(network.unet_loras)
            print(
                f"LoRA target no_self_attn: kept {len(network.unet_loras)} U-Net LoRA modules, "
                f"removed {removed} self-attn modules",
                flush=True,
            )
            return
        raise ValueError(f"Unsupported --lora-target={target!r}.")


class StudentBackend(Backend):
    @property
    def trainable_module(self):
        """The module whose train/eval mode tracks the optimizer's parameters.

        Adapter training keeps the base frozen and toggles the adapter; a full
        finetune has no adapter, so backends that support one override this.
        """
        return self.network

    @staticmethod
    def read_lora_rank(path: str) -> int | None:
        if not path.endswith(".safetensors"):
            return None
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("lora_down.weight"):
                    return int(handle.get_tensor(key).shape[0])
        return None

    @abstractmethod
    def params(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, out_dir, step: int):
        raise NotImplementedError

    def vae_signature(self):
        from i2vt.utils import vae_signature

        return vae_signature(self.vae)


class TeacherBackend(Backend):
    prediction_type = "velocity"

    @abstractmethod
    def predict(self, step: DistillationStep, conditioning) -> torch.Tensor:
        """Return a detached target prediction in the same normalized coordinates."""
        raise NotImplementedError

    def submit(self, step, conditioning) -> Future:
        result = Future()
        try:
            result.set_result(self.predict(step, conditioning))
        except Exception as exc:
            result.set_exception(exc)
        return result

    @abstractmethod
    def vae_signature(self):
        raise NotImplementedError

    def close(self):
        pass
