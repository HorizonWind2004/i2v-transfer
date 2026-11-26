"""Small shared utilities: caches, paths, rollout timesteps and VAE checks."""

from __future__ import annotations

import inspect
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


# Cache ----------------------------------------------------------------------
class InMemoryDiskCache:
    """Small LRU memory cache with optional per-namespace on-disk fallback."""

    def __init__(
        self, *, namespace: str, cache_size: int, root: Path | None, device: torch.device, dtype: torch.dtype
    ):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = max(0, int(cache_size))
        self._device = device
        self._dtype = dtype
        self._cache_dir = Path(root) / namespace if root is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get(self, key: str) -> Any | None:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
            return self._to_device_dtype(value)
        if self._cache_dir is None:
            return None
        value = self._load_disk(key)
        if value is None:
            return None
        self._cache[key] = value
        self._cache.move_to_end(key)
        self._trim_memory()
        return self._to_device_dtype(value)

    def put(self, key: str, value: Any) -> None:
        value = self._to_cpu_copy(value)
        self._cache[key] = value
        self._cache.move_to_end(key)
        self._trim_memory()
        self._save_disk(key, value)

    def _trim_memory(self) -> None:
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _disk_path(self, key: str) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("Disk cache is disabled")
        return self._cache_dir / f"{key}.pt"

    def _to_cpu_copy(self, value: Any):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu_copy(v) for v in value)
        return value

    def _to_device_dtype(self, value: Any):
        if torch.is_tensor(value):
            # Only floating tensors take the model's dtype. bfloat16 holds integers exactly only up
            # to 256, so casting an attention mask to it would change the sequence lengths it encodes.
            if value.is_floating_point():
                return value.to(device=self._device, dtype=self._dtype)
            return value.to(device=self._device)
        if isinstance(value, dict):
            return {k: self._to_device_dtype(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_device_dtype(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_device_dtype(v) for v in value)
        return value

    def _load_disk(self, key: str):
        path = self._disk_path(key)
        if not path.exists():
            return None
        try:
            try:
                payload = torch.load(path, map_location=self._device, weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location=self._device)
        except Exception:
            return None
        return self._to_cpu_copy(payload)

    def _save_disk(self, key: str, value: Any) -> None:
        if self._cache_dir is None:
            return
        path = self._disk_path(key)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(self._to_cpu_copy(value), tmp)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()


# Paths ----------------------------------------------------------------------
def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid}


UNRESOLVED_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def expand_env_path(value: str, *, what: str = "path") -> str:
    expanded = os.path.expandvars(str(value))
    missing = sorted({m.group(1) for m in UNRESOLVED_VAR.finditer(expanded)})
    if missing:
        raise SystemExit(
            f"{what} references unset environment variable(s): {', '.join(missing)}. "
            "Set them (see configs/README.md) or pass an explicit path."
        )
    return expanded


def resolve_model_path(model_root: str, subpath: str | None, default_subpath: str) -> str:
    model_root = expand_env_path(model_root, what="model root")
    subpath = expand_env_path(subpath or default_subpath, what="model subpath")
    if os.path.isabs(subpath):
        return subpath
    direct = os.path.join(model_root, subpath)
    if os.path.exists(direct):
        return direct
    sibling = os.path.join(os.path.dirname(os.path.abspath(model_root)), subpath)
    if os.path.exists(sibling):
        return sibling
    return subpath


# Timesteps ------------------------------------------------------------------
def _uniform_random_timesteps(
    scheduler, *, steps: int, device: torch.device | str, low: int, high: int
) -> list[float]:
    if steps <= 0:
        raise ValueError(f"rollout steps must be positive, got {steps}")
    if low > high:
        raise ValueError(f"invalid rollout timestep range: low={low} high={high}")
    count = high - low + 1
    if count < steps:
        raise ValueError(f"rollout timestep range [{low}, {high}] has only {count} values, fewer than steps={steps}")
    return [
        float(x.item()) for x in torch.sort(torch.randperm(count, device=device)[:steps], descending=True).values + low
    ]


def _uniform_grid_timesteps(*, steps: int, low: int, high: int) -> list[float]:
    if steps <= 0:
        raise ValueError(f"rollout steps must be positive, got {steps}")
    if low > high:
        raise ValueError(f"invalid rollout timestep range: low={low} high={high}")
    return [float(x) for x in torch.linspace(high, low, steps).round().tolist()]


def configure_rollout_timesteps(
    scheduler,
    *,
    steps: int,
    device: torch.device | str,
    mode: str = "scheduler",
    min_timestep: int = 1,
    max_timestep: int = 0,
    mu: float = 1.0,
) -> list[float]:
    mode = (mode or "scheduler").strip().lower()
    if mode == "scheduler":
        scheduler.set_timesteps(steps, device=device, mu=mu)
        return [float(x.item()) for x in scheduler.timesteps]
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    low = max(1, int(min_timestep))
    high = min(int(max_timestep) if int(max_timestep) > 0 else num_train_timesteps, num_train_timesteps)
    if mode == "uniform_random":
        timesteps = _uniform_random_timesteps(scheduler, steps=steps, device=device, low=low, high=high)
    elif mode == "uniform_grid":
        timesteps = _uniform_grid_timesteps(steps=steps, low=low, high=high)
    else:
        raise ValueError(f"unknown rollout timestep mode: {mode!r}")
    scheduler.set_timesteps(device=device, timesteps=timesteps, mu=mu)
    scheduler.timesteps = scheduler.sigmas[:-1].to(device=device) * float(num_train_timesteps)
    return [float(x.item()) for x in scheduler.timesteps]


# VAE ------------------------------------------------------------------------
def vae_signature(vae) -> dict[str, Any]:
    config = getattr(vae, "config", None)

    def read(name: str, default=None):
        if config is not None:
            if hasattr(config, name):
                return getattr(config, name)
            if isinstance(config, dict) and name in config:
                return config[name]
        return getattr(vae, name, default)

    def numeric_tuple(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().flatten().tolist()
        return tuple(float(item) for item in value)

    return {
        "class": vae.__class__.__name__,
        "latent_channels": read("latent_channels"),
        "spatial_compression_ratio": read("spatial_compression_ratio", read("ffactor_spatial")),
        "temporal_compression_ratio": read("temporal_compression_ratio", read("ffactor_temporal")),
        "latents_mean": numeric_tuple(read("latents_mean", getattr(vae, "mean", None))),
        "latents_std": numeric_tuple(read("latents_std", getattr(vae, "std", None))),
    }


def assert_compatible_vae(student_sig: dict[str, Any], teacher_sig: dict[str, Any]):
    exact_keys = ["latent_channels", "spatial_compression_ratio", "temporal_compression_ratio"]
    mismatches = {
        key: (student_sig.get(key), teacher_sig.get(key))
        for key in exact_keys
        if student_sig.get(key) != teacher_sig.get(key)
    }
    for key in ("latents_mean", "latents_std"):
        student_value, teacher_value = student_sig.get(key), teacher_sig.get(key)
        compatible = (
            student_value is not None
            and teacher_value is not None
            and len(student_value) == len(teacher_value)
            and all(abs(a - b) <= 1e-2 for a, b in zip(student_value, teacher_value))
        )
        if not compatible:
            mismatches[key] = (student_value, teacher_value)
    if mismatches:
        raise RuntimeError(f"VAE mismatch: student={student_sig}, teacher={teacher_sig}, mismatches={mismatches}")


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
