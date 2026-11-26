"""Config for the offline teacher.

It needs only the VAE that puts a stored edit into the student's latent space.
"""

from __future__ import annotations

from dataclasses import dataclass

from i2vt.config import ConfigMixin
from i2vt.utils import dtype_from_name


@dataclass
class OfflineTargetConfig(ConfigMixin):
    component_root: str
    component_config_path: str
    cache_dir: str
    cache_size: int
    weight_dtype: object


def model_config(c) -> OfflineTargetConfig:
    from importlib.resources import files

    return OfflineTargetConfig(
        # Reuse the student's components: the target has to land in the same
        # normalized latent space the student's own source latents live in.
        component_root=c.student.model_kwargs["component_root"],
        component_config_path=str(files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")),
        cache_dir=c.run.cache_dir,
        cache_size=max(1, int(c.teacher.cache_size)),
        weight_dtype=dtype_from_name(c.student.weight_dtype),
    )
