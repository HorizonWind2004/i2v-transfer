"""Distillation losses."""

from .distillation import distillation_loss, velocity_channel_cosine_loss, x0_channel_cosine_loss

__all__ = ["distillation_loss", "velocity_channel_cosine_loss", "x0_channel_cosine_loss"]
