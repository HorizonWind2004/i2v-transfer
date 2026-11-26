"""The scored video uses the same near-lossless encoder settings as the comparison video beside it."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from i2vt.evaluation import infer

ENCODER_KEYS = {"codec", "quality", "macro_block_size", "output_params"}


def call_args(backend, tensor):
    with patch.object(infer.imageio, "mimwrite") as mw, patch.object(infer.imageio, "mimsave") as ms:
        infer._write_video(tensor, "/tmp/x.mp4", 16, backend)
        assert not ms.called, "videos must not be written with the imageio defaults"
        assert mw.called
        return mw.call_args


@pytest.mark.parametrize("backend", ["bernini", "videocof"])
def test_both_backends_use_the_same_encoder_settings(backend):
    t = torch.zeros(1, 3, 2, 8, 8)
    kwargs = call_args(backend, t).kwargs
    assert ENCODER_KEYS <= set(kwargs), f"{backend} is missing encoder arguments: {ENCODER_KEYS - set(kwargs)}"
    assert kwargs["codec"] == "libx264"
    assert kwargs["quality"] == 10
    assert "-crf" in kwargs["output_params"] and "8" in kwargs["output_params"]


def test_encoder_settings_match_between_backends():
    t = torch.zeros(1, 3, 2, 8, 8)
    a = {k: v for k, v in call_args("bernini", t).kwargs.items() if k in ENCODER_KEYS}
    b = {k: v for k, v in call_args("videocof", t).kwargs.items() if k in ENCODER_KEYS}
    assert a == b, "both branches must use the same encoder settings, or one run's two videos differ in quality"


def test_videocof_rounds_instead_of_truncating():
    """0.7/255 truncates to 0 and rounds to 1. Truncation darkens every pixel systematically, by up to 1/255.

    0.5 is avoided on purpose: numpy rounds half to even, so 0.5 would land on 0
    and the test could not tell truncation from rounding.
    """
    t = torch.full((1, 3, 2, 4, 4), 0.7 / 255)
    arr = call_args("videocof", t).args[1]
    assert arr.dtype == np.uint8
    assert arr.max() == 1, f"expected rounding to 1, got {arr.max()} (truncation would give 0)"


def test_videocof_frame_layout_unchanged():
    # (B, C, T, H, W) -> (T, H, W, C). Changing the encoding must not change the layout
    t = torch.zeros(1, 3, 5, 8, 6)
    arr = call_args("videocof", t).args[1]
    assert arr.shape == (5, 8, 6, 3)
