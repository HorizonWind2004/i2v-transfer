"""Training frames keep the source aspect ratio at the training pixel budget, as evaluation does."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.mark.parametrize("src,expected", [
    ((1280, 720), (688, 384)), ((720, 1280), (384, 688)), ((1024, 768), (592, 448)), ((512, 512), (512, 512)),
])
def test_area_resize_keeps_aspect_at_the_training_pixel_budget(src, expected):
    from i2vt.pipelines.pipeline_utils import training_size

    w, h = training_size(src, height=512, width=512, mode="area")
    assert (w, h) == expected and w % 16 == 0 and h % 16 == 0
    assert abs(w * h - 512 * 512) / (512 * 512) < 0.05
    assert training_size(src, height=512, width=512, mode="squash") == (512, 512)


def test_offline_target_is_resized_to_the_source_grid():
    from diffusers import FlowMatchEulerDiscreteScheduler

    from i2vt.config import OfflineConfig
    from i2vt.models.bernini.modeling_bernini import BerniniStudentBackend

    student = BerniniStudentBackend.__new__(BerniniStudentBackend)
    student.device, student.weight_dtype = torch.device("cpu"), torch.float32
    student.config = SimpleNamespace(width=512, height=512, flow_shift=5.0, resize_mode="area")
    student.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    sizes = []

    def latents(im, size=None):
        sizes.append(size)
        return torch.randn(1, 16, 1, 4, 4)

    student._image_latents = latents
    student._encode_prompt = lambda p: (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
    student._velocity = lambda noisy, t, pe, ne, src, **kw: noisy
    batch = {"source": Image.new("RGB", (1280, 720)), "target": Image.new("RGB", (1024, 1024)),
             "video_edit_prompt": "edit"}
    next(iter(student.offline_steps(batch, SimpleNamespace(num_inference_steps=1, train_steps=0), OfflineConfig())))
    assert sizes == [(688, 384), (688, 384)]


def test_invalid_resize_mode_is_rejected():
    from tests.helpers import offline_config

    config = offline_config()
    config.data.resize_mode = "stretch"
    with pytest.raises(ValueError):
        config.validate()
