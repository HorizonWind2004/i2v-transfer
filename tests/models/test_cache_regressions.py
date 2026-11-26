from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from i2vt.utils import InMemoryDiskCache


@pytest.mark.parametrize("disk", [False, True])
def test_nested_cache_preserves_attention_boundaries_and_boolean_masks(tmp_path, disk):
    def cache():
        return InMemoryDiskCache(
            namespace="regression",
            root=tmp_path if disk else None,
            cache_size=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

    original = {
        "mask": torch.ones(1, 2323, dtype=torch.int64),
        "nested": [
            (
                torch.tensor([0, 2323, 4646], dtype=torch.int32),
                torch.tensor([True, False]),
                torch.ones(2, dtype=torch.float32),
            )
        ],
    }
    store = cache()
    store.put("sample", original)
    if disk:
        store = cache()  # Simulate restart: no in-memory entries can hide disk dtype conversion.
    loaded = store.get("sample")
    assert loaded["mask"].dtype == torch.int64
    assert loaded["mask"].sum().item() == 2323
    bounds, mask, embedding = loaded["nested"][0]
    assert bounds.dtype == torch.int32 and bounds.tolist() == [0, 2323, 4646]
    assert mask.dtype == torch.bool and mask.tolist() == [True, False]
    assert embedding.dtype == torch.bfloat16
    assert store.get("unseen-sample") is None


@pytest.mark.parametrize("guidance,encodes", [(1.0, 1), (2.0, 2)])
def test_teacher_warm_cache_does_not_hide_encoding_of_new_samples(tmp_path, guidance, encodes):
    from i2vt.models.joyedit.modeling_joyedit import JoyEditTeacherBackend

    class Pipeline:
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1

    teacher = JoyEditTeacherBackend.__new__(JoyEditTeacherBackend)
    teacher.device, teacher.weight_dtype = torch.device("cpu"), torch.bfloat16
    teacher.config = SimpleNamespace(guidance_scale=guidance, model_path="test", config_path="test", rollout_steps=4)
    pipe = teacher.pipeline = Pipeline()
    pipe.vae = SimpleNamespace(config=SimpleNamespace(latent_channels=2))
    pipe.scheduler = SimpleNamespace(set_timesteps=Mock())
    pipe.encode_prompt = Mock(side_effect=lambda **kw: (torch.ones(1, 3, 2), torch.ones(1, 3, dtype=torch.int64)))
    pipe.pad_sequence = lambda x, length: x
    pipe.prepare_latents = Mock(return_value=(torch.ones(1, 2, 2, 1, 2, 2), None))
    pipe.transformer = Mock(side_effect=lambda **kw: (kw["hidden_states"] * 0.5,))
    teacher._cache = InMemoryDiskCache(
        namespace="teacher", root=tmp_path, cache_size=2, device=teacher.device, dtype=teacher.weight_dtype
    )
    step = SimpleNamespace(noisy_latents=torch.ones(1, 2, 1, 2, 2), timestep=torch.tensor([1000.0]))
    conditioning = dict(source=Image.new("RGB", (2, 2)), width=2, height=2, joy_edit_prompt="old")
    cold = teacher.predict(step, conditioning)
    for _ in range(16):
        torch.testing.assert_close(teacher.predict(step, conditioning), cold, rtol=0, atol=0)
    assert pipe.encode_prompt.call_count == encodes
    teacher.predict(step, dict(conditioning, joy_edit_prompt="unseen"))
    assert pipe.encode_prompt.call_count == 2 * encodes
    assert pipe.prepare_latents.call_count == 2
    assert pipe.transformer.call_count == 18


def test_teacher_source_memo_invalidates_on_image_identity_and_size():
    from i2vt.models.joyedit.modeling_joyedit import JoyEditTeacherBackend

    teacher = JoyEditTeacherBackend.__new__(JoyEditTeacherBackend)
    teacher.image_digest = Mock(side_effect=lambda image: image.tobytes().hex())
    source = Image.new("RGB", (2, 2), "red")
    first = teacher._resized_source(source, 2, 2)
    assert teacher._resized_source(source, 2, 2) is first
    assert teacher.image_digest.call_count == 1
    teacher._resized_source(source, 4, 4)
    teacher._resized_source(Image.new("RGB", (2, 2), "blue"), 4, 4)
    assert teacher.image_digest.call_count == 3
