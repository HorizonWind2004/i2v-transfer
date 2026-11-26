from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from i2vt.models.bernini.inference_config import (
    OFFICIAL_WAN_NEGATIVE_PROMPT,
    V2V_SYSTEM_PROMPT,
)
from i2vt.models.bernini.modeling_bernini import BerniniStudentBackend
from i2vt.utils import InMemoryDiskCache, assert_compatible_vae, vae_signature


class ModelContractTest(unittest.TestCase):
    def test_bernini_training_uses_the_same_prompts_inference_hardcodes(self) -> None:
        student = BerniniStudentBackend.__new__(BerniniStudentBackend)
        student.config = SimpleNamespace(
            max_sequence_length=512,
            component_root="/components",
        )
        student.cache_size = 2
        student.device = torch.device("cpu")
        student.weight_dtype = torch.float32
        student._prompt_cache = InMemoryDiskCache(
            namespace="test", root=None, cache_size=2, device=student.device, dtype=student.weight_dtype
        )
        student.tokenizer = object()
        student.text_encoder = object()
        encoded: list[str] = []

        def fake_encode(tokenizer, text_encoder, prompt: str, **kwargs) -> torch.Tensor:
            encoded.append(prompt)
            return torch.tensor([len(encoded)], dtype=torch.float32)

        from unittest.mock import patch

        from i2vt.pipelines.bernini.pipeline_bernini import BerniniVideoEditPipeline

        with patch.object(BerniniVideoEditPipeline, "encode_text", side_effect=fake_encode):
            first = student._encode_prompt("Apply watercolor style.")
            second = student._encode_prompt("Apply watercolor style.")
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        # sample_video builds the conditional prompt this way and passes
        # OFFICIAL_WAN_NEGATIVE_PROMPT as the negative, with no way to opt out.
        # Training has to construct exactly the same pair, or the LoRA is
        # optimised for a condition that never occurs at evaluation time.
        self.assertEqual(
            encoded,
            [
                V2V_SYSTEM_PROMPT + "Apply watercolor style.",
                OFFICIAL_WAN_NEGATIVE_PROMPT,
            ],
        )

    def test_bernini_prompt_construction_is_not_configurable(self) -> None:
        # Inference always uses the system prompt and the official negative prompt, so training
        # must not offer a switch for either.
        import inspect

        from i2vt.models.bernini import modeling_bernini
        from i2vt.models.bernini.configuration_bernini import BerniniConfig
        from i2vt.models.bernini.inference_config import RolloutKwargs

        fields = set(BerniniConfig.__dataclass_fields__) | set(RolloutKwargs.__dataclass_fields__)
        self.assertEqual(fields & {"use_v2v_system_prompt", "use_official_negative_prompt"}, set())

        source = inspect.getsource(modeling_bernini.BerniniStudentBackend._encode_prompt)
        self.assertNotIn("use_v2v_system_prompt", source)
        self.assertNotIn("use_official_negative_prompt", source)

    def test_vae_signature_checks_latent_normalization_with_dtype_tolerance(self) -> None:
        class DummyVae:
            def __init__(self, dtype, std_delta=0.0):
                self.config = SimpleNamespace(
                    latent_channels=2,
                    spatial_compression_ratio=8,
                    temporal_compression_ratio=4,
                )
                self.mean = torch.tensor([-0.7571, 1.5508], dtype=dtype)
                self.std = torch.tensor([2.8184 + std_delta, 2.0743], dtype=dtype)

        student_sig = vae_signature(DummyVae(torch.float32))
        teacher_sig = vae_signature(DummyVae(torch.bfloat16))
        assert_compatible_vae(student_sig, teacher_sig)
        with self.assertRaises(RuntimeError):
            assert_compatible_vae(student_sig, vae_signature(DummyVae(torch.float32, std_delta=0.1)))


if __name__ == "__main__":
    unittest.main()


def test_zero_memory_cache_can_persist_without_key_error(tmp_path):
    cache = InMemoryDiskCache(namespace="test", root=str(tmp_path), cache_size=0, device="cpu", dtype=torch.float32)
    cache.put("sample", torch.tensor([2.0], requires_grad=True))
    assert "sample" not in cache
    torch.testing.assert_close(cache.get("sample"), torch.tensor([2.0]))
    assert not cache.get("sample").requires_grad
    disabled = InMemoryDiskCache(namespace="test", root=None, cache_size=0, device="cpu", dtype=torch.float32)
    disabled.put("sample", torch.tensor([2.0]))
    assert disabled.get("sample") is None


def test_joyedit_preserves_prediction_shape_with_and_without_item_axis():
    from unittest.mock import Mock

    from PIL import Image

    from i2vt.models.joyedit.modeling_joyedit import JoyEditTeacherBackend
    from i2vt.models.modeling_utils import DistillationStep

    for shape in [(1, 2, 1, 2, 2), (1, 1, 2, 1, 2, 2)]:
        for guidance_scale in [1.0, 2.0]:
            teacher = JoyEditTeacherBackend.__new__(JoyEditTeacherBackend)
            teacher.device = torch.device("cpu")
            teacher.weight_dtype = torch.float32
            teacher.config = SimpleNamespace(
                guidance_scale=guidance_scale, rollout_steps=2, model_path="test", config_path=None
            )
            teacher._cache = Mock()
            teacher._cache.get.return_value = {
                "prompt_embeds": torch.ones(1, 2, 3),
                "prompt_attention_mask": torch.ones(1, 2),
                "reference_latent": torch.zeros(1, 1, 2, 1, 2, 2),
                "condition": None,
            }
            teacher.pipeline = SimpleNamespace(
                do_classifier_free_guidance=guidance_scale > 1,
                scheduler=Mock(),
                transformer=lambda **kwargs: (kwargs["hidden_states"] * 2,),
            )
            latent = torch.ones(shape, requires_grad=True)
            step = DistillationStep(latent, latent, torch.tensor(500.0), torch.tensor(0.5))
            result = teacher.predict(
                step, {"joy_edit_prompt": "paint", "source": Image.new("RGB", (2, 2)), "height": 2, "width": 2}
            )
            torch.testing.assert_close(result, latent.detach() * 2)
            assert not result.requires_grad
