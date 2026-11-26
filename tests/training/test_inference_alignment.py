"""Training feeds the student and the teacher the conditions their official inference uses."""

from __future__ import annotations

import html
import re
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


def _official_prompt_clean(text: str) -> str:
    """Copied from bytedance/Bernini bernini/pipeline.py::_prompt_clean."""
    import ftfy

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("instruction", [
    "Make the person smile.",
    "  Make   the person\nsmile  gently. ",
    "Replace the sign with &amp;quot;OPEN&amp;quot;.",
    "cafÃ© sign turns red",
])
def test_v2v_prompt_matches_official_construction(instruction):
    from i2vt.models.bernini.inference_config import V2V_SYSTEM_PROMPT, v2v_model_prompt

    # Official: SYSTEM_PROMPTS["v2v"] + _prompt_clean(prompt).
    assert V2V_SYSTEM_PROMPT == "You are a helpful assistant specialized in video editing."
    assert v2v_model_prompt(instruction) == V2V_SYSTEM_PROMPT + _official_prompt_clean(instruction)


def test_training_encodes_the_same_prompt_as_inference(monkeypatch):
    from i2vt.models.bernini import modeling_bernini as mb
    from i2vt.models.bernini.inference_config import OFFICIAL_WAN_NEGATIVE_PROMPT, v2v_model_prompt

    seen = []
    monkeypatch.setattr(mb.BerniniVideoEditPipeline, "encode_text",
                        staticmethod(lambda tok, enc, prompt, **kw: seen.append(prompt) or torch.zeros(1, 4, 2)))
    student = mb.BerniniStudentBackend.__new__(mb.BerniniStudentBackend)
    student.config = SimpleNamespace(max_sequence_length=512, component_root="x")
    student.tokenizer = student.text_encoder = None
    student.device = torch.device("cpu")
    student.weight_dtype = torch.float32
    student._prompt_cache = SimpleNamespace(get=lambda k: None, put=lambda k, v: None)
    student._encode_prompt("  Make the person   smile. ")
    assert seen == [v2v_model_prompt("  Make the person   smile. "), OFFICIAL_WAN_NEGATIVE_PROMPT]


def test_training_text_embeds_are_zero_past_the_prompt_like_the_official_renderer():
    from i2vt.pipelines.bernini.pipeline_bernini import BerniniVideoEditPipeline

    class Tok:
        def __call__(self, prompt, max_length, **kw):
            mask = torch.zeros(1, max_length, dtype=torch.long)
            mask[:, :3] = 1
            return SimpleNamespace(input_ids=torch.zeros(1, max_length, dtype=torch.long), attention_mask=mask)

    def encoder(input_ids, attention_mask):
        # A real T5 returns non-zero vectors at pad positions too.
        return (torch.ones(1, input_ids.shape[1], 4),)

    embeds = BerniniVideoEditPipeline.encode_text(Tok(), encoder, "p", max_length=8,
                                                  device=torch.device("cpu"), dtype=torch.float32)
    assert embeds.shape == (1, 8, 4)
    assert torch.all(embeds[:, :3] == 1)
    assert torch.all(embeds[:, 3:] == 0)


def test_joyedit_bucket_crop_is_a_no_op_on_an_already_bucketed_image():
    pytest.importorskip("torchvision")
    from i2vt.vendor.joyedit.modules.utils import _dynamic_resize_from_bucket

    src = Image.effect_noise((1280, 720), 64).convert("RGB")
    once = _dynamic_resize_from_bucket(src, basesize=1024)
    twice = _dynamic_resize_from_bucket(once, basesize=1024)
    assert twice.size == once.size
    assert list(twice.getdata()) == list(once.getdata())
