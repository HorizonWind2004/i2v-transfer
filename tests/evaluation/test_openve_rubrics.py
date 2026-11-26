"""OpenVE-Bench rubrics: official text pinned, our four rubrics, parser slots, rubric-aware resume."""

import hashlib

import pytest

from i2vt.evaluation import rubrics, score
from i2vt.evaluation.rubrics import openve as official

# SHA-256 of the strings in OpenVE-Team/OpenVE-3M@af330a19 OpenVE-Bench/gemini_benchmark.py.
OFFICIAL_SHA256 = {
    "GLOBAL_STYLE": "478ac19a847a7064c203e2a2c94ee22a6428c1c3c64fb22d47bd1c245ed69a7d",
    "CREATIVE_EDIT": "ec68b57b62bac9689f38ca84a32b43ac1bf3e9ab9fe760013ee69949e00ed616",
    "CAMERA_EDIT": "111f30f762048926f3ef53db174cdb5ac22cc2669ac52448daec72fc5917dbc0",
    "LOCAL_CHANGE": "7355ed4dadfd2110dbe939a8809ff8bd564047b44b74ac8530ab2c663af41518",
    "BACKGROUND_CHANGE": "7deaa4925ee969b13cc3c49613edf721abef6d50e0ec2ae5f52b9e3e25dc6ae7",
    "LOCAL_REMOVE": "e1d716d17c7ca8f75fc1200fbd9cc73cc55a9d60249277bd3de17e52e448ab16",
    "LOCAL_ADD": "7ff265eb1698713a46813454fb4e68d177f19d00a1d5e0574386fb9fd903a1c4",
    "SUBTITLE_EDIT": "d5bb9c6ed8eb80162eaded8f343a5072a1b48886d1336bf87340380f4f97aada",
}

OPENVE_CATEGORIES = ["global_style", "local_change", "background_change", "local_remove", "local_add",
                     "subtitle_edit", "camera_edit", "creative_edit"]
OUR_CATEGORIES = ["action_edit", "expression_edit", "motion_edit", "relight"]


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(OFFICIAL_SHA256))
def test_official_rubrics_are_verbatim(name):
    assert _sha(getattr(official, name)) == OFFICIAL_SHA256[name]


def test_openve_categories_use_official_rubrics():
    for cat in OPENVE_CATEGORIES:
        assert score.prompt_type[cat] in vars(official).values()
        assert score.rubric_source(cat) == official.SOURCE


def test_only_four_rubrics_of_our_own():
    extra = set(score.prompt_type) - set(OPENVE_CATEGORIES) - {"subtitles_edit", "camera_multi_shot_edit"}
    assert extra == set(OUR_CATEGORIES)
    for cat in OUR_CATEGORIES:
        assert score.rubric_source(cat) == rubrics.OURS_SOURCE


def test_every_rubric_fills_and_declares_three_slots():
    for cat, template in score.prompt_type.items():
        filled = score.fill_rubric(template, "Remove the red car.")
        assert "Remove the red car." in filled
        assert "{edit_prompt}" not in filled and "<edit_prompt>" not in filled
        labels = score.rubric_labels(filled)
        assert sorted(score.LABEL_SLOT.get(label, i) for i, label in enumerate(labels)) == [0, 1, 2], cat


def test_official_answer_parsed_by_its_own_labels():
    rubric = score.fill_rubric(score.prompt_type["local_remove"], "Remove the cat.")
    answer = ("Brief reasoning: ghosting remains.\n**Prompt Compliance:** 3\n"
              "Visual & Temporal Naturalness: 2\nPhysical & Detail Coherence: 1")
    assert score.check_format(answer, rubric) == [3, 2, 1]
    assert score.check_format(answer) is False  # default labels do not apply to official rubrics


def test_slots_follow_labels_not_line_order():
    # official global_style lists Content Preservation before Temporal Stability
    rubric = score.fill_rubric(score.prompt_type["global_style"], "Make it watercolor.")
    answer = "Style Fidelity: 5\nContent Preservation: 4\nTemporal Stability: 2"
    assert score.check_format(answer, rubric) == [5, 2, 4]
    # our rubrics: default labels, any line order
    answer = "Instruction Compliance: 5\nConsistency & Detail Fidelity: 3\nVisual Quality & Stability: 4"
    assert score.check_format(answer, score.fill_rubric(score.prompt_type["motion_edit"], "Jump.")) == [5, 4, 3]


def test_resume_only_counts_rows_from_the_same_rubric(tmp_path):
    import json

    path = tmp_path / "scores.jsonl"
    rows = [
        {"base": "unlabelled", "edited_type": "local_remove", "protocol": "gemini_video", "scores": [3, 3, 3]},
        {"base": "other", "edited_type": "local_remove", "protocol": "gemini_video", "scores": [3, 3, 3],
         "rubric": rubrics.OURS_SOURCE},
        {"base": "new", "edited_type": "local_remove", "protocol": "gemini_video", "scores": [4, 4, 4],
         "rubric": rubrics.OFFICIAL_SOURCE},
        {"base": "ours", "edited_type": "motion_edit", "protocol": "gemini_video", "scores": [5, 5, 5],
         "rubric": rubrics.OURS_SOURCE},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Only a row scored with the rubric its edit type uses now counts as done.
    assert score.valid_score_bases(path, check_rubric=True) == {"new", "ours"}
    assert score.valid_score_bases(path) == {"unlabelled", "other", "new", "ours"}
