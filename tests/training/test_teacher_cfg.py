"""Teacher CFG follows JoyEdit's own guidance and can be set per edit type and per sub-task."""

import dataclasses
import json

import torch


def test_teacher_cfg_matches_joyedit_native_guidance():
    from i2vt.models.joyedit.modeling_joyedit import joyedit_guidance

    torch.manual_seed(0)
    uncond, cond = torch.randn(1, 2, 16, 1, 4, 4), torch.randn(1, 2, 16, 1, 4, 4)
    # vendor/joyedit/modules/models/pipeline.py, "perform guidance":
    native = uncond + 2.0 * (cond - uncond)
    native = native * (torch.norm(cond, dim=2, keepdim=True) / torch.norm(native, dim=2, keepdim=True))
    out = joyedit_guidance(uncond, cond, 2.0)
    assert torch.allclose(out, native)
    assert torch.allclose(torch.norm(out, dim=2), torch.norm(cond, dim=2), atol=1e-5)
    # CFG 1 is plain conditional prediction.
    assert torch.allclose(joyedit_guidance(uncond, cond, 1.0), cond, atol=1e-6)


def test_sample_task_comes_from_metadata_or_a_task_column():
    from i2vt.data.dataset import _row_task

    assert _row_task({"metadata": json.dumps({"task": "smile", "member": "x.mp4"})}) == "smile"
    assert _row_task({"task": "jump", "metadata": json.dumps({"task": "smile"})}) == "jump"
    assert _row_task({"metadata": "not json"}) is None
    assert _row_task({}) is None


def test_teacher_guidance_tables_reach_the_real_teacher_config():
    from i2vt.models.joyedit.configuration_joyedit import model_config
    from i2vt.models.joyedit.modeling_joyedit import JoyEditTeacherBackend
    from tests.helpers import load_config

    config = load_config()
    config.teacher = dataclasses.replace(
        config.teacher, guidance_scale_by_type={"motion_edit": 3.0}, guidance_scale_by_task={"smile": 1.0}
    )
    # The teacher receives the JoyEditConfig built by model_config(), not the ExperimentConfig.
    joy = model_config(config)
    assert joy.guidance_scale_by_type == {"motion_edit": 3.0}
    assert joy.guidance_scale_by_task == {"smile": 1.0}

    backend = JoyEditTeacherBackend.__new__(JoyEditTeacherBackend)
    backend.config = joy
    assert backend._scale_for({"edit_type": "action_edit", "task": "smile"}) == 1.0
    assert backend._scale_for({"edit_type": "action_edit", "task": "head_down"}) == 2.0
    assert backend._scale_for({"edit_type": "motion_edit", "task": "jump"}) == 3.0
