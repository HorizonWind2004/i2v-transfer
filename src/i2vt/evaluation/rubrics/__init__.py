"""Judge rubrics, one table per benchmark family.

- `openve`: the eight OpenVE-Bench categories, verbatim from the OpenVE-3M authors (do not edit).
- `openve_pp`: ours, for OpenVE++ (action change, expression change, body movement, scene relighting).
"""

from . import openve, openve_pp

OFFICIAL_SOURCE = openve.SOURCE
OURS_SOURCE = openve_pp.SOURCE
OURS_CATEGORIES = frozenset(openve_pp.prompt_type)

_openve_prompt_type = {
    "global_style": openve.GLOBAL_STYLE,
    "local_change": openve.LOCAL_CHANGE,
    "background_change": openve.BACKGROUND_CHANGE,
    "local_remove": openve.LOCAL_REMOVE,
    "local_add": openve.LOCAL_ADD,
    "subtitle_edit": openve.SUBTITLE_EDIT,
    "subtitles_edit": openve.SUBTITLE_EDIT,
    "camera_edit": openve.CAMERA_EDIT,
    "camera_multi_shot_edit": openve.CAMERA_EDIT,
    "creative_edit": openve.CREATIVE_EDIT,
}

prompt_type = {**_openve_prompt_type, **openve_pp.prompt_type}


def rubric_source(edited_type):
    """Where the rubric used for `edited_type` comes from; written into every score record."""
    return OURS_SOURCE if edited_type in OURS_CATEGORIES else OFFICIAL_SOURCE


def fill_rubric(template, edit_prompt):
    """Insert the edit instruction: ours use `{edit_prompt}`, the official ones `<edit_prompt>`."""
    return template.replace("{edit_prompt}", edit_prompt).replace("<edit_prompt>", edit_prompt)
