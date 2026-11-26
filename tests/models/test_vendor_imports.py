import importlib
import sys
from importlib.resources import files

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "i2vt.vendor.bernini.models",
        "i2vt.vendor.wan.pipeline.pipeline_wan",
        "i2vt.vendor.joyedit.infer_runtime.model",
    ],
)
def test_vendor_models_import_without_source_path_injection(module):
    before = list(sys.path)
    importlib.import_module(module)
    assert sys.path == before


def test_component_yaml_is_package_resource():
    path = files("i2vt.vendor.wan").joinpath("config/wan2.1/wan_civitai.yaml")
    assert "vae_kwargs" in path.read_text()


def test_joyedit_checkpoint_config_supports_original_imports(tmp_path):
    from i2vt.vendor.joyedit.infer_runtime.infer_config import load_infer_config_class_from_pyfile

    path = tmp_path / "infer_config.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "from infer_runtime.infer_config import InferConfig\n"
        "@dataclass\nclass TestConfig(InferConfig):\n    text_token_max_length: int = 123\n"
    )
    cls = load_infer_config_class_from_pyfile(str(path))
    assert cls().text_token_max_length == 123
    assert "infer_runtime" not in sys.modules


def test_joyedit_config_class_paths_resolve_inside_the_package():
    from i2vt.vendor.joyedit.modules.models.scheduler import FlowMatchDiscreteScheduler
    from i2vt.vendor.joyedit.modules.utils.utils import get_obj_from_str

    assert get_obj_from_str("modules.models.scheduler.FlowMatchDiscreteScheduler") is FlowMatchDiscreteScheduler
