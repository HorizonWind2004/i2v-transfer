from pathlib import Path

import yaml

from i2vt.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
ROOT_ENV = {"I2V_ASSET_ROOT": "/assets", "I2V_OUTPUT_ROOT": "/runs", "I2V_CACHE_ROOT": "/cache"}


def load_raw(backend="bernini"):
    """The fixture config of ``backend`` as a dict, with ROOT_ENV substituted."""
    data = yaml.safe_load((FIXTURES / f"{backend}.yaml").read_text())

    def expand(obj):
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        if isinstance(obj, str):
            for k, v in ROOT_ENV.items():
                obj = obj.replace("${" + k + "}", v)
        return obj

    return expand(data)


def load_config(backend="bernini"):
    return ExperimentConfig.from_dict(load_raw(backend))


def offline_config():
    """The Bernini fixture switched to offline training against stored teacher edits."""
    raw = load_raw("bernini")
    raw["teacher"] = {"backend": "offline"}
    raw["train"]["objective"] = "offline"
    raw["train"]["rollout"] = {"num_inference_steps": 1, "chunk_size": 1}
    return ExperimentConfig.from_dict(raw)
