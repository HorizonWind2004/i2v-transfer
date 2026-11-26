"""Configuration and launch contracts; these tests require no ML runtime."""

import dataclasses
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from i2vt.config import ExperimentConfig
from tests.helpers import ROOT, ROOT_ENV, load_config, load_raw

RELEASED = sorted((ROOT / "configs" / "train").glob("*.yaml"))


def load_released(path, monkeypatch):
    for key, value in ROOT_ENV.items():
        monkeypatch.setenv(key, value)
    return ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize("backend,lr,steps,loss", [("bernini", 2e-5, 100, "v_cos"), ("videocof", 5e-5, 200, "x")])
def test_training_and_pipeline_receive_effective_parameters(backend, lr, steps, loss):
    c = load_config(backend)
    assert (c.train.learning_rate, c.train.max_steps, c.train.loss_type) == (lr, steps, loss)
    model = importlib.import_module(f"i2vt.models.{backend}.configuration_{backend}").model_config(c)
    pipeline = importlib.import_module(f"i2vt.pipelines.{backend}.configuration_{backend}").pipeline_config(c)
    assert model.lora_path == (None if backend == "bernini" else "/assets/checkpoints/VideoCoF/videocof.safetensors")
    assert model.rollout_steps == 10
    assert model.rollout_guidance_scale == 1.0
    assert pipeline.lora_path == str(c.checkpoint_path)
    assert pipeline.num_inference_steps == (40 if backend == "bernini" else 4)
    assert pipeline.source_frames == 81 and pipeline.fps == 16
    assert pipeline.shift == 5.0
    assert model.lora_rank == 128 and model.lora_alpha == 64.0
    assert c.train.gradient_accumulation_steps == (1 if backend == "bernini" else 8)
    assert c.runtime.student_gpus == [2, 3, 4, 5, 6, 7]
    if backend == "bernini":
        assert model.flow_shift == 5.0
        assert pipeline.guidance_mode == "v2v_apg" and pipeline.omega_vid == 1.25
        assert pipeline.norm_threshold == [50.0, 50.0, 50.0]
    else:
        assert pipeline.enable_acceleration_lora
        assert pipeline.acceleration_lora_path != pipeline.lora_path
        assert c.train.validation.samples == 6


def test_public_fields_have_identical_order():
    def shape(value):
        return [(k, shape(v) if isinstance(v, dict) and not k.endswith("kwargs") else None) for k, v in value.items()]

    assert shape(load_config().to_dict()) == shape(load_config("videocof").to_dict())


@pytest.mark.parametrize(
    "path,value",
    [
        (("train", "learning_rate"), "2e-5"),
        (("train", "max_steps"), True),
        (("train", "loss_type"), "vcos"),
        (("eval", "inference", "kwargs", "omega_vid"), "1.25"),
        (("student", "model_kwargs", "component_root"), 42),
        (("runtime", "student_gpus"), [0, 0]),
    ],
)
def test_reject_invalid_values(path, value):
    data = load_config().to_dict()
    cur = data
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(data)


def test_unknown_kwargs_are_not_silently_ignored():
    data = load_config().to_dict()
    data["eval"]["inference"]["kwargs"]["omgea_vid"] = 1.25
    with pytest.raises(ValueError, match="omgea_vid"):
        ExperimentConfig.from_dict(data)


def test_missing_required_lr():
    data = load_config().to_dict()
    del data["train"]["learning_rate"]
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig.from_dict(data)


def test_round_trip_and_explicit_checkpoint(tmp_path):
    c = load_config()
    c.eval.checkpoint_path = "/checkpoints/custom.safetensors"
    file = tmp_path / "config.yaml"
    c.to_yaml(file)
    restored = ExperimentConfig.from_yaml(file)
    assert restored == c
    assert str(restored.checkpoint_path) == "/checkpoints/custom.safetensors"


def test_unset_keys_take_the_defaults_and_resolve_completely():
    raw = load_raw("bernini")
    c = ExperimentConfig.from_dict(raw)
    assert "eval_gpus" in raw["runtime"]
    del raw["runtime"]["eval_gpus"], raw["run"]["cache_dir"]
    c = ExperimentConfig.from_dict(raw)
    assert c.runtime.eval_gpus == c.runtime.student_gpus
    assert c.run.cache_dir == f"cache/{c.run.name}"
    assert c.eval.inference.kwargs["omega_vid"] == 1.25
    assert c.eval.filter_types and c.eval.scoring.model == "gemini-2.5-pro"
    # The resolved form holds every value, so it reloads to the same configuration.
    assert ExperimentConfig.from_dict(c.to_dict()) == c


def test_older_mode_and_objective_names_are_accepted():
    raw = load_raw("bernini")
    raw["run"]["mode"] = "train_then_openve"
    raw["train"]["objective"] = "opd"
    c = ExperimentConfig.from_dict(raw)
    assert (c.run.mode, c.train.objective) == ("train_eval", "online")


@pytest.mark.parametrize("backend", ["bernini", "videocof"])
def test_skip_checkpoint_evaluates_base_weights(backend):
    c = load_config(backend)
    c.run.mode = "eval"
    c.eval.skip_checkpoint = True
    c.validate()
    assert c.checkpoint_path is None
    pipeline = importlib.import_module(f"i2vt.pipelines.{backend}.configuration_{backend}").pipeline_config(c)
    # Both pipelines read an empty path as "no I2VT LoRA"; "None" would be a file.
    assert pipeline.lora_path == ""
    if backend == "videocof":
        # The VideoCoF editing LoRA is part of the base editor, so a base-weights evaluation still loads it.
        assert importlib.import_module("i2vt.models.videocof.configuration_videocof").model_config(c).lora_path


def test_skip_checkpoint_rejected_when_training_writes_one():
    data = load_config().to_dict()
    data["eval"]["skip_checkpoint"] = True
    data["run"]["mode"] = "train_eval"
    with pytest.raises(ValueError, match="skip_checkpoint"):
        ExperimentConfig.from_dict(data)


def test_import_and_dry_run_from_outside_checkout_without_torch(tmp_path):
    code = "from i2vt import ExperimentConfig, TrainingConfig; import sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    env = {**os.environ, **ROOT_ENV, "LR": "99", "LOSS_TYPE": "wrong"}
    result = subprocess.run(
        [sys.executable, "-m", "i2vt.run", str(ROOT / "tests/fixtures/bernini.yaml"), "--dry-run"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "learning_rate: 2.0e-05" in result.stdout and "loss_type: v_cos" in result.stdout
    assert not (tmp_path / "resolved_config.yaml").exists()


@pytest.mark.parametrize("path", RELEASED, ids=lambda p: p.stem)
def test_released_configs_dry_run_without_torch(path, tmp_path):
    env = {**os.environ, **ROOT_ENV}
    code = (
        "import sys; from i2vt.run import main; main([sys.argv[1], '--dry-run']); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], cwd=tmp_path, env=env, check=True, capture_output=True, text=True
    )
    assert f"name: {path.stem}" in result.stdout


def test_unset_root_stops_before_starting_processes(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ROOT_ENV}
    result = subprocess.run(
        [sys.executable, "-m", "i2vt.run", str(ROOT / "tests/fixtures/bernini.yaml"), "--dry-run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Unresolved environment" in result.stderr


def test_released_configs_set_only_non_default_keys(monkeypatch, tmp_path):
    """Every key a released YAML sets differs from the default it would otherwise get."""

    def leaves(data, prefix=()):
        for key, value in data.items():
            if isinstance(value, dict) and not key.endswith(("kwargs", "by_type", "by_task")):
                yield from leaves(value, prefix + (key,))
            else:
                yield prefix + (key,), value

    required = {("run", "name"), ("runtime", "student_gpus"), ("student", "backend"), ("student", "model_path"),
                ("train", "learning_rate"), ("train", "max_steps"), ("train", "gradient_accumulation_steps"),
                ("teacher", "model_path")}
    for path in RELEASED:
        raw = yaml.safe_load(path.read_text())
        for key, value in leaves(raw):
            if key in required or key[-1].endswith(("kwargs", "by_type", "by_task")):
                continue
            stripped = yaml.safe_load(path.read_text())
            cur = stripped
            for part in key[:-1]:
                cur = cur[part]
            del cur[key[-1]]
            tmp = tmp_path / path.name
            tmp.write_text(yaml.safe_dump(stripped))
            default = load_released(tmp, monkeypatch)
            resolved = load_released(path, monkeypatch).to_dict()
            for part in key:
                resolved = resolved[part]
            got = default.to_dict()
            for part in key:
                got = got[part]
            assert got != resolved, f"{path.name}: {'.'.join(key)} repeats its default {value!r}"


def test_videocof_evaluation_uses_the_official_videocof_settings(monkeypatch):
    """I2VT-CoF is evaluated as VideoCoF is: 33 frames, one reasoning frame, 4 FusionX steps, no resizing."""
    config = load_released(ROOT / "configs/train/i2vt-cof.yaml", monkeypatch)
    inference = config.eval.inference
    assert (inference.num_frames, inference.source_frames) == (33, 33)
    assert inference.kwargs["reasoning_frames"] == 1
    assert inference.kwargs["max_video_side"] == 0
    assert inference.num_inference_steps == 4
    assert inference.kwargs["enable_acceleration_lora"]


@pytest.mark.parametrize("name", ["i2vt-1.3b", "i2vt-14b"])
def test_bernini_models_are_evaluated_with_40_steps(name, monkeypatch):
    config = load_released(ROOT / f"configs/train/{name}.yaml", monkeypatch)
    assert config.eval.inference.num_inference_steps == 40
    assert config.inference_path == Path("outputs/inference") / name / "openve"


def test_benchmarks_have_their_judges(monkeypatch):
    from i2vt.data.registry import BENCHMARKS, GEMINI_FLASH, GEMINI_PRO

    assert BENCHMARKS["openve"].judge == GEMINI_PRO
    assert BENCHMARKS["openve_pp"].judge == GEMINI_FLASH
    assert set(BENCHMARKS) == {"openve", "openve_pp"}
    config = load_released(ROOT / "configs/train/i2vt-1.3b.yaml", monkeypatch)
    assert config.eval.scoring.model == GEMINI_PRO
    from i2vt.data import registry

    registry.apply_cli_choices(config, None, "openve_pp")
    assert config.eval.scoring.model == GEMINI_FLASH
    assert config.eval.filter_types == ["action_edit", "expression_edit", "motion_edit", "relight"]
    assert config.inference_path == Path("outputs/inference/i2vt-1.3b/openve_pp")


def test_cotraining_pools_resolve_to_one_directory(monkeypatch, tmp_path):
    """--dataset action+body_movement trains two pools together as parquet files in one directory.

    Given a directory, ImageEditDataset reads every *.parquet in it, so training on several pools
    needs no merged parquet and the combination shows in the run name.
    """
    import huggingface_hub

    from i2vt.data import registry

    source = tmp_path / "snapshot" / "data" / "train-00000-of-00001.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")

    def fake_download(repo, repo_type, allow_patterns, revision):
        return str(tmp_path / "snapshot")

    def fake_pool(file, name, entry):
        pool = tmp_path / "pools" / f"{name}.parquet"
        pool.parent.mkdir(exist_ok=True)
        pool.write_bytes(b"x")
        return str(pool)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(registry, "_pool_file", fake_pool)
    resolved = Path(registry.resolve_train_dataset("hf:action+motion"))
    assert resolved.is_dir()
    links = sorted(p.name for p in resolved.glob("*.parquet"))
    assert links == ["00_action.parquet", "01_body_movement.parquet"]
    assert {p.resolve().name for p in resolved.glob("*.parquet")} == {"action.parquet", "body_movement.parquet"}

    # A single name returns the file itself, with no directory around it.
    assert Path(registry.resolve_train_dataset("hf:action")).is_file()
    # The full set is the shard directory itself.
    assert Path(registry.resolve_train_dataset("hf:i2vt_17k")) == source.parent


def test_a_pool_keeps_only_its_rows_of_the_full_set(tmp_path):
    """Each pool is a filter on I2VT-17K, written once beside the download and reused afterwards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from i2vt.data import registry

    source = tmp_path / "snapshot" / "data"
    source.mkdir(parents=True)
    table = pa.table({
        "key": ["a", "b", "c", "d"],
        "edit_type": ["global_style", "action_edit", "relight", "action_edit"],
        "pool": ["openve", "action", "relight_wide", "action"],
    })
    # Two shards, so a pool collects its rows across shards in order.
    pq.write_table(table.slice(0, 2), source / "train-00000-of-00002.parquet", row_group_size=1)
    pq.write_table(table.slice(2), source / "train-00001-of-00002.parquet", row_group_size=1)

    action = registry._pool_file(str(source), "action", registry.TRAIN_DATASETS["action"])
    assert pq.read_table(action).column("key").to_pylist() == ["b", "d"]
    assert action == str(tmp_path / "snapshot" / "pools" / "action.parquet")
    relight = registry._pool_file(str(source), "relight_wide", registry.TRAIN_DATASETS["relight_wide"])
    assert pq.read_table(relight).column("key").to_pylist() == ["c"]
    assert registry._pool_file(str(source), "action", registry.TRAIN_DATASETS["action"]) == action


def test_an_unknown_pool_in_a_combination_is_refused():
    from i2vt.data import registry

    with pytest.raises(ValueError, match="unknown training set"):
        registry.apply_cli_choices(SimpleNamespace(), "action+nope", None)


def test_teacher_cfg_can_be_set_per_edit_type():
    """Listed edit types use their own teacher CFG; the rest use teacher.guidance_scale."""
    from i2vt.models.joyedit.modeling_joyedit import JoyEditTeacherBackend

    backend = JoyEditTeacherBackend.__new__(JoyEditTeacherBackend)
    backend.config = SimpleNamespace(guidance_scale=2.0, guidance_scale_by_type={"local_add": 1.0},
                                     guidance_scale_by_task=None)
    assert backend._scale_for({"edit_type": "local_add"}) == 1.0
    assert backend._scale_for({"edit_type": "global_style"}) == 2.0
    assert backend._scale_for({"edit_type": ["local_add"]}) == 1.0
    assert backend._scale_for({}) == 2.0
    backend.config = SimpleNamespace(guidance_scale=2.0, guidance_scale_by_type=None, guidance_scale_by_task=None)
    assert backend._scale_for({"edit_type": "local_add"}) == 2.0


def test_per_type_teacher_cfg_rejects_unknown_edit_types():
    config = load_config("videocof")
    teacher = dataclasses.replace(config.teacher, guidance_scale_by_type={"nope": 1.0})
    with pytest.raises(ValueError, match="unknown edit types"):
        dataclasses.replace(config, teacher=teacher).validate()


def test_a_different_training_set_gets_its_own_run():
    """--dataset changes the training set, so the run name and cache must differ from the default run."""
    from i2vt.data import registry

    config = load_config("videocof")
    config.data.train_dataset = "hf:openve"
    name, cache = config.run.name, config.run.cache_dir
    registry.apply_cli_choices(config, "action+motion", None)
    assert config.run.name == f"{name}_action_body_movement"
    assert config.run.cache_dir == f"{cache}_action_body_movement"
    assert config.data.train_dataset == "hf:action+motion"


def test_loss_type_defaults_to_v_mse():
    """v-MSE is the default loss of the config and of the loss function."""
    import inspect

    from i2vt.config import TrainingConfig
    from i2vt.loss.distillation import distillation_loss

    assert TrainingConfig.__dataclass_fields__["loss_type"].default == "v"
    assert inspect.signature(distillation_loss).parameters["loss_type"].default == "v"
    raw = load_config("bernini").to_dict()
    del raw["train"]["loss_type"]
    assert ExperimentConfig.from_dict(raw).train.loss_type == "v"
