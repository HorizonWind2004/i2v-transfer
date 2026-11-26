import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from i2vt import run
from tests.helpers import load_config


@pytest.mark.parametrize(
    "mode,expected", [("train", ["train"]), ("eval", ["eval"]), ("train_eval", ["train", "eval"])]
)
def test_modes_share_exact_resolved_configuration(tmp_path, monkeypatch, mode, expected):
    c = load_config()
    c.run.output_dir = str(tmp_path)
    c.run.mode = mode
    c.eval.checkpoint_path = str(tmp_path / "final.safetensors")
    Path(c.eval.checkpoint_path).touch()
    monkeypatch.setattr(run, "preflight", lambda c: None)
    stages = []

    def stage(name):
        def call(config, path):
            assert config.from_yaml(path) == config
            stages.append(name)

        return call

    monkeypatch.setattr(run, "run_training", stage("train"))
    monkeypatch.setattr(run, "run_evaluation", stage("eval"))
    run.run(c)
    assert stages == expected


def test_training_failure_never_scores_existing_checkpoint(tmp_path, monkeypatch):
    c = load_config()
    c.run.mode = "train_eval"
    c.run.output_dir = str(tmp_path)
    c.eval.checkpoint_path = str(tmp_path / "old.safetensors")
    Path(c.eval.checkpoint_path).touch()
    monkeypatch.setattr(run, "preflight", lambda c: None)

    def fail(*_):
        raise RuntimeError("training failed")

    monkeypatch.setattr(run, "run_training", fail)
    evaluate = Mock()
    monkeypatch.setattr(run, "run_evaluation", evaluate)
    with pytest.raises(RuntimeError):
        run.run(c)
    evaluate.assert_not_called()


def test_process_failure_cleans_other_workers(tmp_path):
    with pytest.raises(RuntimeError):
        with run.ProcessGroup() as group:
            waiting = group.start([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "waiting.log")
            failing = group.start([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failing.log")
            group.wait([waiting, failing])
    assert waiting.poll() is not None and failing.poll() == 7
    assert all(log.closed for log in group.logs)


def test_changed_config_cannot_reuse_run_directory(tmp_path, monkeypatch):
    c = load_config()
    c.run.output_dir = str(tmp_path)
    c.output_path.mkdir()
    previous = copy.deepcopy(c)
    previous.train.learning_rate = 0.1
    previous.to_yaml(c.output_path / "resolved_config.yaml")
    monkeypatch.setattr(run, "preflight", lambda c: None)
    with pytest.raises(ValueError, match="different resolved"):
        run.run(c)


@pytest.mark.parametrize("worker,index", [("train", None), ("infer", 2), ("score", None)])
def test_single_cli_dispatches_resolved_config_to_workers(tmp_path, monkeypatch, worker, index):
    config = load_config()
    path = tmp_path / "resolved.yaml"
    config.to_yaml(path)
    handler = Mock()
    if worker == "train":
        monkeypatch.setattr(run, "train_worker", handler)
    else:
        from i2vt.evaluation import infer, score

        monkeypatch.setattr(
            infer if worker == "infer" else score, "infer" if worker == "infer" else "score_config", handler
        )
    argv = [str(path), "--worker", worker]
    if index is not None:
        argv += ["--index", str(index)]
    run.main(argv)
    handler.assert_called_once_with(config, *([] if index is None else [index]))


def test_cli_rejects_missing_worker_index_before_loading_models(tmp_path):
    path = tmp_path / "resolved.yaml"
    load_config().to_yaml(path)
    with pytest.raises(SystemExit) as exc:
        run.main([str(path), "--worker", "infer"])
    assert exc.value.code == 2


def test_deterministic_seed_covers_python_numpy_and_torch():
    import random

    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    from unittest.mock import patch

    try:
        with torch.random.fork_rng(devices=[]), patch.dict("os.environ"):
            run.seed_everything(1234, True)
            first = (random.random(), np.random.rand(), torch.rand(4))
            run.seed_everything(1234, True)
            second = (random.random(), np.random.rand(), torch.rand(4))
            assert first[:2] == second[:2]
            assert torch.equal(first[2], second[2])
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_deterministic, benchmark


def test_steps_override_gets_its_own_inference_directory():
    c = load_config()
    default = c.eval.inference.num_inference_steps
    path = c.inference_path
    run.apply_inference_steps(c, default)
    assert c.inference_path == path and path.name == "openve"
    run.apply_inference_steps(c, 20)
    assert c.eval.inference.num_inference_steps == 20
    assert c.inference_path == path.with_name("openve_20steps")
    with pytest.raises(SystemExit):
        run.apply_inference_steps(c, 0)


def test_cli_options_reach_the_resolved_configuration(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.yaml"
    load_config().to_yaml(path)
    checkpoint = tmp_path / "lora.safetensors"
    run.main([str(path), "--dry-run", "--mode", "eval", "--bench", "openve_pp", "--checkpoint", str(checkpoint),
              "--judge", "gemini-test", "--steps", "20"])
    out = capsys.readouterr().out
    assert "mode: eval" in out and "bench: openve_pp" in out and f"checkpoint_path: {checkpoint}" in out
    assert "model: gemini-test" in out and "num_inference_steps: 20" in out
    assert out.count("- relight") == 1
    with pytest.raises(SystemExit):
        run.main([str(path), "--dry-run", "--mode", "nope"])


def test_ranks_are_reseeded_after_the_teacher_is_built(monkeypatch):
    """JoyEdit's builder seeds every rank alike, so the worker reseeds per rank before training."""
    import torch
    import torch.distributed as dist

    import i2vt.models
    import i2vt.trainer

    calls = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "new_group", lambda **k: "group")
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(run, "seed_everything", lambda seed, deterministic: calls.append(("seed", seed)))
    monkeypatch.setattr(i2vt.models, "create_teacher", lambda config, device: calls.append(("teacher",)) or "teacher")

    class Trainer:
        def __init__(self, config, *, teacher):
            calls.append(("trainer", teacher))

        def train(self, rank, world, device, group):
            calls.append(("train", rank, world, group))

    monkeypatch.setattr(i2vt.trainer, "Trainer", Trainer)
    config = load_config()
    run.train_worker(config)
    seed = ("seed", config.train.seed + 3)
    assert calls == [seed, ("teacher",), seed, ("trainer", "teacher"), ("train", 3, 4, "group")]
