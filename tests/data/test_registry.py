"""Named training sets and benchmarks resolve to the Hugging Face Hub."""

from types import SimpleNamespace

import pytest


def test_named_training_set_and_benchmark_resolve_from_the_hub(monkeypatch, tmp_path):
    import huggingface_hub

    from i2vt.data import registry

    calls = []

    def fake_snapshot(repo, repo_type, allow_patterns, revision=None):
        if repo == registry.I2VT_17K:
            calls.append((repo, allow_patterns, revision))
            return str(tmp_path / "train")
        assert allow_patterns is None
        bench = tmp_path / "snap"
        bench.mkdir(parents=True, exist_ok=True)
        (bench / "metadata.csv").write_text("file_name,edited_type,prompt\n")
        return str(tmp_path / "snap")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(registry, "_pool_file", lambda file, name, entry: f"{file}:{name}")
    config = SimpleNamespace(
        run=SimpleNamespace(name="r", cache_dir="/c/r", mode="train"),
        data=SimpleNamespace(train_dataset="hf:openve"),
        eval=SimpleNamespace(bench="openve", bench_dir=None, bench_csv=None, filter_types=["global_style"],
                             scoring=SimpleNamespace(model="gemini-2.5-pro")),
    )
    registry.apply_cli_choices(config, "body_movement@abc123", "openve_pp")
    assert config.run.name == "r_body_movement" and config.run.cache_dir == "/c/r_body_movement"
    assert config.eval.filter_types == ["action_edit", "expression_edit", "motion_edit", "relight"]
    assert config.eval.scoring.model == registry.GEMINI_FLASH
    registry.resolve_hf_sources(config)
    assert calls == [("sanaka87/I2VT-17K", ["data/train-*.parquet"], "abc123")]
    assert config.data.train_dataset == f"{tmp_path / 'train/data'}:body_movement"
    assert config.eval.bench_csv == str(tmp_path / "snap/metadata.csv")


def test_unknown_names_are_refused():
    from i2vt.data import registry

    config = SimpleNamespace(run=SimpleNamespace(name="r", cache_dir="c"),
                             data=SimpleNamespace(train_dataset="x"), eval=SimpleNamespace(bench="openve"))
    with pytest.raises(ValueError):
        registry.apply_cli_choices(config, "walking", None)
    with pytest.raises(ValueError):
        registry.apply_cli_choices(config, None, "walking")


def test_relight_wide_is_registered_and_combines_with_relight():
    from i2vt.data import registry

    entry = registry.TRAIN_DATASETS["relight_wide"]
    assert entry.filtered and entry.shards == registry.I2VT_17K_SHARDS
    assert entry.repo == registry.TRAIN_DATASETS["relight"].repo
    assert not registry.TRAIN_DATASETS["i2vt_17k"].filtered
    # The combined form parses into two pools, each a different subset of the same file.
    assert registry._names("hf:relight+relight_wide") == ["hf:relight", "hf:relight_wide"]
    assert registry._split("hf:relight_v2", registry.TRAIN_DATASETS, "training set")[0] is entry


def test_the_pools_add_up_to_i2vt_17k():
    from i2vt.data import registry

    counts = [int(registry.TRAIN_DATASETS[name].note.split(", ")[-1].split()[0])
              for name in ("openve", "action", "body_movement", "relight", "relight_wide")]
    assert sum(counts) == 17227


def test_openve_pp_benchmark_covers_all_four_groups():
    from i2vt.data import registry

    entry = registry.BENCHMARKS["openve_pp"]
    assert entry.repo == registry.OPENVE_PP and entry.subdir == ""
    assert entry.filter_types == ("action_edit", "expression_edit", "motion_edit", "relight")
    assert "relight" not in registry.BENCHMARKS
