"""The installable source tree, vendored licenses and released configs."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_source_layout_and_vendor_attribution():
    assert (ROOT / "src/i2vt/__init__.py").is_file()
    assert not (ROOT / "i2vt").exists()
    for vendor in ("bernini", "joyedit", "wan"):
        assert (ROOT / "src/i2vt/vendor" / vendor / "LICENSE").is_file()
        assert (ROOT / "src/i2vt/vendor" / vendor / "SOURCE").is_file()


def test_vendor_licenses_ship_with_the_package():
    text = (ROOT / "pyproject.toml").read_text()
    for vendor in ("bernini", "joyedit", "wan"):
        line = next(line for line in text.splitlines() if line.startswith(f'"i2vt.vendor.{vendor}"'))
        assert '"LICENSE"' in line and '"SOURCE"' in line, line


def test_library_does_not_import_examples_or_scripts():
    for path in (ROOT / "src/i2vt").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("scripts.", "examples.")), path
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith(("scripts.", "examples.")) for alias in node.names), path


def test_the_released_configs_are_the_only_committed_configs():
    tracked = subprocess.run(["git", "ls-files", "configs"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    names = sorted(p.removeprefix("configs/") for p in tracked.split() if p.endswith(".yaml"))
    assert names == ["train/i2vt-1.3b.yaml", "train/i2vt-14b.yaml", "train/i2vt-cof.yaml"]


def test_every_released_config_trains_with_v_mse():
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        assert yaml.safe_load(path.read_text())["train"].get("loss_type", "v") == "v", path


def test_summary_averages_every_run_over_the_same_cases(tmp_path):
    def write(directory, rows):
        directory.mkdir(parents=True)
        (directory / "openve_judge_gemini_video_action_edit_score.jsonl").write_text(
            "".join(json.dumps({"base": b, "edited_type": "action_edit", "scores": s}) + "\n" for b, s in rows)
        )

    write(tmp_path / "a", [("c1", [5, 5, 5]), ("c2", [1, 1, 1]), ("c2", [3, 3, 3])])
    write(tmp_path / "b", [("c1", [2, 2, 2])])
    out = subprocess.run(
        [sys.executable, "scripts/summarize_scores.py", "--run", f"A={tmp_path / 'a'}", "--run", f"B={tmp_path / 'b'}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    line_a = next(row for row in out.splitlines() if row.startswith("A "))
    assert " 1 " in line_a and "5.00" in line_a  # only c1, shared with B
