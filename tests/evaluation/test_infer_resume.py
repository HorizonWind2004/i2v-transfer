import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from i2vt.evaluation import infer


def output():
    return SimpleNamespace(
        frames=None,
        fps=16,
        metadata={"backend": "videocof", "prompt": "edit"},
        reasoning_frames=None,
        source_frames=None,
    )


def test_completed_case_is_skipped_but_interrupted_video_is_regenerated(tmp_path, monkeypatch):
    from i2vt import pipelines

    rows = [dict(edited_type="style", base=name, video_path=name, prompt="edit") for name in ("done", "partial")]
    for row in rows:
        case = tmp_path / "style" / row["base"]
        case.mkdir(parents=True)
        (case / f"gen_{row['base']}.mp4").write_bytes(b"video")
    (tmp_path / "style/done/gen_done_info.txt").write_text(json.dumps(output().metadata))
    pipeline = Mock(return_value=output())
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(pipelines, "create_pipeline", factory)
    monkeypatch.setattr(infer, "work_rows", lambda _: rows)
    monkeypatch.setattr(infer, "_write_video", lambda _, path, *args: path.write_bytes(b"complete video"))
    config = SimpleNamespace(
        inference_path=tmp_path,
        runtime=SimpleNamespace(eval_gpus=[0]),
        eval=SimpleNamespace(skip_existing=True),
    )
    infer.infer(config, 0)
    pipeline.assert_called_once_with(video="partial", prompt="edit")
    factory.assert_called_once_with(config)
    assert infer.completed_output(tmp_path / "style/partial", "partial")
    infer.infer(config, 0)
    factory.assert_called_once()  # Entirely completed runs need no model or CUDA allocation.


def test_failed_rewrite_invalidates_previous_completion(tmp_path, monkeypatch):
    info = tmp_path / "gen_case_info.txt"
    info.write_text(json.dumps(output().metadata))
    (tmp_path / "gen_case.mp4").write_bytes(b"old video")

    def interrupted(*args):
        assert not info.exists()
        raise RuntimeError("encoder interrupted")

    monkeypatch.setattr(infer, "_write_video", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        infer.save_pipeline_output(output(), tmp_path, "case")
    assert not infer.completed_output(tmp_path, "case")


@pytest.mark.parametrize("metadata,video", [("{", b"video"), ("null", b"video"), ("{}", b"")])
def test_incomplete_outputs_are_not_skipped(tmp_path, metadata, video):
    (tmp_path / "gen_case_info.txt").write_text(metadata)
    (tmp_path / "gen_case.mp4").write_bytes(video)
    assert not infer.completed_output(tmp_path, "case")


def test_auxiliary_video_failure_does_not_mark_main_video_complete(tmp_path, monkeypatch):
    value = output()
    value.reasoning_frames = object()
    calls = []

    def write(frames, path, *args):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("reasoning video failed")
        path.write_bytes(b"main saved")

    monkeypatch.setattr(infer, "_write_video", write)
    with pytest.raises(RuntimeError, match="reasoning video"):
        infer.save_pipeline_output(value, tmp_path, "case")
    assert (tmp_path / "gen_case.mp4").exists()
    assert not infer.completed_output(tmp_path, "case")
