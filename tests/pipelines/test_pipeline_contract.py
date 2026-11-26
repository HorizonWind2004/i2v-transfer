from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import BatchEncoding

from i2vt.evaluation.infer import work_rows
from i2vt.pipelines.bernini import pipeline_bernini as bernini
from i2vt.pipelines.videocof import pipeline_videocof as videocof
from tests.helpers import load_config


def test_videocof_pipeline_reuse_and_output_do_not_write_files(tmp_path, monkeypatch):
    c = load_config("videocof")
    pipe = videocof.VideoCoFVideoEditPipeline.__new__(videocof.VideoCoFVideoEditPipeline)
    pipe.config = videocof.pipeline_config(c, "cpu")
    pipe.device = torch.device("cpu")
    pipe.pipeline = Mock(return_value=SimpleNamespace(videos=torch.zeros(1, 3, 81, 4, 4)))
    monkeypatch.setattr(videocof, "load_video_frames", lambda *a: (torch.zeros(1, 3, 81, 4, 4), 4, 4))
    for _ in range(2):
        out = pipe(video=str(tmp_path / "source.mp4"), prompt="make red")
        assert out.frames.shape == (1, 3, 81, 4, 4)
        assert out.fps == 16 and out.metadata["backend"] == "videocof"
    assert pipe.pipeline.call_count == 2
    assert pipe.pipeline.call_args.kwargs["num_inference_steps"] == 4
    assert list(tmp_path.iterdir()) == []


def test_bernini_pipeline_reuse_restores_model_device_and_returns_frames(tmp_path, monkeypatch):
    c = load_config()
    pipe = bernini.BerniniVideoEditPipeline.__new__(bernini.BerniniVideoEditPipeline)
    pipe.config = bernini.pipeline_config(c, "cpu")
    pipe.device = torch.device("cpu")
    tokens = BatchEncoding({"input_ids": torch.ones(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2)})
    pipe.tokenizer = Mock(return_value=tokens)
    latent = torch.zeros(1, 16, 2, 2, 2)
    pipe.vae = Mock(dtype=torch.float32)
    pipe.vae.encode.return_value = SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))
    pipe.vae.decode.return_value = (torch.zeros(1, 3, 5, 16, 16),)
    pipe.vae_needs_external_normalization = False
    pipe.model = Mock()
    pipe.model.sample.return_value = latent
    pipe.dit_root = Path("model")
    pipe.base_components_root = Path("components")
    monkeypatch.setattr(bernini, "load_bernini_video", lambda *a: torch.zeros(1, 3, 5, 16, 16))
    for _ in range(2):
        out = pipe(video="source.mp4", prompt="make red")
        assert out.frames.shape == (1, 3, 5, 16, 16)
        assert out.metadata["generation"]["sampled_source_frames"] == 5
    assert pipe.model.sample.call_count == 2
    assert pipe.model.sample.call_args.kwargs["num_frames"] == 5
    assert pipe.model.to.call_count == 4
    assert list(tmp_path.iterdir()) == []


def test_openve_worklist_uses_actual_benchmark_columns(tmp_path):
    c = load_config()
    c.eval.bench_dir = str(tmp_path)
    c.eval.bench_csv = str(tmp_path / "benchmark.csv")
    Path(c.eval.bench_csv).write_text(
        "edited_type,original_video,prompt\nglobal_style,OpenVE-Bench/videos/a.mp4,paint\ncamera_edit,OpenVE-Bench/videos/b.mp4,pan\n"
    )
    rows = work_rows(c)
    assert rows == [
        {"edited_type": "global_style", "video_path": str(tmp_path / "videos/a.mp4"), "prompt": "paint", "base": "a"}
    ]


def test_videocof_pipeline_preserves_source_length_and_sampling_arguments(monkeypatch):
    c = load_config("videocof")
    pipe = videocof.VideoCoFVideoEditPipeline.__new__(videocof.VideoCoFVideoEditPipeline)
    pipe.config = videocof.pipeline_config(c, "cpu")
    pipe.config.source_frames = pipe.config.num_frames = 9
    pipe.config.fps = 24
    pipe.config.shift = 3.0
    pipe.device = torch.device("cpu")
    edit = torch.ones(1, 3, 9, 4, 4)
    full = torch.cat([torch.zeros(1, 3, 1, 4, 4), edit], dim=2)
    pipe.pipeline = Mock(return_value=SimpleNamespace(videos=full))
    source = torch.zeros_like(edit)
    monkeypatch.setattr(videocof, "load_video_frames", lambda *args: (source, 4, 4))
    output = pipe("source.mp4", "paint")
    assert torch.equal(output.frames, edit)
    assert torch.equal(output.reasoning_frames, full)
    assert output.fps == 24
    args = pipe.pipeline.call_args.kwargs
    assert args["video"] is source
    assert args["source_frames"] == args["num_frames"] == 9
    assert args["cot"] and args["repeat_rope"]
    assert args["num_inference_steps"] == 4 and args["shift"] == 3.0


def test_openve_saves_generation_reasoning_and_comparison(tmp_path, monkeypatch):
    import json

    from i2vt.evaluation import infer

    writes = []
    monkeypatch.setattr(
        infer.imageio, "mimsave", lambda path, frames, **kw: writes.append((Path(path).name, frames.shape, kw))
    )
    monkeypatch.setattr(
        infer.imageio, "mimwrite", lambda path, frames, **kw: writes.append((Path(path).name, frames.shape, kw))
    )
    output = SimpleNamespace(
        frames=torch.ones(1, 3, 5, 4, 4),
        source_frames=torch.zeros(1, 3, 7, 4, 4),
        reasoning_frames=torch.ones(1, 3, 6, 4, 4),
        fps=24,
        metadata={"backend": "videocof", "prompt": "paint"},
    )
    infer.save_pipeline_output(output, tmp_path, "sample")
    assert [(name, shape) for name, shape, _ in writes] == [
        ("gen_sample.mp4", (5, 4, 4, 3)),
        ("gen_sample_reason_edit.mp4", (6, 4, 4, 3)),
        ("gen_sample_compare.mp4", (5, 4, 8, 3)),
    ]
    assert all(kwargs["fps"] == 24 for _, _, kwargs in writes)
    assert json.loads((tmp_path / "gen_sample_info.txt").read_text()) == output.metadata
