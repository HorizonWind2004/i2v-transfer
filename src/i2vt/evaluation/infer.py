"""Run benchmark inference and save videos and metadata."""

import csv
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np

from i2vt.data.registry import case_video
from i2vt.pipelines.pipeline_utils import normalize_to_01


def _write_video(tensor, path, fps, backend):
    """Encode a (1, C, T, H, W) tensor as H.264 with the same settings for every backend.

    This is the file the judge scores, so it is encoded near-losslessly (crf 8) rather than with
    imageio's defaults, and pixels are rounded rather than truncated.
    """
    video = tensor.detach().float().cpu()
    if backend == "bernini":
        video = normalize_to_01(video)
    # VideoCoF output is already in [0, 1] (matches save_videos_grid for a batch of one).
    frames = video[0].permute(1, 2, 3, 0).numpy()
    imageio.mimwrite(
        str(path),
        (frames * 255).round().astype(np.uint8),
        fps=fps,
        codec="libx264",
        quality=10,
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-crf", "8"],
    )


def save_pipeline_output(output, directory, stem):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = directory / f"gen_{stem}_info.txt"
    # Metadata is the completion marker: invalidate it before replacing videos.
    info.unlink(missing_ok=True)
    _write_video(output.frames, directory / f"gen_{stem}.mp4", output.fps, output.metadata["backend"])
    if output.reasoning_frames is not None:
        _write_video(
            output.reasoning_frames, directory / f"gen_{stem}_reason_edit.mp4", output.fps, output.metadata["backend"]
        )
    if output.source_frames is not None:
        a, b = normalize_to_01(output.source_frames), normalize_to_01(output.frames)
        t, h, w = [min(a.shape[i], b.shape[i]) for i in (2, 3, 4)]
        _write_video(
            torch.cat([a[:, :, :t, :h, :w], b[:, :, :t, :h, :w]], dim=4),
            directory / f"gen_{stem}_compare.mp4",
            output.fps,
            "bernini",
        )
    temporary = info.with_name(f"{info.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(output.metadata, indent=2), encoding="utf-8")
        temporary.replace(info)
    finally:
        temporary.unlink(missing_ok=True)


def completed_output(directory, stem):
    """Only skip cases whose videos and final metadata were saved successfully."""
    directory = Path(directory)
    video = directory / f"gen_{stem}.mp4"
    info = directory / f"gen_{stem}_info.txt"
    try:
        return video.stat().st_size > 0 and isinstance(json.loads(info.read_text(encoding="utf-8")), dict)
    except (OSError, ValueError):
        return False


def work_rows(config):
    with open(config.eval.bench_csv, encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row["edited_type"] not in config.eval.filter_types:
                continue
            raw = case_video(row)
            path = Path(raw.replace("OpenVE-Bench/", config.eval.bench_dir.rstrip("/") + "/"))
            if not path.is_absolute() and not raw.startswith("OpenVE-Bench/"):
                path = Path(config.eval.bench_dir) / raw
            rows.append(
                {
                    "edited_type": row["edited_type"],
                    "video_path": str(path),
                    "prompt": row["prompt"],
                    "base": path.stem,
                }
            )
    return rows if config.eval.limit is None else rows[: config.eval.limit]


def _peak_memory():
    import torch

    if not torch.cuda.is_available():
        return ""
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return f" peak={peak:.1f}GiB"


def infer(config, rank):
    from i2vt.pipelines import create_pipeline

    rows = work_rows(config)[rank :: len(config.runtime.eval_gpus)]
    pipeline = None
    for row in rows:
        case = config.inference_path / row["edited_type"] / row["base"]
        if config.eval.skip_existing and completed_output(case, row["base"]):
            print(f"Skipping completed output: {case}", flush=True)
            continue
        if pipeline is None:
            pipeline = create_pipeline(config)
        started = time.perf_counter()
        output = pipeline(video=row["video_path"], prompt=row["prompt"])
        save_pipeline_output(output, case, row["base"])
        print(
            f"[infer] rank={rank} {row['edited_type']}/{row['base']} "
            f"{time.perf_counter() - started:.1f}s{_peak_memory()}",
            flush=True,
        )
