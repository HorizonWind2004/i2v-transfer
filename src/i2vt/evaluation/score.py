"""Score edited videos with a Gemini judge, following the OpenVE-Bench protocol.

Gemini reads the source and the edited video in full together with the rubric of the edit type, and
returns three 1-5 scores: instruction compliance, visual quality and stability, and consistency and
detail fidelity. Scores are appended to one jsonl per category in the inference directory, so an
interrupted run resumes where it stopped.

    python -m i2vt.evaluation.score --video_paths outputs/inference/i2vt-1.3b/openve \\
        --bench-csv <bench>/benchmark_videos.csv --bench-dir <bench> --filter-types global_style
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

from i2vt.data.registry import case_video

from .rubrics import fill_rubric, prompt_type, rubric_source

DEFAULT_MODEL_ID = "gemini-2.5-pro"
PROTOCOL = "gemini_video"
SCORE_DIMS = ("instruction", "visual", "consistency")


def avg_score_by_edited_type(jsonl_path):
    score_sum = defaultdict(float)
    score_count = defaultdict(int)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
            if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
                continue
            edited_type = record.get("edited_type")
            scores = record.get("scores", [])
            if edited_type and _valid_scores(scores):
                for dim, value in zip(SCORE_DIMS, scores):
                    score_sum[(edited_type, dim)] += value
                score_count[edited_type] += 1
    print(dict(score_count))
    return {
        edited_type: {dim: score_sum[(edited_type, dim)] / count for dim in SCORE_DIMS}
        for edited_type, count in score_count.items()
        if count > 0
    }


def load_api_keys(api_key_file=None):
    """Gemini API keys from ``api_key_file`` (one per line), else GEMINI_API_KEY or GOOGLE_API_KEY.

    The environment variable may hold several comma-separated keys; each request picks one.
    """
    keys = []
    if api_key_file:
        keys.extend(
            line.strip()
            for line in Path(api_key_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not keys:
        value = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        keys.extend(x.strip() for x in value.replace("\n", ",").split(",") if x.strip())
    keys = list(dict.fromkeys(keys))
    if not keys:
        raise SystemExit("No Gemini API key found. Set GEMINI_API_KEY or pass --api-key-file.")
    return keys


def _parse_score(line):
    """The number after the colon of a score line, rounded to the nearest integer (4.5 -> 5)."""
    return math.floor(float(line.split(":")[-1].strip().rstrip(".")) + 0.5)


DEFAULT_SCORE_LABELS = ("Instruction Compliance", "Visual Quality & Stability", "Consistency & Detail Fidelity")
_LABEL_LINE = re.compile(r"^([^\n:]+):\s*(?:1-5|A number from 1 to 5)\.\s*$", re.M)


def rubric_labels(rubric):
    """The three score labels a rubric asks for, in order, read from its response-format block.

    The OpenVE-Bench rubrics name their axes per category (e.g. "Prompt Compliance" /
    "Visual & Temporal Naturalness" / "Physical & Detail Coherence" for removal);
    ours use DEFAULT_SCORE_LABELS.
    """
    block = rubric.split("Example Response Format", 1)[-1]
    labels = tuple(label.strip() for label in _LABEL_LINE.findall(block))
    if len(labels) != 3:
        raise ValueError(f"rubric does not declare three score lines: {labels}")
    return labels


# Which SCORE_DIMS slot each judge label fills. Slots go by label, never by line position: the
# OpenVE-Bench rubrics name their axes per category and do not always list visual before
# consistency (global_style asks for Content Preservation, then Temporal Stability).
LABEL_SLOT = {
    "Instruction Compliance": 0,
    "Visual Quality & Stability": 1,
    "Consistency & Detail Fidelity": 2,
    # OpenVE-Bench axes
    "Prompt Compliance": 0,
    "Style Fidelity": 0,
    "Temporal Stability": 1,
    "Content Preservation": 2,
    "Temporal & Visual Coherence": 1,
    "Physical Plausibility & Detail Preservation": 2,
    "Visual Naturalness & Temporal Stability": 1,
    "Physical & Motion Integrity": 2,
    "Visual & Temporal Seamlessness": 1,
    "Physical Consistency": 2,
    "Visual & Temporal Naturalness": 1,
    "Physical & Detail Coherence": 2,
    "Subtitle Attribute Fidelity": 1,
    "Integrity of Unedited Content": 2,
}


def check_format(out, rubric=None):
    """[instruction, visual, consistency] scores from a judge answer, or False.

    Looks only for the three labels the rubric declares (DEFAULT_SCORE_LABELS when no rubric is
    given) and files each under its LABEL_SLOT, so the order is the same whatever the rubric.
    """
    labels = rubric_labels(rubric) if rubric else DEFAULT_SCORE_LABELS
    slots = [LABEL_SLOT.get(label, i) for i, label in enumerate(labels)]
    if sorted(slots) != [0, 1, 2]:
        raise ValueError(f"labels {labels} do not map onto three distinct score slots")
    try:
        found = {}
        for line in out.splitlines():
            line = line.replace("**", "").strip()
            for label, slot in zip(labels, slots):
                if line.startswith(label + ":"):
                    found[slot] = _parse_score(line)
        if len(found) != 3:
            return False
        scores = [found[0], found[1], found[2]]
        if any(score not in range(1, 6) for score in scores):
            print("score is not in range(1, 6)")
            return False
        return scores
    except Exception as e:
        print(e)
        return False


def score_pair(task):
    """Score one (source, edited) pair with Gemini; None when every attempt fails."""
    from .gemini_video import evaluate_video_pair

    original, edited, prompt, metadata, idx, keys, model, timeout = task
    try:
        scores, result = evaluate_video_pair(
            original,
            edited,
            prompt,
            api_key=random.choice(keys),
            model=model,
            timeout=timeout,
            parse_response=partial(check_format, rubric=prompt),
        )
        return scores, result, metadata, str(original), str(edited)
    except Exception as exc:
        print(f"Task {idx} scoring failed: {exc}", flush=True)
        return None


def batch_process_videos(tasks, max_workers):
    if max_workers <= 1:
        for task in tasks:
            yield score_pair(task)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_pair, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def score_path(directory, model, suffix=""):
    return Path(directory) / f"openve_{model}_{PROTOCOL}{suffix}_score.jsonl"


def result_video_for_case(save_dir, edited_type, original_video):
    base = Path(original_video).stem
    case_dir = Path(save_dir) / edited_type / base
    preferred = case_dir / f"gen_{base}.mp4"
    if preferred.exists():
        return preferred, base
    if case_dir.is_dir():
        for path in sorted(case_dir.glob("gen_*.mp4")):
            if not path.name.endswith(("_compare.mp4", "_reason_edit.mp4")):
                return path, base
    return None, base


def build_tasks(args, save_dir, api_keys):
    tasks = []
    allowed = set(args.filter_types.split(",")) if args.filter_types != "all" else None
    done = set() if args.overwrite else valid_score_bases(
        score_path(save_dir, args.model_id, args.output_suffix), check_rubric=True
    )
    with open(args.bench_csv, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            edited_type = row["edited_type"]
            if allowed is not None and edited_type not in allowed:
                continue
            if edited_type not in prompt_type:
                continue
            original_video = case_video(row)
            original_video_path = Path(
                original_video.replace("OpenVE-Bench/", str(Path(args.bench_dir)).rstrip("/") + "/")
            )
            if not original_video_path.is_absolute() and not original_video.startswith("OpenVE-Bench/"):
                original_video_path = Path(args.bench_dir) / original_video_path
            edited_video_path, base = result_video_for_case(save_dir, edited_type, original_video)
            if args.case and base not in set(args.case):
                continue
            if base in done:
                print(f"[skip] {base}")
                continue
            if edited_video_path is None:
                print(f"[missing] {base}")
                continue
            prompt = row["prompt"]
            metadata = {"prompt": prompt, "edited_type": edited_type, "base": base}
            sys_prompt = fill_rubric(prompt_type[edited_type], prompt)
            tasks.append(
                (original_video_path, edited_video_path, sys_prompt, metadata, idx, api_keys, args.model_id, args.timeout)
            )
            if args.limit and len(tasks) >= args.limit:
                break
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score edited videos with a Gemini judge")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Gemini judge")
    parser.add_argument(
        "--video_paths", type=str, nargs="+", required=True, help="inference directories to score"
    )
    parser.add_argument("--bench-csv", required=True, help="case list of the benchmark (benchmark_videos.csv or metadata.csv)")
    parser.add_argument("--bench-dir", required=True, help="benchmark directory holding the source videos")
    parser.add_argument("--filter-types", default="all", help="comma-separated edit types, or all")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-suffix", default="", help="Suffix appended to the score jsonl filename")
    args = parser.parse_args(argv)
    if min(args.timeout, args.max_workers) <= 0:
        parser.error("timeout and max-workers must be positive")

    api_keys = ["dry-run-key"] if args.dry_run else load_api_keys(args.api_key_file)

    for save_dir in args.video_paths:
        out_path = score_path(save_dir, args.model_id, args.output_suffix)
        tasks = build_tasks(args, save_dir, api_keys)
        if args.dry_run:
            print(f"Would evaluate {len(tasks)} tasks for {save_dir}")
            for task in tasks[:10]:
                print(task[3]["base"], "->", task[1])
            continue
        failures = 0
        with open(out_path, "a" if not args.overwrite else "w", encoding="utf-8") as f:
            for item in batch_process_videos(tasks, args.max_workers):
                if not item:
                    failures += 1
                    continue
                scores, result, metadata, original_video, edited_video = item
                record = {
                    "base": metadata["base"],
                    "protocol": PROTOCOL,
                    "model": args.model_id,
                    "rubric": rubric_source(metadata["edited_type"]),
                    "scores": scores,
                    "prompt": metadata["prompt"],
                    "edited_type": metadata["edited_type"],
                    "original_video": original_video,
                    "edited_video": edited_video,
                    "result": result,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        if failures:
            raise RuntimeError(f"Scoring failed for {failures} cases; successful results were saved for resume")
        averages = avg_score_by_edited_type(out_path)
        if averages:
            print(f"{out_path} All scores (" + " / ".join(SCORE_DIMS) + "):")
            for dim in SCORE_DIMS:
                mean = sum(a[dim] for a in averages.values()) / len(averages)
                print(f"  {dim}: {mean:.2f}")
            for edited_type, avg in averages.items():
                print(f" {edited_type}: " + " / ".join(f"{avg[dim]:.2f}" for dim in SCORE_DIMS))


def score_arguments(config):
    scoring = config.eval.scoring
    from .infer import work_rows

    rows = work_rows(config)
    if not rows:
        raise ValueError("No benchmark cases match the evaluation configuration")
    for category in dict.fromkeys(row["edited_type"] for row in rows):
        bases = {row["base"] for row in rows if row["edited_type"] == category}
        command = [
            "--video_paths",
            str(config.inference_path),
            "--bench-csv",
            config.eval.bench_csv,
            "--bench-dir",
            config.eval.bench_dir,
            "--filter-types",
            category,
            "--output-suffix",
            f"_{category}",
            "--model_id",
            scoring.model,
        ]
        if scoring.api_key_file:
            command += ["--api-key-file", scoring.api_key_file]
        for base in sorted(bases):
            command += ["--case", base]
        yield category, bases, command


def _valid_scores(scores):
    return isinstance(scores, list) and len(scores) == 3 and all(type(x) in (int, float) and 1 <= x <= 5 for x in scores)


def valid_score_bases(path, *, check_rubric=False):
    """Bases with a valid score row; with `check_rubric`, only rows scored with the current rubric."""
    bases = set()
    path = Path(path)
    if not path.exists():
        return bases
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if (
            row.get("base")
            and _valid_scores(row.get("scores"))
            and row.get("protocol") == PROTOCOL
            and (not check_rubric or row.get("rubric") == rubric_source(row.get("edited_type")))
        ):
            bases.add(row["base"])
    return bases


def score_config(config):
    """Score every evaluated case of a run with its configured judge; fail if any case is left."""
    for category, expected, command in score_arguments(config):
        main(command)
        path = score_path(config.inference_path, config.eval.scoring.model, f"_{category}")
        missing = expected - valid_score_bases(path, check_rubric=True)
        if missing:
            raise RuntimeError(f"Incomplete scores for {category}: {sorted(missing)}")


if __name__ == "__main__":
    main()
