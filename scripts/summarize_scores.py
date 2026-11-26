"""Average Gemini scores per run over the cases every run has.

Reads the score jsonl files that `i2vt` evaluation writes into each inference directory and prints
the three 1-5 scores (instruction compliance, visual quality and stability, consistency and detail
fidelity) per run, overall and per edit type.

    python scripts/summarize_scores.py \\
        --run I2VT-1.3B=outputs/inference/i2vt-1.3b/openve_pp \\
        --run I2VT-14B=outputs/inference/i2vt-14b/openve_pp
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from i2vt.data.registry import case_video

DIMS = ("instruction", "visual", "consistency")


def load_scores(directory: Path) -> dict[str, dict]:
    """The last score row per case across the run's *_score.jsonl files."""
    rows = {}
    for path in sorted(directory.glob("*_score.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["base"]] = row
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="a label and the inference directory holding its score files")
    parser.add_argument("--bench-csv", help="count only the cases listed in this benchmark CSV")
    parser.add_argument("--no-align", action="store_true",
                        help="average each run over its own cases instead of the cases every run has")
    args = parser.parse_args(argv)
    runs = {}
    for spec in args.run:
        label, _, directory = spec.partition("=")
        runs[label] = load_scores(Path(directory))
    keep = None
    if args.bench_csv:
        with open(args.bench_csv, encoding="utf-8") as handle:
            keep = {Path(case_video(row)).stem for row in csv.DictReader(handle)}
    common = None
    if not args.no_align:
        present = [set(rows) for rows in runs.values() if rows]
        common = set.intersection(*present) if present else set()
        print(f"{len(common)} cases shared by all runs")
    print(f"{'run':20s}{'n':>5s}  " + "  ".join(f"{d:>11s}" for d in DIMS))
    for label, rows in runs.items():
        cases = [r for base, r in rows.items()
                 if (keep is None or base in keep) and (common is None or base in common)]
        if not cases:
            print(f"{label:20s}  (no scored cases)")
            continue
        totals = [sum(r["scores"][i] for r in cases) / len(cases) for i in range(3)]
        print(f"{label:20s}{len(cases):>5d}  " + "  ".join(f"{x:11.2f}" for x in totals))
        by_type = defaultdict(list)
        for r in cases:
            by_type[r["edited_type"]].append(r["scores"])
        for edited_type, scores in sorted(by_type.items()):
            means = [sum(s[i] for s in scores) / len(scores) for i in range(3)]
            print(f"  {edited_type:18s}{len(scores):>5d}  " + "  ".join(f"{x:11.2f}" for x in means))


if __name__ == "__main__":
    main()
