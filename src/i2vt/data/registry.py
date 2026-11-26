"""Named training sets and benchmarks, fetched from the Hugging Face Hub.

A config names its data instead of pointing at local paths::

    data:
      train_dataset: hf:i2vt_17k     # or hf:action+relight, or hf:action@<revision>
    eval:
      bench: openve_pp               # filter_types and the judge follow

``resolve_hf_sources`` swaps each name for the files it downloads into the Hub cache before the run
writes its resolved configuration, so workers read local paths and the snapshot directory (named by
the commit hash) records the revision.

Every training row holds the source frame and the teacher's edit, so online and offline training on
the same name see the same rows. I2VT-17K is a set of parquet shards whose ``pool`` column names the
pool of each row; the first use of a pool writes its rows to ``pools/<name>.parquet`` next to the
download. OpenVE++ is a Hugging Face video folder: ``metadata.csv`` names each clip in ``file_name``.
"""

from __future__ import annotations

from dataclasses import dataclass

I2VT_17K = "sanaka87/I2VT-17K"
I2VT_17K_SHARDS = "data/train-*.parquet"
OPENVE_PP = "sanaka87/OpenVEPP-Bench"
GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-3.8-flash"


@dataclass(frozen=True)
class TrainSet:
    repo: str
    shards: str  # glob of the parquet shards in the repo
    note: str
    # A pool is the rows whose `pool` column equals the pool's name; False keeps every row.
    filtered: bool = True


@dataclass(frozen=True)
class Benchmark:
    repo: str
    subdir: str  # "" for a repo whose root is the benchmark
    filter_types: tuple[str, ...]
    judge: str
    note: str
    csv: str = "benchmark_videos.csv"  # one row per case, relative to the benchmark directory


OPENVE_EDIT_TYPES = (
    "global_style", "background_change", "local_change", "local_remove", "local_add",
    "subtitle_edit", "creative_edit")

TRAIN_DATASETS = {
    "openve": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "OpenVE-Bench edit types, 6026 pairs"),
    "action": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "action and expression changes, 7306 pairs"),
    "body_movement": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "jump, crouch, stand up, sit down, turn away, 1305 pairs"),
    "relight": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "scene relighting, 726 pairs"),
    "relight_wide": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "scene relighting with more lights and directions, 1864 pairs"),
    # The five pools above, in this order (6026 + 7306 + 1305 + 726 + 1864 = 17227 pairs).
    "i2vt_17k": TrainSet(I2VT_17K, I2VT_17K_SHARDS, "I2VT-17K, all five pools, 17227 pairs", filtered=False),
}
TRAIN_ALIASES = {"motion": "body_movement", "relight_v2": "relight_wide"}

BENCHMARKS = {
    "openve": Benchmark("Lewandofski/OpenVE-Bench", "", OPENVE_EDIT_TYPES, GEMINI_PRO,
                        "OpenVE-Bench without camera edits, 388 clips"),
    "openve_pp": Benchmark(OPENVE_PP, "", ("action_edit", "expression_edit", "motion_edit", "relight"), GEMINI_FLASH,
                           "OpenVE++ action change 73, expression change 42, body movement 42, "
                           "scene relighting 50, 207 clips", csv="metadata.csv"),
}


def is_hf(spec) -> bool:
    return isinstance(spec, str) and spec.startswith("hf:")


def _pool_name(part: str) -> str:
    """`hf:motion@abc` -> "body_movement": the canonical pool name without the revision."""
    name = part[len("hf:"):].split("@")[0]
    return TRAIN_ALIASES.get(name, name)


def _split(spec: str, table: dict, kind: str):
    name, _, revision = spec[len("hf:"):].partition("@")
    if table is TRAIN_DATASETS:
        name = TRAIN_ALIASES.get(name, name)
    if name not in table:
        raise ValueError(f"unknown {kind} {name!r}; choose from {sorted(table)}")
    return table[name], revision or None


def _names(spec: str) -> list[str]:
    """`hf:action+relight` -> ["hf:action", "hf:relight"]: pools trained together."""
    body = spec[len("hf:"):]
    return [f"hf:{part}" for part in body.split("+") if part]


def _snapshot_root(source: str):
    """The Hub snapshot directory of a downloaded shard directory or of a file inside it."""
    from pathlib import Path

    path = Path(source)
    return path.parent if path.is_dir() else path.parent.parent


def resolve_train_dataset(spec: str) -> str:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    parts = _names(spec)
    files = []
    for part in parts:
        entry, revision = _split(part, TRAIN_DATASETS, "training set")
        root = snapshot_download(entry.repo, repo_type="dataset", allow_patterns=[entry.shards], revision=revision)
        file = str(Path(root) / Path(entry.shards).parent)
        if entry.filtered:
            file = _pool_file(file, _pool_name(part), entry)
        files.append(file)
    if len(files) == 1:
        return files[0]
    # ImageEditDataset reads every *.parquet in a directory, so pools trained together are a
    # directory of links. Its name includes a hash of the resolved files, so another revision
    # gets another directory.
    import hashlib
    from pathlib import Path

    tag = "_".join(_pool_name(p) for p in parts)
    digest = hashlib.sha256("\0".join(files).encode()).hexdigest()[:12]
    root = _snapshot_root(files[0]) / "cotrain" / f"{tag}-{digest}"
    root.mkdir(parents=True, exist_ok=True)
    for index, file in enumerate(files):
        # The full set is a directory of shards; link each shard so the order stays the same.
        sources = sorted(Path(file).glob("*.parquet")) if Path(file).is_dir() else [Path(file)]
        for shard_index, shard in enumerate(sources):
            suffix = f"_{shard_index:05d}" if Path(file).is_dir() else ""
            link = root / f"{index:02d}_{_pool_name(parts[index])}{suffix}.parquet"
            if not link.exists():
                link.symlink_to(shard)
    return str(root)


def _pool_file(source: str, name: str, entry: TrainSet) -> str:
    """Write the rows of ``source`` whose ``pool`` column is ``name`` to a parquet beside it, once.

    ``source`` is the shard directory (or one parquet file) in a Hub snapshot directory named by its
    commit, so each revision gets its own pool files. Row groups are filtered one at a time to keep
    memory at one group of images.
    """
    from pathlib import Path

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    target = _snapshot_root(source) / "pools" / f"{name}.parquet"
    if target.is_file():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".parquet.partial")
    shards = sorted(Path(source).glob("*.parquet")) if Path(source).is_dir() else [Path(source)]
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {source}")
    readers = [pq.ParquetFile(shard) for shard in shards]
    rows = 0
    with pq.ParquetWriter(partial, readers[0].schema_arrow) as writer:
        for reader in readers:
            for index in range(reader.num_row_groups):
                group = reader.read_row_group(index)
                kept = group.filter(pc.equal(group.column("pool"), name))
                if kept.num_rows:
                    writer.write_table(kept)
                    rows += kept.num_rows
    if rows == 0:
        partial.unlink()
        raise ValueError(f"pool {name!r}: no rows of {source} have pool {name!r}")
    partial.replace(target)
    return str(target)


def benchmark(name: str) -> Benchmark:
    return _split(f"hf:{name}", BENCHMARKS, "benchmark")[0]


def case_video(row: dict) -> str:
    """Source clip of a benchmark row: ``original_video`` in OpenVE-Bench, ``file_name`` in OpenVE++."""
    return row.get("original_video") or row["file_name"]


def resolve_benchmark(name: str) -> tuple[str, str]:
    """(bench_dir, bench_csv) of a named benchmark, downloaded from the Hub."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    entry = benchmark(name)
    patterns = [f"{entry.subdir}/**"] if entry.subdir else None
    root = Path(snapshot_download(entry.repo, repo_type="dataset", allow_patterns=patterns))
    bench_dir = root / entry.subdir if entry.subdir else root
    csv = bench_dir / entry.csv
    if not csv.is_file():
        raise FileNotFoundError(f"{name}: {csv} missing after download")
    return str(bench_dir), str(csv)


def resolve_hf_sources(config, *, train: bool | None = None) -> list[str]:
    """Replace hf: names in ``config`` with downloaded paths. Returns log lines.

    ``train`` defaults to "unless the run only evaluates", so evaluation does not fetch the
    training set.
    """
    log = []
    if train is None:
        train = config.run.mode != "eval"
    if train and is_hf(config.data.train_dataset):
        spec = config.data.train_dataset
        config.data.train_dataset = resolve_train_dataset(spec)
        log.append(f"train_dataset {spec} -> {config.data.train_dataset}")
    if config.eval.bench_dir is None:
        config.eval.bench_dir, config.eval.bench_csv = resolve_benchmark(config.eval.bench)
        log.append(f"benchmark {config.eval.bench} -> {config.eval.bench_dir}")
    return log


def apply_cli_choices(config, dataset: str | None, bench: str | None) -> None:
    """--dataset / --bench: pick a named set without editing the YAML."""
    if dataset:
        spec = dataset if is_hf(dataset) else f"hf:{dataset}"
        for part in _names(spec):
            _split(part, TRAIN_DATASETS, "training set")
        if spec != config.data.train_dataset:
            # Another training set is another run: keep its outputs and cache apart.
            tag = "_".join(_pool_name(p) for p in _names(spec))
            config.run.name = f"{config.run.name}_{tag}"
            config.run.cache_dir = f"{config.run.cache_dir}_{tag}"
        config.data.train_dataset = spec
    if bench and bench != config.eval.bench:
        entry = benchmark(bench)
        config.eval.bench = bench
        config.eval.bench_dir = config.eval.bench_csv = None
        config.eval.filter_types = list(entry.filter_types)
        config.eval.scoring.model = entry.judge
