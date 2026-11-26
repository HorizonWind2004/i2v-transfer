from __future__ import annotations

import csv
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base / p


def _open_image_from_value(value: Any, base: Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(_resolve_path(path, base)).convert("RGB")
    if isinstance(value, str) and value:
        return Image.open(_resolve_path(value, base)).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def _row_task(row: dict[str, Any]) -> str | None:
    """Sub-task under edit_type (e.g. smile or head_down within action_edit).

    I2VT-17K keeps it in the metadata JSON. A task column takes precedence.
    """
    task = row.get("task")
    if task:
        return str(task)
    metadata = row.get("metadata")
    if isinstance(metadata, str) and metadata:
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return None
    task = metadata.get("task") if isinstance(metadata, dict) else None
    return str(task) if task else None


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


class ImageEditDataset(Dataset):
    """Loads image-edit rows from CSV or HF-style parquet.

    Parquet image columns use the HuggingFace Image feature layout:
    struct<bytes: binary, path: string>.
    """

    def __init__(self, base_path: str | Path, dataset_path: str | Path, parquet_cache_size: int = 2):
        self.base = Path(base_path)
        self.path = Path(dataset_path)
        if self.path.suffix.lower() == ".csv":
            self.rows = self._load_csv(self.path)
            self._parquet_files = []
            self._row_groups = []
            self._row_group_cache = OrderedDict()
        else:
            self.rows = []
            self._parquet_files, self._row_groups = self._open_parquet(self.path)
            self._row_group_cache: OrderedDict[tuple[int, int], list[dict[str, Any]]] = OrderedDict()
        self.parquet_cache_size = max(1, int(parquet_cache_size))

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _open_parquet(path: Path):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading parquet datasets requires pyarrow.") from exc

        if path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found under {path}")
        else:
            files = [path]
        parquet_files = [pq.ParquetFile(file) for file in files]
        row_groups: list[tuple[int, int, int, int]] = []
        start = 0
        for file_idx, parquet_file in enumerate(parquet_files):
            for row_group_idx in range(parquet_file.num_row_groups):
                row_count = parquet_file.metadata.row_group(row_group_idx).num_rows
                row_groups.append((start, start + row_count, file_idx, row_group_idx))
                start += row_count
        return parquet_files, row_groups

    def __len__(self):
        if self._row_groups:
            return self._row_groups[-1][1]
        return len(self.rows)

    def _get_parquet_row(self, idx: int) -> dict[str, Any]:
        for start, end, file_idx, row_group_idx in self._row_groups:
            if start <= idx < end:
                key = (file_idx, row_group_idx)
                rows = self._row_group_cache.get(key)
                if rows is None:
                    rows = self._parquet_files[file_idx].read_row_group(row_group_idx).to_pylist()
                    self._row_group_cache[key] = rows
                    while len(self._row_group_cache) > self.parquet_cache_size:
                        self._row_group_cache.popitem(last=False)
                else:
                    self._row_group_cache.move_to_end(key)
                return rows[idx - start]
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        row = self._get_parquet_row(idx) if self._row_groups else self.rows[idx]
        src = _first_nonempty(row, ("source_image", "image", "src_video", "source", "source_path"))
        joyedit_src = _first_nonempty(row, ("joyedit_input_image", "joy_input_image", "image_teacher_source"))
        # The stored teacher edit, read only by offline training. Online training asks the
        # teacher at the student's own latents, so a dataset without this column still works.
        target = _first_nonempty(row, ("joyedit_target_image", "target_image", "edited_image"))
        prompt = _first_nonempty(row, ("prompt", "image_edit_prompt", "instruction", "edit"))
        if src is None or prompt is None:
            raise KeyError(f"Need a source image and an edit instruction, got {row.keys()}")
        video_edit_prompt = _first_nonempty(row, ("video_edit_prompt", "video_prompt"))
        joy_edit_prompt = _first_nonempty(row, ("joy_edit_prompt", "image_edit_prompt", "image_prompt"))
        source_path = _first_nonempty(row, ("source_path", "src_video", "source"))
        if isinstance(source_path, dict):
            source_path = source_path.get("path")
        path = str(_resolve_path(source_path, self.base)) if source_path else ""
        prompt = str(prompt)
        source = _open_image_from_value(src, self.base)
        return {
            "source": source,
            "joyedit_source": _open_image_from_value(joyedit_src, self.base) if joyedit_src is not None else source,
            "target": _open_image_from_value(target, self.base) if target is not None else None,
            "prompt": prompt,
            "video_edit_prompt": str(video_edit_prompt or prompt),
            "joy_edit_prompt": str(joy_edit_prompt or prompt),
            "path": path,
            "key": _first_nonempty(row, ("key", "id", "uid")),
            "edit_type": _first_nonempty(row, ("edit_type", "category")),
            "task": _row_task(row),
        }


def build_image_edit_dataset(base_path: str | Path, dataset_path: str | Path) -> ImageEditDataset:
    return ImageEditDataset(base_path, dataset_path)
