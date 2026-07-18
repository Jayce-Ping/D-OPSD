"""Pure-Python prompt-pair records and collation helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


Resolution = tuple[int, int]


def _resolve_existing_path(path: str | Path, base_dir: Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute() or resolved_path.exists():
        return resolved_path.resolve()
    base_candidate = (base_dir / resolved_path).resolve()
    if base_candidate.exists():
        return base_candidate
    return resolved_path.resolve()


def _validate_resolution(value: Resolution, name: str) -> Resolution:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            f"expected {name} as a (width, height) pair, "
            f"got {type(value).__name__}: {value!r}"
        )
    width, height = value
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError(
            f"expected integer width and height for {name}, "
            f"got ({type(width).__name__}, {type(height).__name__}): {value!r}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"expected positive width and height for {name}, got {value!r}"
        )
    return width, height


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(
                    f"expected non-empty JSON object at {path}:{line_number}, got blank line"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"expected valid JSON object at {path}:{line_number}, "
                    f"got JSON decode error: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(
                    f"expected JSON object at {path}:{line_number}, "
                    f"got {type(row).__name__}: {row!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"expected at least one data row, got empty JSONL file: {path}")
    return rows


def _declared_aspect_ratio(item: dict[str, Any], row_index: int, path: Path) -> float | None:
    if "h*w" in item:
        value = item["h*w"]
        label = "h*w"
        height_index, width_index = 0, 1
    elif "w*h" in item:
        value = item["w*h"]
        label = "w*h"
        width_index, height_index = 0, 1
    else:
        return None

    if not isinstance(value, str) or value.count("*") != 1:
        raise ValueError(
            f"expected {label!r} as '<positive>*<positive>', got {value!r}; "
            f"row={row_index}, path={path}"
        )
    parts = value.split("*")
    try:
        width = float(parts[width_index])
        height = float(parts[height_index])
    except ValueError as error:
        raise ValueError(
            f"expected numeric dimensions in {label!r}, got {value!r}; "
            f"row={row_index}, path={path}"
        ) from error
    if width <= 0 or height <= 0:
        raise ValueError(
            f"expected positive dimensions in {label!r}, got {value!r}; "
            f"row={row_index}, path={path}"
        )
    return width / height


def _required_prompt(
    item: dict[str, Any],
    key: str,
    row_index: int,
    path: Path,
) -> str:
    if key not in item:
        raise KeyError(
            f"expected prompt field {key!r}, got missing field; "
            f"row={row_index}, path={path}"
        )
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"expected non-empty string for prompt field {key!r}, "
            f"got {type(value).__name__}: {value!r}; "
            f"row={row_index}, path={path}"
        )
    return value


class PromptPairRecords:
    """Prompt-pair records with aspect and fixed-resolution bucket metadata."""

    def __init__(
        self,
        jsonl_path: str | Path,
        target_resolutions: Sequence[Resolution],
        default_resolution: Resolution,
        data_root: str | Path | None = None,
    ) -> None:
        base_dir = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else Path(__file__).resolve().parent
        )
        self.jsonl_path = _resolve_existing_path(jsonl_path, base_dir)
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(
                f"expected prompt-pair JSONL file, got missing path: {self.jsonl_path}"
            )

        original_targets = [
            _validate_resolution(resolution, "target_resolution")
            for resolution in target_resolutions
        ]
        if not original_targets:
            raise ValueError(
                f"expected at least one target resolution, got {target_resolutions!r}"
            )
        self._aspect_target_count = len(original_targets)
        self.default_resolution = _validate_resolution(
            default_resolution, "default_resolution"
        )
        self.target_resolutions = list(original_targets)
        if self.default_resolution not in self.target_resolutions:
            self.target_resolutions.append(self.default_resolution)
        self._default_bucket_index = self.target_resolutions.index(
            self.default_resolution
        )

        self.data = _load_jsonl(self.jsonl_path)
        self.buckets: dict[int, list[int]] = {
            index: [] for index in range(len(self.target_resolutions))
        }
        self._build_buckets()

    def _build_buckets(self) -> None:
        aspect_targets = self.target_resolutions[: self._aspect_target_count]
        for row_index, item in enumerate(self.data):
            aspect_ratio = _declared_aspect_ratio(
                item, row_index, self.jsonl_path
            )
            if aspect_ratio is None:
                bucket_index = self._default_bucket_index
            else:
                bucket_index = min(
                    range(len(aspect_targets)),
                    key=lambda index: abs(
                        math.log(aspect_ratio)
                        - math.log(
                            aspect_targets[index][0] / aspect_targets[index][1]
                        )
                    ),
                )
            self.buckets[bucket_index].append(row_index)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, info: tuple[int, Resolution, tuple[str, str]]) -> dict[str, object]:
        if not isinstance(info, (tuple, list)) or len(info) != 3:
            raise TypeError(
                "expected item request as "
                "(row_index, (width, height), (student_key, teacher_key)), "
                f"got {type(info).__name__}: {info!r}"
            )
        row_index, resolution, prompt_key_pair = info
        if not isinstance(row_index, int):
            raise TypeError(
                f"expected integer row index, got {type(row_index).__name__}: "
                f"{row_index!r}"
            )
        if row_index < 0 or row_index >= len(self.data):
            raise IndexError(
                f"expected row index in [0, {len(self.data)}), got {row_index}"
            )
        width, height = _validate_resolution(resolution, "item resolution")
        if (
            not isinstance(prompt_key_pair, (tuple, list))
            or len(prompt_key_pair) != 2
            or not all(isinstance(key, str) and key for key in prompt_key_pair)
        ):
            raise TypeError(
                "expected prompt key pair as two non-empty strings, "
                f"got {prompt_key_pair!r}"
            )

        student_key, teacher_key = prompt_key_pair
        item = self.data[row_index]
        return {
            "student_prompt": _required_prompt(
                item, student_key, row_index, self.jsonl_path
            ),
            "teacher_prompt": _required_prompt(
                item, teacher_key, row_index, self.jsonl_path
            ),
            "prompt_pair": f"{student_key}->{teacher_key}",
            "width": width,
            "height": height,
        }


def read_validation_prompt_pairs(
    jsonl_path: str | Path,
    student_prompt_key: str,
    teacher_prompt_key: str,
    num_prompts: int,
    data_root: str | Path | None = None,
) -> list[tuple[str, str]]:
    if type(num_prompts) is not int:
        raise TypeError(
            f"expected int for num_prompts, got {type(num_prompts).__name__}: "
            f"{num_prompts!r}"
        )
    if num_prompts <= 0:
        raise ValueError(f"expected num_prompts > 0, got {num_prompts}")

    base_dir = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else Path(__file__).resolve().parent
    )
    path = _resolve_existing_path(jsonl_path, base_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"expected validation prompt JSONL file, got missing path: {path}"
        )
    rows = _load_jsonl(path)
    pairs = []
    for row_index, item in enumerate(rows[:num_prompts]):
        pairs.append(
            (
                _required_prompt(item, student_prompt_key, row_index, path),
                _required_prompt(item, teacher_prompt_key, row_index, path),
            )
        )
    return pairs


def collate_prompt_pairs(examples: list[dict[str, object]]) -> dict[str, object]:
    if not examples:
        raise ValueError("expected at least one prompt-pair example, got empty batch")

    first_resolution = (examples[0]["width"], examples[0]["height"])
    for index, example in enumerate(examples[1:], start=1):
        resolution = (example["width"], example["height"])
        if resolution != first_resolution:
            raise ValueError(
                "expected one resolution per batch, "
                f"got {first_resolution!r} at index 0 and {resolution!r} "
                f"at index {index}"
            )

    return {
        "student_prompts": [example["student_prompt"] for example in examples],
        "teacher_prompts": [example["teacher_prompt"] for example in examples],
        "prompt_pairs": [example["prompt_pair"] for example in examples],
        "width": first_resolution[0],
        "height": first_resolution[1],
    }
