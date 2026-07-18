#!/usr/bin/env python3
"""Build aligned p0/p1 JSONL files from Geneval prompt datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_ORIGINAL_DIR = CODE_ROOT / "Flow-Factory-Private" / "dataset" / "geneval"
DEFAULT_ENHANCED_DIR = (
    CODE_ROOT / "Flow-Factory-Private" / "dataset" / "geneval_enhanced"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "geneval_pe"
SPLITS = ("train", "test")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"expected JSONL input file, got missing path: {path}")

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
    return rows


def _required_prompt(
    row: dict[str, Any],
    key: str,
    *,
    description: str,
    path: Path,
    split: str,
    row_index: int,
) -> str:
    if key not in row:
        raise KeyError(
            f"expected field {key!r} for {description}, got missing field; "
            f"path={path}, split={split!r}, row={row_index}"
        )
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"expected non-empty string for {description}, "
            f"got {type(value).__name__}: {value!r}; "
            f"path={path}, split={split!r}, row={row_index}"
        )
    return value


def merge_split(
    original_path: Path,
    enhanced_path: Path,
    split: str,
) -> list[dict[str, object]]:
    """Validate and merge one aligned Geneval split."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"expected non-empty split name, got {split!r}")

    original_path = Path(original_path).expanduser().resolve()
    enhanced_path = Path(enhanced_path).expanduser().resolve()
    original_rows = _load_jsonl(original_path)
    enhanced_rows = _load_jsonl(enhanced_path)

    if len(original_rows) != len(enhanced_rows):
        raise ValueError(
            f"expected equal row counts for split {split!r}, "
            f"got original={len(original_rows)} from {original_path}, "
            f"enhanced={len(enhanced_rows)} from {enhanced_path}"
        )

    merged_rows: list[dict[str, object]] = []
    first_row_by_prompt: dict[str, int] = {}
    for row_index, (original, enhanced) in enumerate(
        zip(original_rows, enhanced_rows)
    ):
        p0 = _required_prompt(
            original,
            "prompt",
            description="original prompt",
            path=original_path,
            split=split,
            row_index=row_index,
        )
        enhanced_original = _required_prompt(
            enhanced,
            "orig_prompt",
            description="enhanced orig_prompt",
            path=enhanced_path,
            split=split,
            row_index=row_index,
        )
        p1 = _required_prompt(
            enhanced,
            "prompt",
            description="enhanced prompt",
            path=enhanced_path,
            split=split,
            row_index=row_index,
        )

        if p0 != enhanced_original:
            raise ValueError(
                "expected aligned original prompts, "
                f"got geneval.prompt={p0!r} and "
                f"geneval_enhanced.orig_prompt={enhanced_original!r}; "
                f"split={split!r}, row={row_index}"
            )
        if p0 in first_row_by_prompt:
            raise ValueError(
                f"expected unique p0 prompts, got duplicate {p0!r}; "
                f"split={split!r}, row={row_index}, "
                f"first_row={first_row_by_prompt[p0]}"
            )
        first_row_by_prompt[p0] = row_index

        merged_rows.append(
            {
                "p0": p0,
                "p1": p1,
                "tag": original.get("tag"),
                "include": original.get("include"),
                "exclude": original.get("exclude"),
                "source_split": split,
                "source_row": row_index,
            }
        )

    return merged_rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(
    original_dir: Path,
    enhanced_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Build train/test p0/p1 files and return their row counts."""
    original_dir = Path(original_dir).expanduser().resolve()
    enhanced_dir = Path(enhanced_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split in SPLITS:
        rows_by_split[split] = merge_split(
            original_dir / f"{split}.jsonl",
            enhanced_dir / f"{split}.jsonl",
            split,
        )

    for split, rows in rows_by_split.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    return {split: len(rows) for split, rows in rows_by_split.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired Geneval p0/p1 JSONL files for PE distillation."
    )
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--enhanced-dir", type=Path, default=DEFAULT_ENHANCED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_dataset(args.original_dir, args.enhanced_dir, args.output_dir)
    for split in SPLITS:
        print(f"{split}: {counts[split]}")


if __name__ == "__main__":
    main()
