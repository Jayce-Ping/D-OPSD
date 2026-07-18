import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_geneval_pe_dataset import build_dataset, merge_split


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def original_row(prompt: str) -> dict:
    return {
        "tag": "color_attr",
        "include": "[]",
        "prompt": prompt,
        "exclude": "[]",
    }


def enhanced_row(original_prompt: str, prompt: str) -> dict:
    return {
        "tag": "color_attr",
        "include": "[]",
        "prompt": prompt,
        "exclude": "[]",
        "orig_prompt": original_prompt,
    }


class GenevalDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.original_path = self.root / "original.jsonl"
        self.enhanced_path = self.root / "enhanced.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_merge_split_builds_paired_rows(self) -> None:
        write_jsonl(self.original_path, [original_row("a red cube")])
        write_jsonl(
            self.enhanced_path,
            [enhanced_row("a red cube", "A polished red cube on a table.")],
        )

        rows = merge_split(self.original_path, self.enhanced_path, "train")

        self.assertEqual(
            rows[0],
            {
                "p0": "a red cube",
                "p1": "A polished red cube on a table.",
                "tag": "color_attr",
                "include": "[]",
                "exclude": "[]",
                "source_split": "train",
                "source_row": 0,
            },
        )

    def test_merge_split_rejects_count_mismatch(self) -> None:
        write_jsonl(self.original_path, [original_row("a red cube")])
        write_jsonl(self.enhanced_path, [])

        with self.assertRaisesRegex(
            ValueError,
            r"expected equal row counts for split 'train'.*original=1.*enhanced=0",
        ):
            merge_split(self.original_path, self.enhanced_path, "train")

    def test_merge_split_rejects_prompt_mismatch(self) -> None:
        write_jsonl(self.original_path, [original_row("a red cube")])
        write_jsonl(
            self.enhanced_path,
            [enhanced_row("a blue cube", "A polished blue cube.")],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"expected aligned original prompts.*split='train'.*row=0",
        ):
            merge_split(self.original_path, self.enhanced_path, "train")

    def test_merge_split_rejects_empty_enhanced_prompt(self) -> None:
        write_jsonl(self.original_path, [original_row("a red cube")])
        write_jsonl(self.enhanced_path, [enhanced_row("a red cube", "  ")])

        with self.assertRaisesRegex(
            ValueError,
            r"expected non-empty string for enhanced prompt.*split='train'.*row=0",
        ):
            merge_split(self.original_path, self.enhanced_path, "train")

    def test_merge_split_rejects_duplicate_original_prompt(self) -> None:
        write_jsonl(
            self.original_path,
            [original_row("a red cube"), original_row("a red cube")],
        )
        write_jsonl(
            self.enhanced_path,
            [
                enhanced_row("a red cube", "First enhancement."),
                enhanced_row("a red cube", "Second enhancement."),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"expected unique p0 prompts.*split='train'.*row=1.*first_row=0",
        ):
            merge_split(self.original_path, self.enhanced_path, "train")

    def test_build_dataset_writes_both_splits(self) -> None:
        original_dir = self.root / "geneval"
        enhanced_dir = self.root / "geneval_enhanced"
        output_dir = self.root / "output"
        for split in ("train", "test"):
            write_jsonl(original_dir / f"{split}.jsonl", [original_row(f"{split} p0")])
            write_jsonl(
                enhanced_dir / f"{split}.jsonl",
                [enhanced_row(f"{split} p0", f"{split} p1")],
            )

        counts = build_dataset(original_dir, enhanced_dir, output_dir)

        self.assertEqual(counts, {"train": 1, "test": 1})
        self.assertTrue((output_dir / "train.jsonl").is_file())
        self.assertTrue((output_dir / "test.jsonl").is_file())

    def test_build_dataset_validates_all_splits_before_writing(self) -> None:
        original_dir = self.root / "geneval"
        enhanced_dir = self.root / "geneval_enhanced"
        output_dir = self.root / "output"
        write_jsonl(original_dir / "train.jsonl", [original_row("train p0")])
        write_jsonl(
            enhanced_dir / "train.jsonl",
            [enhanced_row("train p0", "train p1")],
        )
        write_jsonl(original_dir / "test.jsonl", [original_row("test p0")])
        write_jsonl(
            enhanced_dir / "test.jsonl",
            [enhanced_row("different p0", "test p1")],
        )

        with self.assertRaisesRegex(ValueError, r"split='test'"):
            build_dataset(original_dir, enhanced_dir, output_dir)

        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
