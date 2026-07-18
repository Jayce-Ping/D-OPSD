import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prompt_pair_data import (
    PromptPairRecords,
    collate_prompt_pairs,
    read_validation_prompt_pairs,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


class PromptPairDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.jsonl_path = self.root / "data.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_text_only_row_uses_default_resolution(self) -> None:
        write_jsonl(self.jsonl_path, [{"p0": "short", "p1": "enhanced"}])
        records = PromptPairRecords(
            self.jsonl_path,
            target_resolutions=[(1024, 1024), (1152, 896)],
            default_resolution=(512, 512),
        )

        self.assertIn((512, 512), records.target_resolutions)
        default_bucket = records.target_resolutions.index((512, 512))
        self.assertEqual(records.buckets[default_bucket], [0])
        item = records[(0, (512, 512), ("p0", "p1"))]
        self.assertEqual((item["width"], item["height"]), (512, 512))

    def test_sized_row_uses_existing_aspect_bucket(self) -> None:
        write_jsonl(
            self.jsonl_path,
            [{"p0": "short", "p1": "enhanced", "h*w": "896*1152"}],
        )
        records = PromptPairRecords(
            self.jsonl_path,
            target_resolutions=[(1024, 1024), (1152, 896)],
            default_resolution=(512, 512),
        )

        aspect_bucket = records.target_resolutions.index((1152, 896))
        default_bucket = records.target_resolutions.index((512, 512))
        self.assertEqual(records.buckets[aspect_bucket], [0])
        self.assertEqual(records.buckets[default_bucket], [])

    def test_missing_prompt_key_fails_with_context(self) -> None:
        write_jsonl(self.jsonl_path, [{"p0": "short"}])
        records = PromptPairRecords(
            self.jsonl_path,
            target_resolutions=[(1024, 1024)],
            default_resolution=(512, 512),
        )

        with self.assertRaisesRegex(
            KeyError,
            r"expected prompt field 'p1'.*row=0.*data.jsonl",
        ):
            records[(0, (512, 512), ("p0", "p1"))]

    def test_empty_prompt_fails_with_context(self) -> None:
        write_jsonl(self.jsonl_path, [{"p0": "short", "p1": " "}])
        records = PromptPairRecords(
            self.jsonl_path,
            target_resolutions=[(1024, 1024)],
            default_resolution=(512, 512),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"expected non-empty string for prompt field 'p1'.*row=0.*data.jsonl",
        ):
            records[(0, (512, 512), ("p0", "p1"))]

    def test_collate_returns_prompts_and_resolution(self) -> None:
        batch = collate_prompt_pairs(
            [
                {
                    "student_prompt": "short 1",
                    "teacher_prompt": "enhanced 1",
                    "prompt_pair": "p0->p1",
                    "width": 512,
                    "height": 512,
                },
                {
                    "student_prompt": "short 2",
                    "teacher_prompt": "enhanced 2",
                    "prompt_pair": "p0->p1",
                    "width": 512,
                    "height": 512,
                },
            ]
        )

        self.assertEqual(batch["student_prompts"], ["short 1", "short 2"])
        self.assertEqual(batch["teacher_prompts"], ["enhanced 1", "enhanced 2"])
        self.assertEqual(batch["width"], 512)
        self.assertEqual(batch["height"], 512)

    def test_collate_rejects_mixed_resolutions(self) -> None:
        examples = [
            {
                "student_prompt": "short 1",
                "teacher_prompt": "enhanced 1",
                "prompt_pair": "p0->p1",
                "width": 512,
                "height": 512,
            },
            {
                "student_prompt": "short 2",
                "teacher_prompt": "enhanced 2",
                "prompt_pair": "p0->p1",
                "width": 768,
                "height": 512,
            },
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"expected one resolution per batch.*\(512, 512\).*\(768, 512\)",
        ):
            collate_prompt_pairs(examples)

    def test_read_validation_prompt_pairs_returns_requested_rows(self) -> None:
        write_jsonl(
            self.jsonl_path,
            [
                {"p0": "short 1", "p1": "enhanced 1"},
                {"p0": "short 2", "p1": "enhanced 2"},
            ],
        )

        pairs = read_validation_prompt_pairs(
            self.jsonl_path,
            student_prompt_key="p0",
            teacher_prompt_key="p1",
            num_prompts=1,
        )

        self.assertEqual(pairs, [("short 1", "enhanced 1")])

    def test_read_validation_prompt_pairs_rejects_missing_field(self) -> None:
        write_jsonl(self.jsonl_path, [{"p0": "short"}])

        with self.assertRaisesRegex(
            KeyError,
            r"expected prompt field 'p1'.*row=0.*data.jsonl",
        ):
            read_validation_prompt_pairs(
                self.jsonl_path,
                student_prompt_key="p0",
                teacher_prompt_key="p1",
                num_prompts=1,
            )


if __name__ == "__main__":
    unittest.main()
