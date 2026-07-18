import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from validation_seeds import create_validation_seeds


class ValidationSeedTests(unittest.TestCase):
    def test_repeated_calls_return_identical_seed_lists(self) -> None:
        self.assertEqual(
            create_validation_seeds(3, 2026),
            create_validation_seeds(3, 2026),
        )

    def test_each_sample_receives_a_distinct_seed(self) -> None:
        self.assertEqual(create_validation_seeds(3, 2026), [2026, 2027, 2028])

    def test_rejects_non_positive_sample_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"expected num_samples > 0, got 0",
        ):
            create_validation_seeds(0, 2026)

    def test_rejects_invalid_types(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"expected int for num_samples, got str",
        ):
            create_validation_seeds("3", 2026)
        with self.assertRaisesRegex(
            TypeError,
            r"expected int for base_seed, got float",
        ):
            create_validation_seeds(3, 2026.0)


if __name__ == "__main__":
    unittest.main()
