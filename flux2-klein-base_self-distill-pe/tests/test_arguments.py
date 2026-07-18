import argparse
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from arguments import positive_int


class ArgumentValidationTests(unittest.TestCase):
    def test_positive_int_accepts_positive_value(self) -> None:
        self.assertEqual(positive_int("512"), 512)

    def test_positive_int_rejects_zero_and_negative_values(self) -> None:
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError,
                    rf"expected a positive integer, got {value!r}",
                ):
                    positive_int(value)

    def test_positive_int_rejects_non_integer_value(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            r"expected a positive integer, got 'wide'",
        ):
            positive_int("wide")


if __name__ == "__main__":
    unittest.main()
