import json
import tempfile
import unittest
from pathlib import Path

from check_archive_health import _unique_papers, evaluate_archive_health


class ArchiveHealthTests(unittest.TestCase):
    def test_growth_and_small_fluctuations_are_healthy(self) -> None:
        self.assertTrue(evaluate_archive_health(120, 100)["healthy"])
        self.assertTrue(evaluate_archive_health(95, 100)["healthy"])
        self.assertTrue(evaluate_archive_health(100, None)["healthy"])

    def test_empty_archive_is_rejected(self) -> None:
        result = evaluate_archive_health(0, 100)
        self.assertFalse(result["healthy"])
        self.assertIn("below the minimum", result["reason"])

    def test_severe_drop_is_rejected(self) -> None:
        result = evaluate_archive_health(50, 100)
        self.assertFalse(result["healthy"])
        self.assertIn("shrank", result["reason"])

    def test_first_ever_build_only_requires_minimum(self) -> None:
        self.assertTrue(evaluate_archive_health(1, None)["healthy"])
        self.assertFalse(evaluate_archive_health(0, None)["healthy"])

    def test_invalid_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_archive_health(10, 10, max_drop_fraction=1.0)
        with self.assertRaises(ValueError):
            evaluate_archive_health(10, 10, min_papers=0)

    def test_reads_unique_papers_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(json.dumps({"unique_papers": 42}), encoding="utf-8")
            self.assertEqual(_unique_papers(path), 42)

    def test_rejects_manifest_without_valid_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(json.dumps({"unique_papers": "many"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _unique_papers(path)


if __name__ == "__main__":
    unittest.main()
