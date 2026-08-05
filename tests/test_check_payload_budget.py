import tempfile
import unittest
from pathlib import Path

from check_payload_budget import evaluate_payload_budget


def _write(path: Path, kilobytes: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * int(kilobytes * 1024))


class PayloadBudgetTests(unittest.TestCase):
    def _dist(self, root: Path) -> Path:
        dist = root / "dist"
        _write(dist / "_astro" / "client.hash.js", 150)
        _write(dist / "_astro" / "vendor.hash.js", 120)
        _write(dist / "index.html", 40)
        _write(dist / "search" / "index.html", 12)
        _write(dist / "data" / "manifest.json", 1)
        return dist

    def test_healthy_build_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = self._dist(Path(temporary_directory))
            result = evaluate_payload_budget(dist)
            self.assertTrue(result["healthy"], result["issues"])
            self.assertEqual(result["js_chunks"], 2)
            self.assertEqual(result["largest_js_kb"], 150.0)

    def test_oversized_chunk_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = self._dist(Path(temporary_directory))
            _write(dist / "_astro" / "huge.hash.js", 260)
            result = evaluate_payload_budget(dist)
            self.assertFalse(result["healthy"])
            self.assertTrue(any("huge.hash.js" in issue for issue in result["issues"]))

    def test_full_archive_leaking_into_feed_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = self._dist(Path(temporary_directory))
            _write(dist / "index.html", 500)
            result = evaluate_payload_budget(dist)
            self.assertFalse(result["healthy"])
            self.assertTrue(any("feed page" in issue for issue in result["issues"]))

    def test_bloated_initial_manifest_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = self._dist(Path(temporary_directory))
            _write(dist / "data" / "manifest.json", 60)
            result = evaluate_payload_budget(dist)
            self.assertFalse(result["healthy"])
            self.assertTrue(any("manifest.json" in issue for issue in result["issues"]))

    def test_total_js_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = self._dist(Path(temporary_directory))
            for index in range(6):
                _write(dist / "_astro" / f"chunk{index}.js", 130)
            result = evaluate_payload_budget(dist)
            self.assertFalse(result["healthy"])
            self.assertTrue(any("Total JS" in issue for issue in result["issues"]))

    def test_missing_build_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                evaluate_payload_budget(Path(temporary_directory) / "dist")


if __name__ == "__main__":
    unittest.main()
