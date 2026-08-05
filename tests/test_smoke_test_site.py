import tempfile
import unittest
from pathlib import Path

from smoke_test_site import evaluate_route_coverage


class SmokeTestSiteTests(unittest.TestCase):
    def _route(self, dist: Path, paper_id: str) -> None:
        route = dist / "papers" / paper_id / "index.html"
        route.parent.mkdir(parents=True, exist_ok=True)
        route.write_text("<h1>ok</h1>", encoding="utf-8")

    def test_full_coverage_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = Path(temporary_directory)
            for paper_id in ("2607.00001", "1307.4048"):
                self._route(dist, paper_id)
            result = evaluate_route_coverage({"2607.00001", "1307.4048"}, dist)
            self.assertTrue(result["healthy"])
            self.assertEqual(result["expected"], 2)
            self.assertEqual(result["missing_count"], 0)

    def test_missing_route_fails_and_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dist = Path(temporary_directory)
            self._route(dist, "2607.00001")
            result = evaluate_route_coverage({"2607.00001", "1307.4048"}, dist)
            self.assertFalse(result["healthy"])
            self.assertEqual(result["missing_count"], 1)
            self.assertIn("1307.4048", result["missing_sample"])


if __name__ == "__main__":
    unittest.main()
