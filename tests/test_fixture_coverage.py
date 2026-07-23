import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "paper_cases.json"
EXPECTED_TOPICS = {
    "ASR",
    "TTS",
    "Machine Translation",
    "Small Language Models",
    "Data Augmentation",
    "Synthetic Generation",
}


class FixtureCoverageTests(unittest.TestCase):
    def test_collector_fixtures_cover_required_regressions(self) -> None:
        cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        covered_topics = {
            topic
            for case in cases
            for topic in case["expected_topics"]
        }

        self.assertEqual(covered_topics, EXPECTED_TOPICS)
        self.assertTrue(all(case["expected_excluded_topics"] for case in cases))
        self.assertTrue(any(len(case["expected_topics"]) > 1 for case in cases))
        self.assertTrue(any(case["short_id"].endswith("v2") for case in cases))
        self.assertTrue(any(case["published"] != case["updated"] for case in cases))
        self.assertTrue(any(case["code_lookup_outcome"] == "missing" for case in cases))
        self.assertTrue(any(case["code_lookup_outcome"] == "http_error" for case in cases))
        self.assertTrue(any(not case["title"].isascii() for case in cases))


if __name__ == "__main__":
    unittest.main()