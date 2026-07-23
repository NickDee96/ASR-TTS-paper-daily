import copy
import json
import unittest
from pathlib import Path

import yaml

from daily_arxiv import _is_relevant_for_topic
from topic_classifier import (
    CLASSIFIER_VERSION,
    TopicConfigError,
    classify_paper,
    classify_topic,
    reclassify_records,
    validate_topic_config,
)


ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


class TopicClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cls.rules = config["keywords"]
        cls.cases = json.loads(
            (FIXTURES / "paper_cases.json").read_text(encoding="utf-8")
        )

    def test_repository_topic_config_is_valid(self) -> None:
        validate_topic_config(self.rules)

    def test_regression_cases_match_expected_topics_and_negatives(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["case"]):
                topics, classification, decisions = classify_paper(
                    case["title"], case["abstract"], self.rules
                )
                self.assertTrue(set(case["expected_topics"]).issubset(topics))
                self.assertTrue(set(case["expected_excluded_topics"]).isdisjoint(topics))
                self.assertEqual(classification["classifier_version"], CLASSIFIER_VERSION)
                self.assertEqual(
                    [match["topic"] for match in classification["matches"]],
                    topics,
                )
                self.assertTrue(all(decision["reason"] for decision in decisions))

    def test_evidence_names_exact_title_abstract_and_exclusion_terms(self) -> None:
        decision = classify_topic(
            "Automatic Speech Recognition with ASR",
            "Speech recognition under text-to-speech contamination.",
            "ASR",
            self.rules["ASR"],
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "excluded")
        self.assertIn("asr", decision["matched_title_terms"])
        self.assertIn("speech recognition", decision["matched_abstract_terms"])
        self.assertIn("text-to-speech", decision["exclusion_hits"])

    def test_legacy_compatibility_wrapper_uses_shared_classifier(self) -> None:
        rules = self.rules["TTS"]
        decision = classify_topic("Expressive TTS", "Speech synthesis", "TTS", rules)
        accepted, score, details = _is_relevant_for_topic(
            "Expressive TTS", "Speech synthesis", rules
        )

        self.assertEqual(accepted, decision["accepted"])
        self.assertEqual(score, decision["score"])
        self.assertEqual(details["matched_title_terms"], decision["matched_title_terms"])

    def test_config_validation_rejects_duplicates_contradictions_and_thresholds(self) -> None:
        invalid = {
            "Broken": {
                "filters": ["ASR", "asr"],
                "include": {"any": ["speech"], "all": []},
                "exclude": ["Speech"],
                "min_score": -1,
                "title_weight": -2,
            }
        }

        with self.assertRaises(TopicConfigError) as raised:
            validate_topic_config(invalid)

        message = str(raised.exception)
        self.assertIn("duplicate", message)
        self.assertIn("contradictions", message)
        self.assertIn("min_score", message)
        self.assertIn("title_weight", message)

    def test_reclassification_reports_changes_before_apply(self) -> None:
        paper = json.loads(
            (FIXTURES / "valid_paper.json").read_text(encoding="utf-8")
        )
        paper["topics"] = ["TTS"]
        paper["classification"] = {
            "classifier_version": "legacy",
            "matches": [{
                "topic": "TTS",
                "score": None,
                "threshold": None,
                "matched_title_terms": [],
                "matched_abstract_terms": [],
                "matched_all_terms": [],
                "exclusion_hits": [],
                "evidence_complete": False,
            }],
        }
        original = copy.deepcopy(paper)

        updated, report = reclassify_records({paper["id"]: paper}, self.rules)

        self.assertEqual(paper, original)
        self.assertEqual(report["changed_papers"], 1)
        self.assertIn("TTS", report["changes"][0]["removed"])
        self.assertEqual(
            updated[paper["id"]]["classification"]["classifier_version"],
            CLASSIFIER_VERSION,
        )


if __name__ == "__main__":
    unittest.main()