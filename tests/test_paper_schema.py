import copy
import json
import unittest
from pathlib import Path

from paper_schema import load_schema, records_from_document, validate_paper


FIXTURES = Path(__file__).parent / "fixtures"


class PaperSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()
        cls.valid_paper = json.loads(
            (FIXTURES / "valid_paper.json").read_text(encoding="utf-8")
        )

    def test_complete_paper_is_valid(self) -> None:
        self.assertEqual(validate_paper(self.valid_paper, self.schema), [])

    def test_invalid_paper_reports_structural_and_semantic_issues(self) -> None:
        paper = json.loads(
            (FIXTURES / "invalid_paper.json").read_text(encoding="utf-8")
        )

        issues = validate_paper(paper, self.schema)
        validators = {issue["validator"] for issue in issues}

        self.assertGreaterEqual(len(issues), 8)
        self.assertIn("required", validators)
        self.assertIn("format", validators)
        self.assertIn("topic_membership", validators)
        self.assertIn("date_order", validators)

    def test_partial_legacy_paper_does_not_invent_metadata(self) -> None:
        paper = {
            "schema_version": 1,
            "id": "2602.20986",
            "record_status": "partial",
            "topics": ["Machine Translation"],
            "classification": {
                "classifier_version": "legacy",
                "matches": [{
                    "topic": "Machine Translation",
                    "score": None,
                    "threshold": None,
                    "matched_title_terms": [],
                    "matched_abstract_terms": [],
                    "matched_all_terms": [],
                    "exclusion_hits": [],
                    "evidence_complete": False,
                }],
            },
            "links": {"abstract": None, "pdf": None},
            "code": {
                "status": "missing",
                "url": None,
                "source": None,
                "confidence": None,
                "checked_at": None,
                "evidence": [],
            },
            "source": {
                "origin": "legacy",
                "fetched_at": None,
                "arxiv_version": None,
            },
            "legacy": {
                "rows": [{"topic": "Machine Translation", "value": "broken row"}]
            },
        }

        self.assertEqual(validate_paper(paper, self.schema), [])

    def test_topic_evidence_must_match_topic_membership(self) -> None:
        paper = copy.deepcopy(self.valid_paper)
        paper["classification"]["matches"].pop()

        issues = validate_paper(paper, self.schema)

        self.assertTrue(
            any(issue["validator"] == "topic_membership" for issue in issues)
        )

    def test_records_from_document_supports_shard_shapes(self) -> None:
        paper = self.valid_paper

        self.assertEqual(len(records_from_document(paper)), 1)
        self.assertEqual(len(records_from_document([paper])), 1)
        self.assertEqual(len(records_from_document({"papers": [paper]})), 1)
        self.assertEqual(len(records_from_document({paper["id"]: paper})), 1)


if __name__ == "__main__":
    unittest.main()