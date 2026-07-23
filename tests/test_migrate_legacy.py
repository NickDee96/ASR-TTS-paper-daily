import json
import unittest
from pathlib import Path

from migrate_legacy import migrate_archive
from paper_schema import load_schema, validate_paper


FIXTURES = Path(__file__).parent / "fixtures"


class LegacyMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive = json.loads(
            (FIXTURES / "legacy_archive.json").read_text(encoding="utf-8")
        )
        cls.expected_ids = {
            paper_id
            for papers in cls.archive.values()
            for paper_id in papers
        }
        cls.schema = load_schema()

    def test_migration_preserves_ids_and_merges_topics(self) -> None:
        records, report, backfill_ids = migrate_archive(
            self.archive,
            self.expected_ids,
        )

        self.assertTrue(report["valid"])
        self.assertEqual(set(records), self.expected_ids)
        self.assertEqual(report["migrated_records"], 6)
        self.assertEqual(backfill_ids, sorted(self.expected_ids))
        self.assertEqual(
            records["2607.00001"]["topics"],
            ["ASR", "Data Augmentation"],
        )
        self.assertEqual(records["2607.00001"]["authors"][0]["name"], "Māia Example")
        self.assertEqual(records["2607.00001"]["code"]["status"], "candidate")
        self.assertEqual(len(records["2607.00001"]["legacy"]["rows"]), 2)
        for record in records.values():
            self.assertEqual(validate_paper(record, self.schema), [])

    def test_migration_is_deterministic(self) -> None:
        first = migrate_archive(self.archive, self.expected_ids)
        second = migrate_archive(self.archive, self.expected_ids)

        self.assertEqual(first, second)

    def test_recoverable_pipe_title_is_preserved_and_queued(self) -> None:
        archive = {
            "Machine Translation": {
                "2602.20986": (
                    "|**2026-02-24**|**Naver Labs Europe @ WSDM CUP|"
                    "Multilingual Retrieval**|Thibault Formal et.al.|"
                    "[2602.20986](http://arxiv.org/abs/2602.20986)|"
                )
            }
        }

        records, report, backfill_ids = migrate_archive(archive, {"2602.20986"})

        self.assertTrue(report["valid"])
        self.assertEqual(
            records["2602.20986"]["title"],
            "Naver Labs Europe @ WSDM CUP|Multilingual Retrieval",
        )
        self.assertEqual(records["2602.20986"]["code"]["status"], "missing")
        self.assertEqual(
            report["recovered_anomalies"],
            {"2602.20986": ["Machine Translation"]},
        )
        self.assertEqual(backfill_ids, ["2602.20986"])

    def test_baseline_mismatch_is_blocking(self) -> None:
        _, report, _ = migrate_archive(
            self.archive,
            self.expected_ids | {"2607.99999"},
        )

        self.assertFalse(report["valid"])
        self.assertEqual(report["missing_expected_ids"], ["2607.99999"])


if __name__ == "__main__":
    unittest.main()