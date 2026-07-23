import json
import tempfile
import unittest
from pathlib import Path

from archive_baseline import analyze_archive, parse_legacy_row


FIXTURES = Path(__file__).parent / "fixtures"


class ArchiveBaselineTests(unittest.TestCase):
    def test_fixture_report_is_deterministic_and_distinguishes_assignments(self) -> None:
        archive = FIXTURES / "legacy_archive.json"

        first = analyze_archive(archive, [archive])
        second = analyze_archive(archive, [archive])

        self.assertEqual(first, second)
        self.assertEqual(first["unique_papers"], 6)
        self.assertEqual(first["topic_assignments"], 7)
        self.assertEqual(first["duplicate_assignments"], 1)
        self.assertEqual(first["multi_topic_papers"], 1)
        self.assertEqual(first["assignments_with_code"], 3)
        self.assertEqual(first["oldest_row_date"], "2026-07-01")
        self.assertEqual(first["newest_row_date"], "2026-07-06")
        self.assertEqual(first["malformed_record_count"], 0)
        self.assertEqual(first["artifact_sizes"][archive.as_posix()], archive.stat().st_size)

    def test_malformed_legacy_row_is_reported(self) -> None:
        archive_data = {
            "Machine Translation": {
                "2602.20986": (
                    "|**2026-02-24**|**Naver Labs Europe @ WSDM CUP|"
                    "Multilingual Retrieval**|Thibault Formal et.al.|"
                    "[2602.20986](http://arxiv.org/abs/2602.20986)|"
                )
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "archive.json"
            archive.write_text(json.dumps(archive_data), encoding="utf-8")

            report = analyze_archive(archive, [])

        self.assertEqual(report["malformed_record_count"], 1)
        self.assertEqual(
            report["malformed_records"][0]["reason"],
            "row is missing the code column",
        )
        self.assertEqual(
            parse_legacy_row(archive_data["Machine Translation"]["2602.20986"])["title"],
            "Naver Labs Europe @ WSDM CUP|Multilingual Retrieval",
        )

    def test_parser_accepts_verified_and_missing_code_rows(self) -> None:
        archive = json.loads((FIXTURES / "legacy_archive.json").read_text(encoding="utf-8"))

        verified = parse_legacy_row(archive["ASR"]["2607.00001"])
        missing = parse_legacy_row(archive["TTS"]["2607.00002"])

        self.assertIsNotNone(verified)
        self.assertIsNotNone(missing)
        self.assertIn("github.com/example/asr", verified["code"])
        self.assertEqual(missing["code"], "null")


if __name__ == "__main__":
    unittest.main()