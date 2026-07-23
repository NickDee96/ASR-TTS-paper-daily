import datetime
import unittest
from pathlib import Path

from render_readme import DEFAULT_RECENT_LIMIT, render_readme


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "expected_readme.md"
NOW = datetime.datetime(2026, 7, 23, 18, tzinfo=datetime.timezone.utc)


def card(paper_id: str, title: str, updated: str) -> dict:
    return {
        "id": paper_id,
        "title": title,
        "abstract": "A focused abstract.",
        "authors": ["Máté Gedeon"],
        "published": updated,
        "updated": updated,
        "topics": ["ASR"],
        "primary_category": "cs.CL",
        "record_status": "complete",
        "code_status": "missing",
        "code_url": None,
        "links": {
            "abstract": f"https://arxiv.org/abs/{paper_id}",
            "pdf": f"https://arxiv.org/pdf/{paper_id}",
        },
    }


def site_card(updated_at: str = "2026-07-23T12:00:00Z") -> dict:
    return {
        "schema_version": 1,
        "updated_at": updated_at,
        "unique_papers": 2,
        "topic_assignments": 3,
        "verified_code": 1,
        "topics": {"TTS": 1, "ASR": 2},
    }


class ReadmeRendererTests(unittest.TestCase):
    def test_output_matches_snapshot_and_escapes_markdown(self):
        first = card("2607.00002", "Speech | Pipes", "2026-07-23")
        first["authors"] = ["Ada Lovelace", "Jose Alvarez"]
        first["topics"] = ["ASR", "TTS"]
        first["code_status"] = "verified"
        first["code_url"] = "https://github.com/example/speech-pipes"
        latest = {
            "schema_version": 1,
            "updated_at": "2026-07-23T12:00:00Z",
            "papers": [first, card("2607.00001", "A Unicode ASR Study", "2026-07-22")],
        }

        rendered = render_readme(
            latest,
            site_card(),
            now=NOW,
            explorer_url="https://example.test/explorer/",
        )

        self.assertEqual(FIXTURE_PATH.read_text(encoding="utf-8"), rendered)

    def test_digest_is_bounded_to_fifty_rows_and_under_size_budget(self):
        papers = [
            card(f"2607.{index:05d}", f"Paper {index} " + "x" * 500, "2026-07-23")
            for index in range(100)
        ]

        rendered = render_readme(
            {"updated_at": "2026-07-23T12:00:00Z", "papers": papers},
            site_card(),
            now=NOW,
        )

        paper_rows = [line for line in rendered.splitlines() if line.startswith("| 2026-")]
        self.assertEqual(DEFAULT_RECENT_LIMIT, len(paper_rows))
        self.assertIn("Paper 49", rendered)
        self.assertNotIn("Paper 50", rendered)
        self.assertLess(len(rendered.encode("utf-8")), 100_000)

    def test_stale_data_has_an_explicit_delayed_message(self):
        rendered = render_readme(
            {"updated_at": "2026-07-20T12:00:00Z", "papers": [card("2607.00001", "Old", "2026-07-20")]},
            site_card("2026-07-20T12:00:00Z"),
            now=NOW,
        )

        self.assertIn("Collection status: Delayed", rendered)
        self.assertIn("2026-07-20T12:00:00Z", rendered)

    def test_failed_run_preserves_last_successful_digest_with_warning(self):
        rendered = render_readme(
            {"updated_at": "2026-07-23T12:00:00Z", "papers": [card("2607.00001", "Still Available", "2026-07-23")]},
            site_card(),
            run_status={"state": "failed"},
            now=NOW,
        )

        self.assertIn("latest collection attempt failed", rendered)
        self.assertIn("Still Available", rendered)

    def test_empty_digest_is_honest_and_has_no_table(self):
        rendered = render_readme(
            {"updated_at": "2026-07-23T12:00:00Z", "papers": []},
            site_card(),
            now=NOW,
        )

        self.assertIn("No papers are currently available", rendered)
        self.assertIn("No recent papers are available yet", rendered)
        self.assertNotIn("| Updated | Paper |", rendered)

    def test_invalid_limits_are_rejected(self):
        latest = {"updated_at": "2026-07-23T12:00:00Z", "papers": []}
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            render_readme(latest, site_card(), recent_limit=51, now=NOW)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            render_readme(latest, site_card(), stale_after_hours=0, now=NOW)


if __name__ == "__main__":
    unittest.main()