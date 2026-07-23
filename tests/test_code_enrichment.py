import datetime
import json
import tempfile
import unittest
from pathlib import Path

from code_enrichment import CodeCandidate, enrich_code_links, resolve_code
from paper_collector import UTC


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_paper.json"
NOW = datetime.datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


class FakeCodeSource:
    def __init__(self, name, responses):
        self.name = name
        self.responses = responses
        self.calls = []

    def lookup(self, paper):
        self.calls.append(paper["id"])
        response = self.responses.get(paper["id"], [])
        if isinstance(response, Exception):
            raise response
        return response


class CodeEnrichmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paper = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_authoritative_mapping_has_precedence_and_is_verified(self) -> None:
        result = resolve_code([
            CodeCandidate(
                url="https://github.com/example/title-match",
                source="github",
                evidence=("title-only",),
            ),
            CodeCandidate(
                url="https://github.com/example/official",
                source="papers-with-code",
                evidence=("official",),
                authoritative=True,
                exact_arxiv_id=True,
            ),
        ], NOW)

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["url"], "https://github.com/example/official")
        self.assertEqual(result["confidence"], 1)

    def test_exact_id_can_verify_but_title_only_stays_candidate(self) -> None:
        exact = resolve_code([CodeCandidate(
            url="https://github.com/example/exact",
            source="github",
            evidence=("exact ID",),
            exact_arxiv_id=True,
        )], NOW)
        ambiguous = resolve_code([CodeCandidate(
            url="https://github.com/example/ambiguous",
            source="github",
            evidence=("title-only",),
        )], NOW)

        self.assertEqual(exact["status"], "verified")
        self.assertEqual(ambiguous["status"], "candidate")
        self.assertLess(ambiguous["confidence"], exact["confidence"])

    def test_deleted_candidate_is_ignored_and_no_result_is_missing(self) -> None:
        result = resolve_code([CodeCandidate(
            url="https://github.com/example/deleted",
            source="github",
            evidence=("deleted",),
            available=False,
        )], NOW)

        self.assertEqual(result["status"], "missing")
        self.assertIsNone(result["url"])

    def test_api_failure_preserves_existing_code(self) -> None:
        paper = json.loads(json.dumps(self.paper))
        existing_code = json.loads(json.dumps(paper["code"]))
        source = FakeCodeSource("failing", {paper["id"]: RuntimeError("offline")})
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "cache.json"

            updated, report = enrich_code_links(
                {paper["id"]: paper},
                [source],
                cache_path,
                NOW,
                max_attempts=1,
                sleep=lambda delay: None,
            )

        self.assertEqual(updated[paper["id"]]["code"], existing_code)
        self.assertEqual(report["outcomes"]["preserved_on_failure"], 1)
        self.assertEqual(report["failed_sources"][0]["error"], "RuntimeError")

    def test_fresh_cache_avoids_duplicate_external_lookup(self) -> None:
        paper = json.loads(json.dumps(self.paper))
        candidate = CodeCandidate(
            url="https://github.com/example/cached",
            source="authoritative",
            evidence=("official",),
            authoritative=True,
        )
        source = FakeCodeSource("authoritative", {paper["id"]: [candidate]})
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "cache.json"

            first, _ = enrich_code_links(
                {paper["id"]: paper}, [source], cache_path, NOW, sleep=lambda delay: None
            )
            second, report = enrich_code_links(
                first,
                [source],
                cache_path,
                NOW + datetime.timedelta(days=1),
                sleep=lambda delay: None,
            )

        self.assertEqual(source.calls, [paper["id"]])
        self.assertEqual(second[paper["id"]]["code"]["status"], "verified")
        self.assertEqual(report["cache_hits"], 1)

    def test_stale_cache_is_used_when_refresh_fails(self) -> None:
        paper = json.loads(json.dumps(self.paper))
        candidate = CodeCandidate(
            url="https://github.com/example/stale",
            source="source",
            evidence=("exact ID",),
            exact_arxiv_id=True,
        )
        source = FakeCodeSource("source", {paper["id"]: [candidate]})
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "cache.json"
            first, _ = enrich_code_links(
                {paper["id"]: paper}, [source], cache_path, NOW, sleep=lambda delay: None
            )
            source.responses[paper["id"]] = RuntimeError("offline")

            second, report = enrich_code_links(
                first,
                [source],
                cache_path,
                NOW + datetime.timedelta(days=8),
                refresh_after=datetime.timedelta(days=7),
                max_attempts=1,
                sleep=lambda delay: None,
            )

        self.assertEqual(second[paper["id"]]["code"]["url"], candidate.url)
        self.assertEqual(report["stale_cache_uses"], 1)


if __name__ == "__main__":
    unittest.main()