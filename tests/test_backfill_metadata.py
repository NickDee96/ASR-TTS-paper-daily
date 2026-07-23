import datetime
import json
import tempfile
import unittest
from pathlib import Path

from backfill_metadata import merge_metadata, run_backfill
from migrate_legacy import migrate_archive
from paper_collector import UTC, PaperCandidate
from paper_schema import load_schema, validate_paper


FIXTURES = Path(__file__).parent / "fixtures"
CHECKED_AT = datetime.datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def metadata_candidate(
    paper_id: str,
    version: int = 1,
    updated: datetime.datetime = CHECKED_AT,
    title: str | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        id=paper_id,
        arxiv_version=f"v{version}",
        title=title or f"Complete {paper_id}",
        abstract=f"Complete abstract for {paper_id}",
        authors=("Full Author", "Second Author"),
        published_at=datetime.datetime(2026, 7, 1, tzinfo=UTC),
        updated_at=updated,
        categories=("cs.CL", "eess.AS"),
        primary_category="cs.CL",
        abstract_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        doi="10.0000/example",
    )


class FakeMetadataSource:
    def __init__(self, papers: dict[str, PaperCandidate], fail_call: int | None = None) -> None:
        self.papers = papers
        self.fail_call = fail_call
        self.calls: list[list[str]] = []

    def fetch(self, paper_ids):
        self.calls.append(list(paper_ids))
        if self.fail_call == len(self.calls):
            raise RuntimeError("backfill failed")
        for paper_id in paper_ids:
            if paper_id in self.papers:
                yield self.papers[paper_id]


class BackfillMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        archive = json.loads(
            (FIXTURES / "legacy_archive.json").read_text(encoding="utf-8")
        )
        expected_ids = {
            paper_id for topic in archive.values() for paper_id in topic
        }
        cls.legacy_records, _, _ = migrate_archive(archive, expected_ids)
        cls.schema = load_schema()

    def _write_input(self, root: Path, records=None) -> Path:
        path = root / "input.json"
        path.write_text(
            json.dumps(records or self.legacy_records, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_backfill_uses_bounded_batches_and_completes_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root)
            source = FakeMetadataSource({
                paper_id: metadata_candidate(paper_id)
                for paper_id in self.legacy_records
            })

            records, _, report = run_backfill(
                input_path=input_path,
                output_path=root / "output.json",
                checkpoint_path=root / "checkpoint.json",
                source=source,
                checked_at=CHECKED_AT,
                batch_size=2,
                sleep=lambda delay: None,
            )

        self.assertEqual([len(batch) for batch in source.calls], [2, 2, 2])
        self.assertTrue(report["complete"])
        self.assertEqual(report["status_counts"]["complete"], 6)
        self.assertEqual(report["status_counts"]["failed"], 0)
        for record in records.values():
            self.assertEqual(record["record_status"], "complete")
            self.assertEqual(record["availability"]["status"], "available")
            self.assertEqual(validate_paper(record, self.schema), [])

    def test_interrupted_run_resumes_after_last_committed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root)
            output_path = root / "output.json"
            checkpoint_path = root / "checkpoint.json"
            papers = {
                paper_id: metadata_candidate(paper_id)
                for paper_id in self.legacy_records
            }
            failing_source = FakeMetadataSource(papers, fail_call=2)

            with self.assertRaisesRegex(RuntimeError, "backfill failed"):
                run_backfill(
                    input_path=input_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint_path,
                    source=failing_source,
                    checked_at=CHECKED_AT,
                    batch_size=2,
                    max_attempts=1,
                    sleep=lambda delay: None,
                )

            committed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(len(committed["completed"]), 2)
            resumed_source = FakeMetadataSource(papers)
            _, _, report = run_backfill(
                input_path=input_path,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                source=resumed_source,
                checked_at=CHECKED_AT,
                batch_size=2,
                sleep=lambda delay: None,
            )

            self.assertTrue(report["complete"])
            self.assertEqual(sum(len(batch) for batch in resumed_source.calls), 4)

    def test_rerun_after_completion_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root)
            output_path = root / "output.json"
            checkpoint_path = root / "checkpoint.json"
            papers = {
                paper_id: metadata_candidate(paper_id)
                for paper_id in self.legacy_records
            }
            first_source = FakeMetadataSource(papers)
            run_backfill(
                input_path,
                output_path,
                checkpoint_path,
                first_source,
                CHECKED_AT,
                batch_size=3,
                sleep=lambda delay: None,
            )
            first_bytes = output_path.read_bytes()
            second_source = FakeMetadataSource(papers)

            _, _, report = run_backfill(
                input_path,
                output_path,
                checkpoint_path,
                second_source,
                CHECKED_AT + datetime.timedelta(days=1),
                batch_size=3,
                sleep=lambda delay: None,
            )

            self.assertEqual(second_source.calls, [])
            self.assertEqual(output_path.read_bytes(), first_bytes)
            self.assertTrue(report["complete"])

    def test_older_metadata_does_not_replace_newer_nonempty_fields(self) -> None:
        record = json.loads(
            (FIXTURES / "valid_paper.json").read_text(encoding="utf-8")
        )
        candidate = metadata_candidate(
            record["id"],
            version=1,
            updated=datetime.datetime(2026, 6, 1, tzinfo=UTC),
            title="Older title",
        )

        merged, _, _ = merge_metadata(record, candidate, CHECKED_AT)

        self.assertEqual(merged["title"], record["title"])
        self.assertEqual(merged["updated"], record["updated"])
        self.assertEqual(merged["source"]["arxiv_version"], "v2")
        self.assertEqual(merged["doi"], "10.0000/example")

    def test_missing_result_is_recorded_as_unavailable(self) -> None:
        paper_id = next(iter(self.legacy_records))
        records = {paper_id: self.legacy_records[paper_id]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root, records)

            updated, checkpoint, report = run_backfill(
                input_path,
                root / "output.json",
                root / "checkpoint.json",
                FakeMetadataSource({}),
                CHECKED_AT,
                sleep=lambda delay: None,
            )

        self.assertEqual(updated[paper_id]["record_status"], "unavailable")
        self.assertEqual(updated[paper_id]["availability"]["status"], "unavailable")
        self.assertEqual(checkpoint["completed"][paper_id]["status"], "unavailable")
        self.assertEqual(report["status_counts"]["unavailable"], 1)

    def test_withdrawn_result_is_distinguished(self) -> None:
        paper_id = next(iter(self.legacy_records))
        records = {paper_id: self.legacy_records[paper_id]}
        withdrawn = metadata_candidate(paper_id, title="[Withdrawn] Superseded paper")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root, records)

            updated, checkpoint, _ = run_backfill(
                input_path,
                root / "output.json",
                root / "checkpoint.json",
                FakeMetadataSource({paper_id: withdrawn}),
                CHECKED_AT,
                sleep=lambda delay: None,
            )

        self.assertEqual(updated[paper_id]["availability"]["status"], "withdrawn")
        self.assertEqual(updated[paper_id]["record_status"], "unavailable")
        self.assertEqual(checkpoint["completed"][paper_id]["status"], "withdrawn")

    def test_incomplete_response_is_checkpointed_as_partial(self) -> None:
        paper_id = next(iter(self.legacy_records))
        records = {paper_id: self.legacy_records[paper_id]}
        incomplete = PaperCandidate(
            id=paper_id,
            arxiv_version="v1",
            title="Recovered title",
            abstract="",
            authors=("Recovered Author",),
            published_at=datetime.datetime(2026, 7, 1, tzinfo=UTC),
            updated_at=CHECKED_AT,
            categories=(),
            primary_category="",
            abstract_url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root, records)

            updated, checkpoint, report = run_backfill(
                input_path,
                root / "output.json",
                root / "checkpoint.json",
                FakeMetadataSource({paper_id: incomplete}),
                CHECKED_AT,
                sleep=lambda delay: None,
            )

        self.assertEqual(updated[paper_id]["record_status"], "partial")
        self.assertEqual(checkpoint["completed"][paper_id]["status"], "partial")
        self.assertEqual(report["status_counts"]["partial"], 1)

    def test_malformed_id_is_recorded_without_an_api_request(self) -> None:
        source_id = next(iter(self.legacy_records))
        malformed_record = json.loads(json.dumps(self.legacy_records[source_id]))
        malformed_record["id"] = "bad-id"
        records = {"bad-id": malformed_record}
        source = FakeMetadataSource({})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = self._write_input(root, records)

            updated, checkpoint, report = run_backfill(
                input_path,
                root / "output.json",
                root / "checkpoint.json",
                source,
                CHECKED_AT,
                sleep=lambda delay: None,
            )

        self.assertEqual(source.calls, [])
        self.assertEqual(updated["bad-id"]["availability"]["status"], "malformed")
        self.assertEqual(checkpoint["completed"]["bad-id"]["status"], "malformed")
        self.assertEqual(report["status_counts"]["malformed"], 1)


if __name__ == "__main__":
    unittest.main()