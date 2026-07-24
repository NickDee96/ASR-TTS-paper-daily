import datetime
import json
import tempfile
import unittest
from pathlib import Path

from paper_collector import (
    UTC,
    PaperCandidate,
    build_submitted_query,
    collect_topic,
    collection_window,
    run_collection,
    versionless_arxiv_id,
)


NOW = datetime.datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def candidate(index: int, version: int = 1, updated_days: int = 0) -> PaperCandidate:
    paper_id = f"2607.{index:05d}"
    published = NOW - datetime.timedelta(days=2)
    return PaperCandidate(
        id=paper_id,
        arxiv_version=f"v{version}",
        title=f"Paper {index}",
        abstract=f"Abstract {index}",
        authors=("Researcher",),
        published_at=published,
        updated_at=published + datetime.timedelta(days=updated_days),
        categories=("cs.CL",),
        primary_category="cs.CL",
        abstract_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


class SequenceSource:
    def __init__(self, attempts: list[list[PaperCandidate] | Exception]) -> None:
        self.attempts = attempts
        self.calls = 0
        self.windows: list[tuple[datetime.datetime, datetime.datetime]] = []

    def search(self, query, window_start, window_end):
        self.windows.append((window_start, window_end))
        response = self.attempts[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        yield from response


class MidPageFailureSource:
    def __init__(self, successful_first_topic: list[PaperCandidate]) -> None:
        self.successful_first_topic = successful_first_topic
        self.calls = 0

    def search(self, query, window_start, window_end):
        self.calls += 1
        if self.calls == 1:
            yield from self.successful_first_topic
            return
        for index in range(120):
            yield candidate(index + 1000)
        raise RuntimeError("page failed")


def accept_all(paper, topic, rules):
    return True


class PaperCollectorTests(unittest.TestCase):
    def test_versionless_ids_and_submitted_query(self) -> None:
        self.assertEqual(versionless_arxiv_id("2607.12345v3"), ("2607.12345", "v3"))
        self.assertEqual(versionless_arxiv_id("2607.12345"), ("2607.12345", None))
        query = build_submitted_query(
            'ASR OR "Speech Recognition"',
            NOW - datetime.timedelta(hours=72),
            NOW,
        )
        self.assertEqual(
            query,
            '(ASR OR "Speech Recognition") AND '
            'submittedDate:[202607201200 TO 202607231200]',
        )

    def test_more_than_one_page_is_fully_inspected_before_filtering(self) -> None:
        papers = [candidate(index) for index in range(250)]
        source = SequenceSource([papers])

        result = collect_topic(
            source=source,
            topic="ASR",
            query="ASR",
            rules={},
            window_start=NOW - datetime.timedelta(days=3),
            window_end=NOW,
            accept=lambda paper, topic, rules: int(paper.id.split(".")[1]) % 5 != 0,
            sleep=lambda delay: None,
        )

        self.assertEqual(result.fetched, 250)
        self.assertEqual(result.accepted, 200)
        self.assertEqual(result.rejected, 50)
        self.assertEqual(len(result.papers), 200)

    def test_retry_uses_bounded_exponential_backoff(self) -> None:
        source = SequenceSource([
            RuntimeError("first"),
            RuntimeError("second"),
            [candidate(1)],
        ])
        delays: list[float] = []

        result = collect_topic(
            source=source,
            topic="ASR",
            query="ASR",
            rules={},
            window_start=NOW - datetime.timedelta(days=3),
            window_end=NOW,
            accept=accept_all,
            max_attempts=3,
            retry_base_seconds=4,
            retry_max_seconds=6,
            sleep=delays.append,
        )

        self.assertEqual(result.attempts, 3)
        self.assertEqual(delays, [4, 6])

    def test_newer_revision_replaces_older_version_without_duplicate(self) -> None:
        version_one = candidate(1, version=1, updated_days=0)
        version_two = candidate(1, version=2, updated_days=1)
        source = SequenceSource([[version_one, version_two]])

        result = collect_topic(
            source=source,
            topic="ASR",
            query="ASR",
            rules={},
            window_start=NOW - datetime.timedelta(days=3),
            window_end=NOW,
            accept=accept_all,
            sleep=lambda delay: None,
        )

        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[version_one.id].arxiv_version, "v2")
        self.assertEqual(result.duplicate_versions, 1)

    def test_collection_window_overlaps_checkpoint(self) -> None:
        checkpoint = {
            "checkpoint_version": 1,
            "topics": {"ASR": {"successful_through": "2026-07-22T12:00:00Z"}},
        }

        start, end = collection_window(
            checkpoint,
            "ASR",
            NOW,
            datetime.timedelta(hours=72),
            datetime.timedelta(days=7),
        )
        first_start, _ = collection_window(
            checkpoint,
            "TTS",
            NOW,
            datetime.timedelta(hours=72),
            datetime.timedelta(days=7),
        )

        self.assertEqual(start, datetime.datetime(2026, 7, 19, 12, 0, tzinfo=UTC))
        self.assertEqual(end, NOW)
        self.assertEqual(first_start, NOW - datetime.timedelta(days=7))

    def test_reconciliation_mode_ignores_checkpoint_for_wider_window(self) -> None:
        checkpoint = {
            "checkpoint_version": 1,
            "topics": {"ASR": {"successful_through": "2026-07-22T12:00:00Z"}},
        }
        source = SequenceSource([[]])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            run_collection(
                topic_specs={"ASR": {"query": "ASR", "rules": {}}},
                source=source,
                checkpoint_path=checkpoint_path,
                output_path=root / "batch.json",
                window_end=NOW,
                accept=accept_all,
                initial_lookback=datetime.timedelta(days=14),
                ignore_checkpoint=True,
                sleep=lambda delay: None,
            )

        self.assertEqual(source.windows[0][0], NOW - datetime.timedelta(days=14))

    def test_checkpoint_stays_unchanged_when_collection_fails(self) -> None:
        checkpoint = {
            "checkpoint_version": 1,
            "topics": {"ASR": {"successful_through": "2026-07-20T12:00:00Z"}},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "checkpoint.json"
            output_path = root / "batch.json"
            checkpoint_text = json.dumps(checkpoint, sort_keys=True)
            checkpoint_path.write_text(checkpoint_text, encoding="utf-8")
            output_path.write_text("previous batch", encoding="utf-8")
            source = MidPageFailureSource([candidate(1)])

            with self.assertRaisesRegex(RuntimeError, "page failed"):
                run_collection(
                    topic_specs={
                        "ASR": {"query": "ASR", "rules": {}},
                        "TTS": {"query": "TTS", "rules": {}},
                    },
                    source=source,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    window_end=NOW,
                    accept=accept_all,
                    max_attempts=2,
                    sleep=lambda delay: None,
                )

            self.assertEqual(checkpoint_path.read_text(encoding="utf-8"), checkpoint_text)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous batch")

    def test_success_commits_batch_before_advancing_all_topic_checkpoints(self) -> None:
        shared = candidate(1)
        source = SequenceSource([[shared], [shared]])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "checkpoint.json"
            output_path = root / "batch.json"

            batch, checkpoint = run_collection(
                topic_specs={
                    "ASR": {"query": "ASR", "rules": {}},
                    "Data Augmentation": {"query": "augmentation", "rules": {}},
                },
                source=source,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                window_end=NOW,
                accept=accept_all,
                sleep=lambda delay: None,
            )

            self.assertEqual(len(batch["papers"]), 1)
            self.assertEqual(
                batch["papers"][shared.id]["matched_topics"],
                ["ASR", "Data Augmentation"],
            )
            self.assertEqual(
                batch["papers"][shared.id]["first_seen_at"],
                "2026-07-23T12:00:00Z",
            )
            self.assertEqual(set(checkpoint["topics"]), {"ASR", "Data Augmentation"})
            self.assertEqual(
                json.loads(checkpoint_path.read_text(encoding="utf-8")),
                checkpoint,
            )
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                batch,
            )


if __name__ == "__main__":
    unittest.main()