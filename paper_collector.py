import argparse
import datetime
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from paper_store import atomic_write_json


UTC = datetime.timezone.utc
VERSION_SUFFIX_PATTERN = re.compile(r"v(?P<version>\d+)$", re.IGNORECASE)


def _as_utc(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_timestamp(value: datetime.datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _query_timestamp(value: datetime.datetime) -> str:
    return _as_utc(value).strftime("%Y%m%d%H%M")


def versionless_arxiv_id(short_id: str) -> tuple[str, str | None]:
    match = VERSION_SUFFIX_PATTERN.search(short_id)
    if match is None:
        return short_id, None
    return short_id[:match.start()], f"v{match.group('version')}"


def build_submitted_query(
    base_query: str,
    window_start: datetime.datetime,
    window_end: datetime.datetime,
) -> str:
    return (
        f"({base_query}) AND submittedDate:"
        f"[{_query_timestamp(window_start)} TO {_query_timestamp(window_end)}]"
    )


@dataclass(frozen=True)
class PaperCandidate:
    id: str
    arxiv_version: str | None
    title: str
    abstract: str
    authors: tuple[str, ...]
    published_at: datetime.datetime
    updated_at: datetime.datetime
    categories: tuple[str, ...]
    primary_category: str
    abstract_url: str
    pdf_url: str | None
    comments: str | None = None
    journal_reference: str | None = None
    doi: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "arxiv_version": self.arxiv_version,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [{"name": author} for author in self.authors],
            "published": _as_utc(self.published_at).date().isoformat(),
            "updated": _as_utc(self.updated_at).date().isoformat(),
            "arxiv_categories": list(self.categories),
            "primary_category": self.primary_category,
            "links": {
                "abstract": self.abstract_url,
                "pdf": self.pdf_url,
            },
            "comments": self.comments,
            "journal_reference": self.journal_reference,
            "doi": self.doi,
        }


def candidate_from_arxiv_result(result: Any) -> PaperCandidate:
    paper_id, version = versionless_arxiv_id(result.get_short_id())
    return PaperCandidate(
        id=paper_id,
        arxiv_version=version,
        title=result.title.strip(),
        abstract=result.summary.replace("\n", " ").strip(),
        authors=tuple(author.name for author in result.authors),
        published_at=_as_utc(result.published),
        updated_at=_as_utc(result.updated),
        categories=tuple(result.categories),
        primary_category=result.primary_category,
        abstract_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=getattr(result, "pdf_url", None),
        comments=getattr(result, "comment", None),
        journal_reference=getattr(result, "journal_ref", None),
        doi=getattr(result, "doi", None),
    )


class PaperSource(Protocol):
    def search(
        self,
        query: str,
        window_start: datetime.datetime,
        window_end: datetime.datetime,
    ) -> Iterable[PaperCandidate]: ...


class ArxivSearchSource:
    def __init__(
        self,
        page_size: int = 100,
        delay_seconds: float = 3,
        num_retries: int = 3,
    ) -> None:
        try:
            import arxiv
        except ImportError as error:
            raise RuntimeError(
                "The arxiv package is required; install requirements.txt first"
            ) from error

        self._arxiv = arxiv
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def _results(self, query: str, sort_by: Any) -> Iterable[Any]:
        search = self._arxiv.Search(
            query=query,
            sort_by=sort_by,
            sort_order=self._arxiv.SortOrder.Descending,
        )
        return self._client.results(search)

    def search(
        self,
        query: str,
        window_start: datetime.datetime,
        window_end: datetime.datetime,
    ) -> Iterable[PaperCandidate]:
        submitted_query = build_submitted_query(query, window_start, window_end)
        for result in self._results(
            submitted_query,
            self._arxiv.SortCriterion.SubmittedDate,
        ):
            yield candidate_from_arxiv_result(result)

        for result in self._results(
            query,
            self._arxiv.SortCriterion.LastUpdatedDate,
        ):
            candidate = candidate_from_arxiv_result(result)
            if candidate.updated_at > window_end:
                continue
            if candidate.updated_at < window_start:
                break
            yield candidate


@dataclass(frozen=True)
class TopicCollection:
    papers: dict[str, PaperCandidate]
    fetched: int
    accepted: int
    rejected: int
    duplicate_versions: int
    attempts: int


CandidateFilter = Callable[[PaperCandidate, str, dict[str, Any]], bool]


def _candidate_version(candidate: PaperCandidate) -> int:
    if candidate.arxiv_version is None:
        return 0
    return int(candidate.arxiv_version.removeprefix("v"))


def _prefer_candidate(
    current: PaperCandidate,
    candidate: PaperCandidate,
) -> PaperCandidate:
    current_key = (_as_utc(current.updated_at), _candidate_version(current))
    candidate_key = (_as_utc(candidate.updated_at), _candidate_version(candidate))
    return candidate if candidate_key > current_key else current


def collect_topic(
    source: PaperSource,
    topic: str,
    query: str,
    rules: dict[str, Any],
    window_start: datetime.datetime,
    window_end: datetime.datetime,
    accept: CandidateFilter,
    max_attempts: int = 3,
    retry_base_seconds: float = 3,
    retry_max_seconds: float = 30,
    sleep: Callable[[float], None] = time.sleep,
) -> TopicCollection:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        papers: dict[str, PaperCandidate] = {}
        fetched = 0
        accepted = 0
        rejected = 0
        duplicate_versions = 0
        try:
            for candidate in source.search(query, window_start, window_end):
                fetched += 1
                if not accept(candidate, topic, rules):
                    rejected += 1
                    continue
                accepted += 1
                current = papers.get(candidate.id)
                if current is None:
                    papers[candidate.id] = candidate
                else:
                    duplicate_versions += 1
                    papers[candidate.id] = _prefer_candidate(current, candidate)
            return TopicCollection(
                papers=papers,
                fetched=fetched,
                accepted=accepted,
                rejected=rejected,
                duplicate_versions=duplicate_versions,
                attempts=attempt,
            )
        except Exception:
            if attempt == max_attempts:
                raise
            delay = min(retry_base_seconds * (2 ** (attempt - 1)), retry_max_seconds)
            sleep(delay)

    raise AssertionError("unreachable")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"checkpoint_version": 1, "topics": {}}
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_version") != 1
        or not isinstance(checkpoint.get("topics"), dict)
    ):
        raise ValueError("Invalid collector checkpoint")
    return checkpoint


def collection_window(
    checkpoint: dict[str, Any],
    topic: str,
    window_end: datetime.datetime,
    overlap: datetime.timedelta,
    initial_lookback: datetime.timedelta,
) -> tuple[datetime.datetime, datetime.datetime]:
    topic_checkpoint = checkpoint["topics"].get(topic)
    if isinstance(topic_checkpoint, dict) and isinstance(
        topic_checkpoint.get("successful_through"), str
    ):
        successful_through = _parse_timestamp(topic_checkpoint["successful_through"])
        start = successful_through - overlap
    else:
        start = _as_utc(window_end) - initial_lookback
    return start, _as_utc(window_end)


def _validate_batch(batch: dict[str, Any]) -> None:
    papers = batch.get("papers")
    if not isinstance(papers, dict):
        raise ValueError("Collection batch papers must be an object")
    for paper_id, paper in papers.items():
        if not isinstance(paper, dict) or paper.get("id") != paper_id:
            raise ValueError(f"Collection batch key mismatch for {paper_id}")
        topics = paper.get("matched_topics")
        if not isinstance(topics, list) or not topics:
            raise ValueError(f"Collection batch paper {paper_id} has no matched topics")


def run_collection(
    topic_specs: dict[str, dict[str, Any]],
    source: PaperSource,
    checkpoint_path: Path,
    output_path: Path,
    window_end: datetime.datetime,
    accept: CandidateFilter,
    overlap: datetime.timedelta = datetime.timedelta(hours=72),
    initial_lookback: datetime.timedelta = datetime.timedelta(days=7),
    max_attempts: int = 3,
    retry_base_seconds: float = 3,
    retry_max_seconds: float = 30,
    sleep: Callable[[float], None] = time.sleep,
    ignore_checkpoint: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("Checkpoint and collection output paths must differ")

    checkpoint = load_checkpoint(checkpoint_path)
    proposed_checkpoint = json.loads(json.dumps(checkpoint))
    batch_candidates: dict[str, PaperCandidate] = {}
    batch_topics: dict[str, set[str]] = {}
    topic_reports: dict[str, dict[str, Any]] = {}
    window_checkpoint = (
        {"checkpoint_version": 1, "topics": {}}
        if ignore_checkpoint
        else checkpoint
    )

    for topic in sorted(topic_specs):
        spec = topic_specs[topic]
        window_start, topic_window_end = collection_window(
            window_checkpoint,
            topic,
            window_end,
            overlap,
            initial_lookback,
        )
        collection = collect_topic(
            source=source,
            topic=topic,
            query=spec["query"],
            rules=spec.get("rules", {}),
            window_start=window_start,
            window_end=topic_window_end,
            accept=accept,
            max_attempts=max_attempts,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            sleep=sleep,
        )

        for paper_id, candidate in collection.papers.items():
            existing = batch_candidates.get(paper_id)
            batch_candidates[paper_id] = (
                candidate if existing is None else _prefer_candidate(existing, candidate)
            )
            batch_topics.setdefault(paper_id, set()).add(topic)

        topic_reports[topic] = {
            "window_start": _format_timestamp(window_start),
            "window_end": _format_timestamp(topic_window_end),
            "fetched": collection.fetched,
            "accepted": collection.accepted,
            "rejected": collection.rejected,
            "unique_accepted": len(collection.papers),
            "duplicate_versions": collection.duplicate_versions,
            "attempts": collection.attempts,
        }
        proposed_checkpoint["topics"][topic] = {
            "successful_through": _format_timestamp(topic_window_end)
        }

    batch_papers: dict[str, dict[str, Any]] = {}
    for paper_id, candidate in sorted(batch_candidates.items()):
        batch_papers[paper_id] = candidate.to_json()
        batch_papers[paper_id]["matched_topics"] = sorted(batch_topics[paper_id])
        batch_papers[paper_id]["first_seen_at"] = _format_timestamp(window_end)

    batch = {
        "batch_version": 1,
        "window_end": _format_timestamp(window_end),
        "topics": topic_reports,
        "papers": batch_papers,
    }
    _validate_batch(batch)
    atomic_write_json(output_path, batch)
    atomic_write_json(checkpoint_path, proposed_checkpoint)
    return batch, proposed_checkpoint


def _configured_query(filters: list[str]) -> str:
    return " OR ".join(
        f'"{term}"' if len(term.split()) > 1 else term
        for term in filters
    )


def _configured_accept(
    candidate: PaperCandidate,
    topic: str,
    rules: dict[str, Any],
) -> bool:
    from topic_classifier import classify_topic

    return classify_topic(candidate.title, candidate.abstract, topic, rules)["accepted"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a resumable arXiv update batch.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collector_config = config.get("collector", {})
    topic_specs = {
        topic: {
            "query": _configured_query(rules["filters"]),
            "rules": rules,
        }
        for topic, rules in config["keywords"].items()
    }
    source = ArxivSearchSource(
        page_size=int(collector_config.get("page_size", 100)),
        delay_seconds=float(collector_config.get("api_delay_seconds", 3)),
        num_retries=int(collector_config.get("api_retries", 3)),
    )
    batch, _ = run_collection(
        topic_specs=topic_specs,
        source=source,
        checkpoint_path=args.checkpoint or Path(collector_config["checkpoint_path"]),
        output_path=args.output or Path(collector_config["batch_path"]),
        window_end=datetime.datetime.now(UTC),
        accept=_configured_accept,
        overlap=datetime.timedelta(hours=float(collector_config.get("overlap_hours", 72))),
        initial_lookback=datetime.timedelta(
            days=float(
                collector_config.get(
                    "reconciliation_lookback_days" if args.reconcile else "initial_lookback_days",
                    14 if args.reconcile else 7,
                )
            )
        ),
        max_attempts=int(collector_config.get("max_attempts", 3)),
        retry_base_seconds=float(collector_config.get("retry_base_seconds", 3)),
        retry_max_seconds=float(collector_config.get("retry_max_seconds", 30)),
        ignore_checkpoint=args.reconcile,
    )
    print(json.dumps({
        "papers": len(batch["papers"]),
        "window_end": batch["window_end"],
        "topics": batch["topics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())