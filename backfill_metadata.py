import argparse
import datetime
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from archive_baseline import ARXIV_ID_PATTERN
from paper_collector import (
    UTC,
    PaperCandidate,
    _as_utc,
    _candidate_version,
    _format_timestamp,
    _prefer_candidate,
    candidate_from_arxiv_result,
)
from paper_schema import load_schema, validate_paper
from paper_store import atomic_write_json


class MetadataSource(Protocol):
    def fetch(self, paper_ids: list[str]) -> Iterable[PaperCandidate]: ...


class ArxivMetadataSource:
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

    def fetch(self, paper_ids: list[str]) -> Iterable[PaperCandidate]:
        search = self._arxiv.Search(id_list=paper_ids, max_results=None)
        for result in self._client.results(search):
            yield candidate_from_arxiv_result(result)


def _ids_checksum(paper_ids: Iterable[str]) -> str:
    payload = "".join(f"{paper_id}\n" for paper_id in sorted(paper_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path, source_checksum: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "backfill_version": 1,
            "source_ids_sha256": source_checksum,
            "completed": {},
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("backfill_version") != 1
        or not isinstance(checkpoint.get("completed"), dict)
    ):
        raise ValueError("Invalid backfill checkpoint")
    if checkpoint.get("source_ids_sha256") != source_checksum:
        raise ValueError("Backfill checkpoint does not match the input paper IDs")
    return checkpoint


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _fetch_batch(
    source: MetadataSource,
    paper_ids: list[str],
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[dict[str, PaperCandidate], int, list[str]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        fetched: dict[str, PaperCandidate] = {}
        unexpected: set[str] = set()
        try:
            for candidate in source.fetch(paper_ids):
                if candidate.id not in paper_ids:
                    unexpected.add(candidate.id)
                    continue
                current = fetched.get(candidate.id)
                fetched[candidate.id] = (
                    candidate if current is None else _prefer_candidate(current, candidate)
                )
            return fetched, attempt, sorted(unexpected)
        except Exception:
            if attempt == max_attempts:
                raise
            delay = min(retry_base_seconds * (2 ** (attempt - 1)), retry_max_seconds)
            sleep(delay)
    raise AssertionError("unreachable")


def _is_withdrawn(candidate: PaperCandidate) -> bool:
    title = candidate.title.casefold().strip()
    abstract = candidate.abstract.casefold().strip()
    return (
        title.startswith("[withdrawn]")
        or title.startswith("withdrawn:")
        or abstract.startswith("this paper has been withdrawn")
    )


def _existing_date(record: dict[str, Any], field: str) -> datetime.date | None:
    value = record.get(field)
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _has_complete_metadata(record: dict[str, Any]) -> bool:
    return all((
        isinstance(record.get("title"), str) and bool(record["title"].strip()),
        isinstance(record.get("abstract"), str) and bool(record["abstract"].strip()),
        isinstance(record.get("authors"), list) and bool(record["authors"]),
        _existing_date(record, "published") is not None,
        _existing_date(record, "updated") is not None,
        isinstance(record.get("arxiv_categories"), list)
        and bool(record["arxiv_categories"]),
        isinstance(record.get("primary_category"), str)
        and bool(record["primary_category"].strip()),
    ))


def merge_metadata(
    record: dict[str, Any],
    candidate: PaperCandidate,
    checked_at: datetime.datetime,
) -> tuple[dict[str, Any], str, str | None]:
    updated_record = json.loads(json.dumps(record))
    existing_updated = _existing_date(record, "updated")
    incoming_updated = _as_utc(candidate.updated_at).date()
    incoming_is_current = existing_updated is None or incoming_updated >= existing_updated

    scalar_values = {
        "title": candidate.title,
        "abstract": candidate.abstract,
        "primary_category": candidate.primary_category,
        "comments": candidate.comments,
        "journal_reference": candidate.journal_reference,
        "doi": candidate.doi,
    }
    for field, value in scalar_values.items():
        if value and (incoming_is_current or not updated_record.get(field)):
            updated_record[field] = value

    authors = [{"name": author} for author in candidate.authors if author]
    if authors and (incoming_is_current or not updated_record.get("authors")):
        updated_record["authors"] = authors
    categories = list(dict.fromkeys(candidate.categories))
    if categories and (incoming_is_current or not updated_record.get("arxiv_categories")):
        updated_record["arxiv_categories"] = categories

    if not _existing_date(updated_record, "published"):
        updated_record["published"] = _as_utc(candidate.published_at).date().isoformat()
    if incoming_is_current:
        updated_record["updated"] = incoming_updated.isoformat()

    links = updated_record.setdefault("links", {"abstract": None, "pdf": None})
    for field, value in {
        "abstract": candidate.abstract_url,
        "pdf": candidate.pdf_url,
    }.items():
        if value and (incoming_is_current or not links.get(field)):
            links[field] = value

    existing_source = updated_record.get("source", {})
    first_seen_at = existing_source.get("first_seen_at")
    existing_version = existing_source.get("arxiv_version")
    existing_version_number = int(existing_version[1:]) if isinstance(existing_version, str) else 0
    incoming_version_number = _candidate_version(candidate)
    if (
        existing_source.get("origin") != "arxiv"
        or incoming_is_current
        or incoming_version_number > existing_version_number
    ):
        updated_record["source"] = {
            "origin": "arxiv",
            "fetched_at": _format_timestamp(checked_at),
            "first_seen_at": first_seen_at,
            "arxiv_version": candidate.arxiv_version,
        }
    elif "first_seen_at" not in existing_source:
        updated_record["source"] = {**existing_source, "first_seen_at": None}

    withdrawn = _is_withdrawn(candidate)
    availability_status = "withdrawn" if withdrawn else "available"
    reason = "arXiv metadata identifies the paper as withdrawn" if withdrawn else None
    updated_record["availability"] = {
        "status": availability_status,
        "reason": reason,
        "checked_at": _format_timestamp(checked_at),
    }
    updated_record["record_status"] = (
        "unavailable" if withdrawn else "complete" if _has_complete_metadata(updated_record) else "partial"
    )
    return updated_record, updated_record["record_status"], reason


def _mark_unavailable(
    record: dict[str, Any],
    status: str,
    reason: str,
    checked_at: datetime.datetime,
) -> dict[str, Any]:
    updated_record = json.loads(json.dumps(record))
    updated_record["availability"] = {
        "status": status,
        "reason": reason,
        "checked_at": _format_timestamp(checked_at),
    }
    if updated_record.get("record_status") != "complete":
        updated_record["record_status"] = "unavailable"
    return updated_record


def run_backfill(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    source: MetadataSource,
    checked_at: datetime.datetime,
    batch_size: int = 100,
    max_attempts: int = 3,
    retry_base_seconds: float = 3,
    retry_max_seconds: float = 30,
    sleep: Callable[[float], None] = time.sleep,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records_path = output_path if output_path.exists() else input_path
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise ValueError("Canonical paper archive must be an object keyed by arXiv ID")

    source_checksum = _ids_checksum(records)
    checkpoint = _load_checkpoint(checkpoint_path, source_checksum)
    pending = sorted(set(records) - set(checkpoint["completed"]))
    schema = load_schema()
    run_batches = 0
    fetched_results = 0
    unexpected_result_ids: set[str] = set()
    attempts_used = 0

    for batch in _chunks(pending, batch_size):
        if max_batches is not None and run_batches >= max_batches:
            break
        valid_ids = [paper_id for paper_id in batch if ARXIV_ID_PATTERN.fullmatch(paper_id)]
        malformed_ids = sorted(set(batch) - set(valid_ids))
        fetched, attempts, unexpected = _fetch_batch(
            source,
            valid_ids,
            max_attempts,
            retry_base_seconds,
            retry_max_seconds,
            sleep,
        ) if valid_ids else ({}, 0, [])
        attempts_used += attempts
        fetched_results += len(fetched)
        unexpected_result_ids.update(unexpected)

        for paper_id in malformed_ids:
            reason = "paper ID does not match the supported arXiv ID formats"
            records[paper_id] = _mark_unavailable(
                records[paper_id], "malformed", reason, checked_at
            )
            checkpoint["completed"][paper_id] = {
                "status": "malformed",
                "reason": reason,
                "checked_at": _format_timestamp(checked_at),
            }

        for paper_id in valid_ids:
            candidate = fetched.get(paper_id)
            if candidate is None:
                reason = "arXiv returned no result for the requested ID"
                records[paper_id] = _mark_unavailable(
                    records[paper_id], "unavailable", reason, checked_at
                )
                checkpoint["completed"][paper_id] = {
                    "status": "unavailable",
                    "reason": reason,
                    "checked_at": _format_timestamp(checked_at),
                }
                continue

            merged, record_status, reason = merge_metadata(
                records[paper_id], candidate, checked_at
            )
            issues = validate_paper(merged, schema)
            if issues:
                raise ValueError(f"Backfilled record {paper_id} is invalid: {issues}")
            records[paper_id] = merged
            availability_status = merged["availability"]["status"]
            checkpoint["completed"][paper_id] = {
                "status": availability_status if record_status == "unavailable" else record_status,
                "reason": reason,
                "checked_at": _format_timestamp(checked_at),
            }

        atomic_write_json(output_path, records)
        atomic_write_json(checkpoint_path, checkpoint)
        run_batches += 1

    remaining = sorted(set(records) - set(checkpoint["completed"]))
    status_counts = Counter(
        item["status"] for item in checkpoint["completed"].values()
    )
    reported_statuses = (
        "complete",
        "partial",
        "unavailable",
        "withdrawn",
        "malformed",
        "failed",
    )
    report = {
        "report_version": 1,
        "source_ids_sha256": source_checksum,
        "total_records": len(records),
        "completed_records": len(checkpoint["completed"]),
        "pending_records": len(remaining),
        "status_counts": {
            status: status_counts.get(status, 0)
            for status in reported_statuses
        },
        "run_batches": run_batches,
        "fetched_results": fetched_results,
        "attempts_used": attempts_used,
        "unexpected_result_ids": sorted(unexpected_result_ids),
        "complete": not remaining,
    }
    return records, checkpoint, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill canonical papers from arXiv IDs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backfill_config = config.get("backfill", {})
    source = ArxivMetadataSource(
        page_size=int(backfill_config.get("batch_size", 100)),
        delay_seconds=float(backfill_config.get("api_delay_seconds", 3)),
        num_retries=int(backfill_config.get("api_retries", 3)),
    )
    records, checkpoint, report = run_backfill(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint or Path(backfill_config["checkpoint_path"]),
        source=source,
        checked_at=datetime.datetime.now(UTC),
        batch_size=int(backfill_config.get("batch_size", 100)),
        max_attempts=int(backfill_config.get("max_attempts", 3)),
        retry_base_seconds=float(backfill_config.get("retry_base_seconds", 3)),
        retry_max_seconds=float(backfill_config.get("retry_max_seconds", 30)),
        max_batches=args.max_batches,
    )
    report_path = args.report or Path(backfill_config["report_path"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())