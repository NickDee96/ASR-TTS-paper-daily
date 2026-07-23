import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from paper_schema import load_schema, validate_paper


class DuplicateKeyError(ValueError):
    pass


JsonValidator = Callable[[Any], None]
ReplaceOperation = Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file, object_pairs_hook=_reject_duplicate_keys)


def atomic_write_json(
    path: Path,
    value: Any,
    validator: JsonValidator | None = None,
    replace: ReplaceOperation = os.replace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(rendered)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        reloaded = load_json(temporary_path)
        if validator is not None:
            validator(reloaded)
        replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_canonical_archive(records: Any) -> None:
    if not isinstance(records, dict):
        raise ValueError("Canonical archive must be an object keyed by arXiv ID")
    schema = load_schema()
    failures: dict[str, list[dict[str, str]]] = {}
    for paper_id, record in records.items():
        if not isinstance(record, dict) or record.get("id") != paper_id:
            failures[paper_id] = [{
                "path": "$.id",
                "validator": "archive_key",
                "message": "record id must equal its archive key",
            }]
            continue
        issues = validate_paper(record, schema)
        if issues:
            failures[paper_id] = issues
    if failures:
        raise ValueError(f"Canonical archive validation failed: {failures}")


def load_canonical_archive(path: Path) -> dict[str, dict[str, Any]]:
    records = load_json(path)
    validate_canonical_archive(records)
    return records


def write_canonical_archive(
    path: Path,
    records: dict[str, dict[str, Any]],
    allowed_removed_ids: Iterable[str] = (),
    replace: ReplaceOperation = os.replace,
) -> None:
    validate_canonical_archive(records)
    if path.exists():
        previous = load_canonical_archive(path)
        allowed_removed = set(allowed_removed_ids)
        unexpected_removed = set(previous) - set(records) - allowed_removed
        if unexpected_removed:
            raise ValueError(
                "Canonical archive would unexpectedly remove paper IDs: "
                + ", ".join(sorted(unexpected_removed)[:20])
            )
    atomic_write_json(path, records, validate_canonical_archive, replace=replace)


def merge_canonical_records(
    existing: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    validate_canonical_archive(existing)
    validate_canonical_archive(incoming)
    merged = json.loads(json.dumps(existing))
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for paper_id in sorted(incoming):
        if paper_id not in merged:
            merged[paper_id] = incoming[paper_id]
            counts["inserted"] += 1
        elif merged[paper_id] == incoming[paper_id]:
            counts["unchanged"] += 1
        else:
            merged[paper_id] = incoming[paper_id]
            counts["updated"] += 1
    validate_canonical_archive(merged)
    return dict(sorted(merged.items())), counts


def canonical_shard_path(root: Path, published: str) -> Path:
    try:
        published_date = datetime.date.fromisoformat(published)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid publication date for shard path: {published}") from error
    relative = Path(f"{published_date.year:04d}") / f"{published_date.month:02d}.json"
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in candidate.parents:
        raise ValueError("Canonical shard path escapes its root")
    return candidate


def build_run_manifest(
    batch: dict[str, Any],
    merge_counts: dict[str, int],
    failed_enrichments: int = 0,
) -> dict[str, Any]:
    topics = batch.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("Collection batch is missing topic reports")
    fetched = sum(int(report.get("fetched", 0)) for report in topics.values())
    accepted = sum(int(report.get("accepted", 0)) for report in topics.values())
    rejected = sum(int(report.get("rejected", 0)) for report in topics.values())
    return {
        "manifest_version": 1,
        "window_end": batch.get("window_end"),
        "topics": topics,
        "totals": {
            "fetched": fetched,
            "accepted": accepted,
            "rejected": rejected,
            "rejected_by_reason": {"topic_filter": rejected},
            "inserted": int(merge_counts.get("inserted", 0)),
            "updated": int(merge_counts.get("updated", 0)),
            "unchanged": int(merge_counts.get("unchanged", 0)),
            "failed_enrichments": int(failed_enrichments),
        },
    }


def validate_run_manifest(manifest: dict[str, Any], allow_empty: bool = False) -> None:
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Run manifest is missing totals")
    fetched = int(totals.get("fetched", 0))
    accepted = int(totals.get("accepted", 0))
    rejected = int(totals.get("rejected", 0))
    if fetched != accepted + rejected:
        raise ValueError("Run manifest fetched count must equal accepted plus rejected")
    if fetched == 0 and not allow_empty:
        raise ValueError("Unexpected empty collection run")
    for name, value in totals.items():
        if name == "rejected_by_reason":
            continue
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Run manifest count {name} must be a non-negative integer")


def reconcile_archive(
    records: dict[str, dict[str, Any]],
    expected_ids: set[str],
    recovered_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validate_canonical_archive(records)
    validate_canonical_archive(recovered_records)
    missing_before = sorted(expected_ids - set(records))
    repairs = {
        paper_id: recovered_records[paper_id]
        for paper_id in missing_before
        if paper_id in recovered_records
    }
    repaired, counts = merge_canonical_records(records, repairs)
    report = {
        "report_version": 1,
        "missing_before": missing_before,
        "repaired_ids": sorted(repairs),
        "unresolved_ids": sorted(expected_ids - set(repaired)),
        "unexpected_ids": sorted(set(repaired) - expected_ids),
        "merge_counts": counts,
    }
    return repaired, report