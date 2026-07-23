import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from archive_baseline import CODE_LINK_PATTERN, parse_legacy_row
from paper_schema import load_schema, validate_paper


def _paper_ids_checksum(paper_ids: set[str]) -> str:
    payload = "".join(f"{paper_id}\n" for paper_id in sorted(paper_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _choose_value(values: list[str]) -> str | None:
    populated = [value for value in values if value]
    if not populated:
        return None
    counts = Counter(populated)
    return sorted(counts, key=lambda value: (-counts[value], -len(value), value))[0]


def _first_author(author_label: str | None) -> str | None:
    if not author_label:
        return None
    author = re.sub(r"\s+et\.al\.?$", "", author_label, flags=re.IGNORECASE).strip()
    return author or None


def _legacy_topic_match(topic: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "score": None,
        "threshold": None,
        "matched_title_terms": [],
        "matched_abstract_terms": [],
        "matched_all_terms": [],
        "exclusion_hits": [],
        "evidence_complete": False,
    }


def _candidate_code(url: str | None) -> dict[str, Any]:
    if url is None:
        return {
            "status": "missing",
            "url": None,
            "source": None,
            "confidence": None,
            "checked_at": None,
            "evidence": [],
        }
    return {
        "status": "candidate",
        "url": url,
        "source": "legacy-unverified",
        "confidence": 0,
        "checked_at": None,
        "evidence": ["migrated from the unverified legacy code column"],
    }


def _build_record(
    paper_id: str,
    rows: list[tuple[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_rows: list[tuple[str, dict[str, str]]] = []
    parse_failures: list[str] = []
    recovered_anomalies: list[str] = []

    for topic, row in rows:
        if not isinstance(row, str):
            parse_failures.append(topic)
            continue
        parsed = parse_legacy_row(row)
        if parsed is None:
            parse_failures.append(topic)
            continue
        if not parsed["code"]:
            recovered_anomalies.append(topic)
        parsed_rows.append((topic, parsed))

    titles = [parsed["title"] for _, parsed in parsed_rows]
    author_labels = [parsed["authors"] for _, parsed in parsed_rows]
    dates = [parsed["date"] for _, parsed in parsed_rows]
    paper_urls = [parsed["paper_url"] for _, parsed in parsed_rows]
    code_urls = [
        match.group("url")
        for _, parsed in parsed_rows
        if (match := CODE_LINK_PATTERN.search(parsed["code"])) is not None
    ]

    title = _choose_value(titles)
    author = _first_author(_choose_value(author_labels))
    updated = max(dates) if dates else None
    paper_url = _choose_value(paper_urls)
    code_url = _choose_value(code_urls)
    topics = sorted({topic for topic, _ in rows})

    record: dict[str, Any] = {
        "schema_version": 1,
        "id": paper_id,
        "record_status": "partial",
        "title": title,
        "abstract": None,
        "authors": [{"name": author}] if author else [],
        "published": None,
        "updated": updated,
        "arxiv_categories": [],
        "primary_category": None,
        "topics": topics,
        "classification": {
            "classifier_version": "legacy",
            "matches": [_legacy_topic_match(topic) for topic in topics],
        },
        "links": {
            "abstract": paper_url,
            "pdf": None,
        },
        "code": _candidate_code(code_url),
        "source": {
            "origin": "legacy",
            "fetched_at": None,
            "arxiv_version": None,
        },
        "legacy": {
            "rows": [
                {"topic": topic, "value": row if isinstance(row, str) else json.dumps(row)}
                for topic, row in sorted(rows, key=lambda item: item[0])
            ]
        },
    }

    conflicts = {
        field: sorted(set(values))
        for field, values in {
            "title": titles,
            "author_label": author_labels,
            "updated": dates,
            "paper_url": paper_urls,
            "code_url": code_urls,
        }.items()
        if len(set(values)) > 1
    }
    details = {
        "parse_failure_topics": sorted(parse_failures),
        "recovered_anomaly_topics": sorted(recovered_anomalies),
        "conflicts": conflicts,
    }
    return record, details


def migrate_archive(
    archive: dict[str, Any],
    expected_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    grouped_rows: dict[str, list[tuple[str, Any]]] = {}
    invalid_topics: list[str] = []
    for topic in sorted(archive):
        topic_rows = archive[topic]
        if not isinstance(topic_rows, dict):
            invalid_topics.append(topic)
            continue
        for paper_id in sorted(topic_rows):
            grouped_rows.setdefault(paper_id, []).append((topic, topic_rows[paper_id]))

    schema = load_schema()
    records: dict[str, dict[str, Any]] = {}
    parse_failures: dict[str, list[str]] = {}
    recovered_anomalies: dict[str, list[str]] = {}
    field_conflicts: dict[str, dict[str, list[str]]] = {}
    schema_issues: dict[str, list[dict[str, str]]] = {}

    for paper_id in sorted(grouped_rows):
        record, details = _build_record(paper_id, grouped_rows[paper_id])
        records[paper_id] = record
        if details["parse_failure_topics"]:
            parse_failures[paper_id] = details["parse_failure_topics"]
        if details["recovered_anomaly_topics"]:
            recovered_anomalies[paper_id] = details["recovered_anomaly_topics"]
        if details["conflicts"]:
            field_conflicts[paper_id] = details["conflicts"]
        issues = validate_paper(record, schema)
        if issues:
            schema_issues[paper_id] = issues

    migrated_ids = set(records)
    baseline_ids = expected_ids if expected_ids is not None else migrated_ids
    missing_expected_ids = sorted(baseline_ids - migrated_ids)
    unexpected_ids = sorted(migrated_ids - baseline_ids)
    backfill_ids = sorted(
        paper_id
        for paper_id, record in records.items()
        if record["record_status"] != "complete"
    )

    report = {
        "report_version": 1,
        "source_topic_count": len(archive),
        "invalid_topics": invalid_topics,
        "baseline_id_count": len(baseline_ids),
        "source_unique_papers": len(grouped_rows),
        "migrated_records": len(records),
        "missing_expected_ids": missing_expected_ids,
        "unexpected_ids": unexpected_ids,
        "paper_ids_sha256": _paper_ids_checksum(migrated_ids),
        "partial_records": len(backfill_ids),
        "backfill_required": len(backfill_ids),
        "parse_failures": parse_failures,
        "recovered_anomalies": recovered_anomalies,
        "field_conflicts": field_conflicts,
        "schema_issues": schema_issues,
        "valid": not any((
            invalid_topics,
            missing_expected_ids,
            unexpected_ids,
            schema_issues,
        )),
    }
    return records, report, backfill_ids


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="\n")


def _write_ids(path: Path, paper_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{paper_id}\n" for paper_id in paper_ids),
        encoding="utf-8",
        newline="\n",
    )


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate the legacy topic-row archive to canonical paper records."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("docs/asr-arxiv-daily.json"),
    )
    parser.add_argument(
        "--baseline-ids",
        type=Path,
        default=Path("migration/baseline-paper-ids.txt"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backfill-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.output is not None and _same_path(args.archive, args.output):
        parser.error("canonical output must not overwrite the legacy archive")

    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    if not isinstance(archive, dict):
        raise ValueError("Legacy archive root must be a JSON object")
    expected_ids = {
        line.strip()
        for line in args.baseline_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    records, report, backfill_ids = migrate_archive(archive, expected_ids)

    if args.report:
        _write_json(args.report, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.backfill_output:
        _write_ids(args.backfill_output, backfill_ids)
    if not args.dry_run and args.output is not None and report["valid"]:
        _write_json(args.output, records)

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())