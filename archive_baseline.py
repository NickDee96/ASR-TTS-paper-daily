import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARXIV_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})$", re.IGNORECASE)
LEGACY_ROW_PATTERN = re.compile(
    r"^\|\*\*(?P<date>\d{4}-\d{2}-\d{2})\*\*"
    r"\|\*\*(?P<title>.*?)\*\*"
    r"\|(?P<authors>.*?)"
    r"\|\[(?P<paper_label>[^]]+)]\((?P<paper_url>[^)]+)\)"
    r"(?:\|(?P<code>.*?))?\|\s*$",
    re.DOTALL,
)
CODE_LINK_PATTERN = re.compile(r"\[link]\((?P<url>https?://[^)]+)\)", re.IGNORECASE)


def parse_legacy_row(row: str) -> dict[str, str] | None:
    match = LEGACY_ROW_PATTERN.match(row)
    if match is None:
        return None

    parsed = {
        key: (value or "").strip()
        for key, value in match.groupdict().items()
    }
    try:
        datetime.date.fromisoformat(parsed["date"])
    except ValueError:
        return None
    return parsed


def _file_sizes(paths: list[Path]) -> dict[str, int | None]:
    return {
        path.as_posix(): path.stat().st_size if path.is_file() else None
        for path in sorted(paths, key=lambda value: value.as_posix())
    }


def analyze_archive(archive_path: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    with archive_path.open("r", encoding="utf-8") as archive_file:
        archive = json.load(archive_file)

    if not isinstance(archive, dict):
        raise ValueError("Legacy archive root must be a JSON object")

    paper_topics: dict[str, set[str]] = {}
    topic_counts: dict[str, dict[str, int]] = {}
    malformed_records: list[dict[str, str]] = []
    dates: list[datetime.date] = []
    topic_assignments = 0
    assignments_with_code = 0

    for topic in sorted(archive):
        papers = archive[topic]
        if not isinstance(papers, dict):
            malformed_records.append({
                "topic": topic,
                "arxiv_id": "",
                "reason": "topic value is not an object",
            })
            continue

        topic_total = 0
        topic_with_code = 0
        for arxiv_id in sorted(papers):
            row = papers[arxiv_id]
            topic_assignments += 1
            topic_total += 1
            paper_topics.setdefault(arxiv_id, set()).add(topic)

            if ARXIV_ID_PATTERN.fullmatch(arxiv_id) is None:
                malformed_records.append({
                    "topic": topic,
                    "arxiv_id": arxiv_id,
                    "reason": "invalid arXiv ID",
                })

            if not isinstance(row, str):
                malformed_records.append({
                    "topic": topic,
                    "arxiv_id": arxiv_id,
                    "reason": "row is not a string",
                })
                continue

            parsed = parse_legacy_row(row)
            if parsed is None:
                malformed_records.append({
                    "topic": topic,
                    "arxiv_id": arxiv_id,
                    "reason": "row does not match the legacy table format",
                })
                continue

            if not parsed["code"]:
                malformed_records.append({
                    "topic": topic,
                    "arxiv_id": arxiv_id,
                    "reason": "row is missing the code column",
                })

            dates.append(datetime.date.fromisoformat(parsed["date"]))
            if CODE_LINK_PATTERN.search(parsed["code"]):
                assignments_with_code += 1
                topic_with_code += 1

        topic_counts[topic] = {
            "assignments": topic_total,
            "with_code": topic_with_code,
        }

    sorted_ids = sorted(paper_topics)
    id_payload = "".join(f"{arxiv_id}\n" for arxiv_id in sorted_ids).encode("utf-8")
    multi_topic_papers = sum(1 for topics in paper_topics.values() if len(topics) > 1)

    return {
        "report_version": 1,
        "archive": archive_path.as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "unique_papers": len(sorted_ids),
        "topic_assignments": topic_assignments,
        "duplicate_assignments": topic_assignments - len(sorted_ids),
        "multi_topic_papers": multi_topic_papers,
        "assignments_with_code": assignments_with_code,
        "code_coverage_percent": round(
            assignments_with_code * 100 / topic_assignments, 2
        ) if topic_assignments else 0.0,
        "oldest_row_date": min(dates).isoformat() if dates else None,
        "newest_row_date": max(dates).isoformat() if dates else None,
        "paper_ids_sha256": hashlib.sha256(id_payload).hexdigest(),
        "topics": topic_counts,
        "malformed_record_count": len(malformed_records),
        "malformed_records": malformed_records,
        "artifact_sizes": _file_sizes(artifact_paths),
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report deterministic metrics for the legacy paper archive."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("docs/asr-arxiv-daily.json"),
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        type=Path,
        default=None,
        help="Artifact whose byte size should be reported; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ids-output",
        type=Path,
        help="Write sorted unique arXiv IDs for migration reconciliation.",
    )
    parser.add_argument("--fail-on-malformed", action="store_true")
    args = parser.parse_args()

    artifacts = args.artifacts or [
        Path("README.md"),
        Path("docs/index.md"),
        args.archive,
    ]
    report = analyze_archive(args.archive, artifacts)
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.output:
        _write_text(args.output, rendered_report)
    else:
        print(rendered_report, end="")

    if args.ids_output:
        ids = sorted({
            arxiv_id
            for topic in json.loads(args.archive.read_text(encoding="utf-8")).values()
            for arxiv_id in topic
        })
        _write_text(args.ids_output, "".join(f"{arxiv_id}\n" for arxiv_id in ids))

    if args.fail_on_malformed and report["malformed_record_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())