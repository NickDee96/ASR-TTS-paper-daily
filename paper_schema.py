import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker


DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schemas" / "paper.schema.json"


def load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    Draft202012Validator.check_schema(schema)
    return schema


def _json_path(parts: Iterable[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _semantic_issues(paper: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    topics = paper.get("topics")
    classification = paper.get("classification")
    if isinstance(topics, list) and isinstance(classification, dict):
        matches = classification.get("matches")
        if isinstance(matches, list):
            match_topics = [
                match.get("topic")
                for match in matches
                if isinstance(match, dict) and isinstance(match.get("topic"), str)
            ]
            if sorted(topics) != sorted(match_topics):
                issues.append({
                    "path": "$.classification.matches",
                    "validator": "topic_membership",
                    "message": "classification match topics must equal topics",
                })

    published = paper.get("published")
    updated = paper.get("updated")
    if isinstance(published, str) and isinstance(updated, str):
        try:
            if datetime.date.fromisoformat(updated) < datetime.date.fromisoformat(published):
                issues.append({
                    "path": "$.updated",
                    "validator": "date_order",
                    "message": "updated date must not be earlier than published date",
                })
        except ValueError:
            pass

    return issues


def validate_paper(
    paper: Any,
    schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    active_schema = schema or load_schema()
    validator = Draft202012Validator(active_schema, format_checker=FormatChecker())
    issues = [
        {
            "path": _json_path(error.absolute_path),
            "validator": str(error.validator),
            "message": error.message,
        }
        for error in validator.iter_errors(paper)
    ]
    if isinstance(paper, dict):
        issues.extend(_semantic_issues(paper))
    return sorted(
        issues,
        key=lambda issue: (issue["path"], issue["validator"], issue["message"]),
    )


def records_from_document(document: Any) -> list[tuple[str, Any]]:
    if isinstance(document, list):
        return [(f"[{index}]", record) for index, record in enumerate(document)]
    if isinstance(document, dict) and "id" in document:
        return [(str(document.get("id", "$")), document)]
    if isinstance(document, dict) and isinstance(document.get("papers"), list):
        return [
            (f"papers[{index}]", record)
            for index, record in enumerate(document["papers"])
        ]
    if isinstance(document, dict):
        return [(str(record_id), record) for record_id, record in sorted(document.items())]
    raise ValueError("JSON document must be a paper, paper array, paper map, or object with papers")


def validate_files(
    paths: list[Path],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    schema = load_schema(schema_path)
    files: list[dict[str, Any]] = []
    valid_records = 0
    invalid_records = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as input_file:
            records = records_from_document(json.load(input_file))

        record_reports = []
        for record_id, record in records:
            issues = validate_paper(record, schema)
            if issues:
                invalid_records += 1
            else:
                valid_records += 1
            record_reports.append({
                "record": record_id,
                "valid": not issues,
                "issues": issues,
            })

        files.append({
            "path": path.as_posix(),
            "records": record_reports,
        })

    return {
        "report_version": 1,
        "schema": schema_path.as_posix(),
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical paper JSON records.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_files(args.paths, args.schema)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    return 1 if report["invalid_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())