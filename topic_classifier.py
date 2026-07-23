import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from paper_store import atomic_write_json, write_canonical_archive


CLASSIFIER_VERSION = "keywords-v2"


class TopicConfigError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = sorted(issues)
        super().__init__("Invalid topic configuration: " + "; ".join(self.issues))


def normalize_term(term: str) -> str:
    normalized = unicodedata.normalize("NFKC", term)
    return " ".join(normalized.casefold().split())


def _term_pattern(term: str) -> re.Pattern[str]:
    normalized = normalize_term(term)
    if len(normalized.split()) > 1 or re.search(r"[^\w]", normalized):
        return re.compile(re.escape(normalized), re.IGNORECASE)
    return re.compile(rf"\b{re.escape(normalized)}\b", re.IGNORECASE)


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    normalized_text = unicodedata.normalize("NFKC", text or "")
    return sorted({term for term in terms if term and _term_pattern(term).search(normalized_text)})


def _rule_terms(rules: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    include = rules.get("include") if isinstance(rules.get("include"), dict) else {}
    include_any = include.get("any") if isinstance(include.get("any"), list) else []
    include_all = include.get("all") if isinstance(include.get("all"), list) else []
    if not include_any:
        include_any = rules.get("filters") if isinstance(rules.get("filters"), list) else []
    exclude = rules.get("exclude") if isinstance(rules.get("exclude"), list) else []
    return (
        [normalize_term(term) for term in include_any],
        [normalize_term(term) for term in include_all],
        [normalize_term(term) for term in exclude],
    )


def validate_topic_config(topic_rules: dict[str, Any]) -> None:
    issues: list[str] = []
    if not isinstance(topic_rules, dict) or not topic_rules:
        raise TopicConfigError(["keywords must be a non-empty object"])

    for topic, rules in sorted(topic_rules.items()):
        if not isinstance(rules, dict):
            issues.append(f"{topic}: rules must be an object")
            continue
        filters = rules.get("filters")
        if not isinstance(filters, list) or not filters:
            issues.append(f"{topic}: filters must be a non-empty list")

        lists = {
            "filters": filters if isinstance(filters, list) else [],
            "include.any": (
                rules.get("include", {}).get("any", [])
                if isinstance(rules.get("include"), dict)
                else []
            ),
            "include.all": (
                rules.get("include", {}).get("all", [])
                if isinstance(rules.get("include"), dict)
                else []
            ),
            "exclude": rules.get("exclude", []),
        }
        normalized_lists: dict[str, list[str]] = {}
        for name, values in lists.items():
            if not isinstance(values, list):
                issues.append(f"{topic}: {name} must be a list")
                continue
            if any(not isinstance(value, str) or not value.strip() for value in values):
                issues.append(f"{topic}: {name} contains an empty or non-string term")
                continue
            normalized = [normalize_term(value) for value in values]
            normalized_lists[name] = normalized
            duplicates = sorted(term for term in set(normalized) if normalized.count(term) > 1)
            if duplicates:
                issues.append(f"{topic}: {name} has duplicate terms {duplicates}")

        active_include = normalized_lists.get("include.any") or normalized_lists.get("filters", [])
        active_include += normalized_lists.get("include.all", [])
        contradictions = sorted(
            set(active_include) & set(normalized_lists.get("exclude", []))
        )
        if contradictions:
            issues.append(f"{topic}: include/exclude contradictions {contradictions}")

        min_score = rules.get("min_score", 1)
        title_weight = rules.get("title_weight", 2)
        if not isinstance(min_score, (int, float)) or min_score < 0:
            issues.append(f"{topic}: min_score must be non-negative")
        if not isinstance(title_weight, (int, float)) or title_weight < 0:
            issues.append(f"{topic}: title_weight must be non-negative")

    if issues:
        raise TopicConfigError(issues)


def classify_topic(
    title: str,
    abstract: str,
    topic: str,
    rules: dict[str, Any] | None,
) -> dict[str, Any]:
    if rules is None:
        return {
            "topic": topic,
            "accepted": True,
            "reason": "accepted",
            "score": 0,
            "threshold": 0,
            "matched_title_terms": [],
            "matched_abstract_terms": [],
            "matched_all_terms": [],
            "exclusion_hits": [],
            "evidence_complete": False,
        }

    include_any, include_all, exclude = _rule_terms(rules)
    title_matches = _matched_terms(title, include_any)
    abstract_matches = _matched_terms(abstract, include_any)
    matched_all = sorted({
        term
        for term in include_all
        if _term_pattern(term).search(unicodedata.normalize("NFKC", title or ""))
        or _term_pattern(term).search(unicodedata.normalize("NFKC", abstract or ""))
    })
    exclusion_hits = sorted(set(
        _matched_terms(title, exclude) + _matched_terms(abstract, exclude)
    ))
    title_weight = rules.get("title_weight", 2)
    threshold = rules.get("min_score")
    if threshold is None:
        threshold = 1 if len(include_any) <= 4 else 2
    score = title_weight * len(title_matches) + len(abstract_matches)

    if exclusion_hits:
        accepted = False
        reason = "excluded"
    elif len(matched_all) != len(set(include_all)):
        accepted = False
        reason = "missing_required_term"
    elif score < threshold:
        accepted = False
        reason = "below_threshold"
    else:
        accepted = True
        reason = "accepted"

    return {
        "topic": topic,
        "accepted": accepted,
        "reason": reason,
        "score": score,
        "threshold": threshold,
        "matched_title_terms": title_matches,
        "matched_abstract_terms": abstract_matches,
        "matched_all_terms": matched_all,
        "exclusion_hits": exclusion_hits,
        "evidence_complete": True,
    }


def classify_paper(
    title: str,
    abstract: str,
    topic_rules: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    decisions = [
        classify_topic(title, abstract, topic, rules)
        for topic, rules in sorted(topic_rules.items())
    ]
    matches = [decision for decision in decisions if decision["accepted"]]
    topics = [decision["topic"] for decision in matches]
    return topics, {
        "classifier_version": CLASSIFIER_VERSION,
        "matches": matches,
    }, decisions


def is_relevant_for_topic(
    title: str,
    abstract: str,
    topic_rules: dict[str, Any] | None,
) -> tuple[bool, int, dict[str, Any]]:
    decision = classify_topic(title, abstract, "configured-topic", topic_rules)
    details = {
        **decision,
        "title_hits": len(decision["matched_title_terms"]),
        "abs_hits": len(decision["matched_abstract_terms"]),
        "min_score": decision["threshold"],
        "all_ok": decision["reason"] != "missing_required_term",
    }
    return decision["accepted"], decision["score"], details


def reclassify_records(
    records: dict[str, dict[str, Any]],
    topic_rules: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validate_topic_config(topic_rules)
    updated_records = json.loads(json.dumps(records))
    changes: list[dict[str, Any]] = []
    added_assignments = 0
    removed_assignments = 0

    for paper_id in sorted(updated_records):
        record = updated_records[paper_id]
        topics, classification, _ = classify_paper(
            record.get("title") or "",
            record.get("abstract") or "",
            topic_rules,
        )
        previous_topics = set(record.get("topics", []))
        next_topics = set(topics)
        added = sorted(next_topics - previous_topics)
        removed = sorted(previous_topics - next_topics)
        if added or removed:
            changes.append({"id": paper_id, "added": added, "removed": removed})
            added_assignments += len(added)
            removed_assignments += len(removed)
        record["topics"] = topics
        record["classification"] = classification

    report = {
        "report_version": 1,
        "classifier_version": CLASSIFIER_VERSION,
        "evaluated_papers": len(updated_records),
        "changed_papers": len(changes),
        "unchanged_papers": len(updated_records) - len(changes),
        "added_assignments": added_assignments,
        "removed_assignments": removed_assignments,
        "changes": changes,
    }
    return updated_records, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit or apply topic classification.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.apply and args.output is None:
        parser.error("--output is required with --apply")

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = json.loads(args.input.read_text(encoding="utf-8"))
    updated_records, report = reclassify_records(records, config["keywords"])
    atomic_write_json(args.report, report)
    if args.apply and args.output is not None:
        write_canonical_archive(args.output, updated_records)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())