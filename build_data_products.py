import argparse
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paper_collector import _format_timestamp, _parse_timestamp
from paper_store import atomic_write_json, load_canonical_archive, validate_canonical_archive


def _render_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_render_json(value)).hexdigest()


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record.get("updated") or "0000-00-00",
        record.get("published") or "0000-00-00",
        record["id"],
    )


def _card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "title": record.get("title"),
        "abstract": record.get("abstract"),
        "authors": [
            author["name"]
            for author in record.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ],
        "published": record.get("published"),
        "updated": record.get("updated"),
        "topics": record.get("topics", []),
        "primary_category": record.get("primary_category"),
        "record_status": record.get("record_status"),
        "code_status": record.get("code", {}).get("status"),
        "code_url": record.get("code", {}).get("url"),
        "links": record.get("links", {}),
    }


def _facet_counts(
    records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    topics: Counter[str] = Counter()
    years: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    code_statuses: Counter[str] = Counter()
    record_statuses: Counter[str] = Counter()
    for record in records.values():
        topics.update(record.get("topics", []))
        published = record.get("published")
        years.update([published[:4] if isinstance(published, str) else "Unknown"])
        categories.update(record.get("arxiv_categories", []))
        code_statuses.update([record.get("code", {}).get("status", "missing")])
        record_statuses.update([record.get("record_status", "partial")])
    return {
        "topics": dict(sorted(topics.items())),
        "years": dict(sorted(years.items(), reverse=True)),
        "categories": dict(sorted(categories.items())),
        "code_status": dict(sorted(code_statuses.items())),
        "record_status": dict(sorted(record_statuses.items())),
    }


def _statistics(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    monthly: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    topic_assignments = 0
    verified_code = 0
    for record in records.values():
        published = record.get("published")
        monthly.update([published[:7] if isinstance(published, str) else "Unknown"])
        categories.update(record.get("arxiv_categories", []))
        topic_assignments += len(record.get("topics", []))
        verified_code += record.get("code", {}).get("status") == "verified"
    unique_papers = len(records)
    return {
        "unique_papers": unique_papers,
        "topic_assignments": topic_assignments,
        "monthly_additions": dict(sorted(monthly.items())),
        "category_distribution": dict(sorted(categories.items())),
        "verified_code": verified_code,
        "verified_code_coverage_percent": round(
            verified_code * 100 / unique_papers, 2
        ) if unique_papers else 0.0,
    }


def _partition_key(record: dict[str, Any]) -> str:
    published = record.get("published")
    if not isinstance(published, str):
        return "undated"
    published_date = datetime.date.fromisoformat(published)
    return f"{published_date.year:04d}/{published_date.month:02d}"


def build_data_products(
    records: dict[str, dict[str, Any]],
    output_root: Path,
    updated_at: datetime.datetime,
    latest_limit: int = 100,
    clean: bool = False,
) -> dict[str, Any]:
    if latest_limit < 1:
        raise ValueError("latest_limit must be at least 1")
    validate_canonical_archive(records)
    updated_timestamp = _format_timestamp(updated_at)
    partitions: dict[str, list[dict[str, Any]]] = {}
    for record in records.values():
        partitions.setdefault(_partition_key(record), []).append(record)

    shard_entries: list[dict[str, Any]] = []
    expected_shard_paths: set[Path] = set()
    for partition in sorted(partitions, reverse=True):
        papers = sorted(partitions[partition], key=_record_sort_key, reverse=True)
        relative_path = (
            Path("papers") / "undated.json"
            if partition == "undated"
            else Path("papers") / f"{partition}.json"
        )
        shard = {
            "schema_version": 1,
            "papers": papers,
        }
        target = output_root / relative_path
        expected_shard_paths.add(target.resolve())
        atomic_write_json(target, shard)
        dated = [record["published"] for record in papers if record.get("published")]
        shard_entries.append({
            "path": relative_path.as_posix(),
            "sha256": _sha256(shard),
            "count": len(papers),
            "date_start": min(dated) if dated else None,
            "date_end": max(dated) if dated else None,
        })

    if clean:
        papers_root = output_root / "papers"
        if papers_root.exists():
            for path in papers_root.rglob("*.json"):
                if path.resolve() not in expected_shard_paths:
                    path.unlink()

    ordered_records = sorted(records.values(), key=_record_sort_key, reverse=True)
    latest = {
        "schema_version": 1,
        "updated_at": updated_timestamp,
        "limit": latest_limit,
        "papers": [_card(record) for record in ordered_records[:latest_limit]],
    }
    facets = {
        "schema_version": 1,
        "updated_at": updated_timestamp,
        "facets": _facet_counts(records),
    }
    statistics = {
        "schema_version": 1,
        "updated_at": updated_timestamp,
        **_statistics(records),
    }
    site_card = {
        "schema_version": 1,
        "updated_at": updated_timestamp,
        "unique_papers": statistics["unique_papers"],
        "topic_assignments": statistics["topic_assignments"],
        "verified_code": statistics["verified_code"],
        "topics": facets["facets"]["topics"],
    }
    atomic_write_json(output_root / "latest.json", latest)
    atomic_write_json(output_root / "facets.json", facets)
    atomic_write_json(output_root / "statistics.json", statistics)
    atomic_write_json(output_root / "site-card.json", site_card)

    manifest = {
        "manifest_version": 1,
        "paper_schema_version": 1,
        "updated_at": updated_timestamp,
        "unique_papers": len(records),
        "date_start": min(
            (record["published"] for record in records.values() if record.get("published")),
            default=None,
        ),
        "date_end": max(
            (record["published"] for record in records.values() if record.get("published")),
            default=None,
        ),
        "shards": sorted(shard_entries, key=lambda shard: shard["path"], reverse=True),
        "products": {
            "latest": "latest.json",
            "facets": "facets.json",
            "statistics": "statistics.json",
            "site_card": "site-card.json",
        },
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build partitioned static paper data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--updated-at", required=True)
    parser.add_argument("--latest-limit", type=int)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    product_config = config.get("data_products", {})
    records = load_canonical_archive(args.input)
    manifest = build_data_products(
        records=records,
        output_root=args.output or Path(product_config["output_path"]),
        updated_at=_parse_timestamp(args.updated_at),
        latest_limit=args.latest_limit or int(product_config.get("latest_limit", 100)),
        clean=args.clean,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())