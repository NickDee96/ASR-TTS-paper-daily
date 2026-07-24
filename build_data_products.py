import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from paper_collector import _format_timestamp, _parse_timestamp
from paper_store import atomic_write_json, load_canonical_archive, validate_canonical_archive


ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def _feed_timestamp(value: str | None, fallback: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T00:00:00Z"
    if isinstance(value, str) and value:
        try:
            return _format_timestamp(_parse_timestamp(value))
        except ValueError:
            return fallback
    return fallback


def _site_href(site_url: str, base_path: str, *segments: str) -> str:
    prefix = site_url.rstrip("/")
    parts = [base_path.strip("/"), *(segment.strip("/") for segment in segments)]
    tail = "/".join(part for part in parts if part)
    if prefix:
        return f"{prefix}/{tail}/" if tail else f"{prefix}/"
    return f"/{tail}/" if tail else "/"


def _append_feed_entry(
    feed: ET.Element,
    record: dict[str, Any],
    site_url: str,
    base_path: str,
    fallback_updated: str,
) -> None:
    namespace = f"{{{ATOM_NAMESPACE}}}"
    entry = ET.SubElement(feed, f"{namespace}entry")
    paper_id = record["id"]
    ET.SubElement(entry, f"{namespace}title").text = record.get("title") or f"arXiv:{paper_id}"
    ET.SubElement(entry, f"{namespace}id").text = f"https://arxiv.org/abs/{paper_id}"
    link = ET.SubElement(entry, f"{namespace}link")
    link.set("rel", "alternate")
    link.set("href", _site_href(site_url, base_path, "papers", paper_id))
    published = _feed_timestamp(record.get("published"), fallback_updated)
    updated = _feed_timestamp(record.get("updated") or record.get("published"), published)
    ET.SubElement(entry, f"{namespace}updated").text = updated
    ET.SubElement(entry, f"{namespace}published").text = published
    for author in record.get("authors", []):
        name = author.get("name") if isinstance(author, dict) else None
        if name:
            author_element = ET.SubElement(entry, f"{namespace}author")
            ET.SubElement(author_element, f"{namespace}name").text = name
    for topic in record.get("topics", []):
        ET.SubElement(entry, f"{namespace}category").set("term", topic)
    abstract = record.get("abstract")
    if abstract:
        ET.SubElement(entry, f"{namespace}summary").text = abstract


def _render_atom_feed(
    *,
    title: str,
    subtitle: str,
    self_url: str,
    home_url: str,
    updated: str,
    records: list[dict[str, Any]],
    site_url: str,
    base_path: str,
    feed_limit: int,
) -> str:
    namespace = f"{{{ATOM_NAMESPACE}}}"
    ET.register_namespace("", ATOM_NAMESPACE)
    feed = ET.Element(f"{namespace}feed")
    ET.SubElement(feed, f"{namespace}title").text = title
    ET.SubElement(feed, f"{namespace}subtitle").text = subtitle
    ET.SubElement(feed, f"{namespace}id").text = self_url
    self_link = ET.SubElement(feed, f"{namespace}link")
    self_link.set("rel", "self")
    self_link.set("href", self_url)
    home_link = ET.SubElement(feed, f"{namespace}link")
    home_link.set("rel", "alternate")
    home_link.set("href", home_url)
    ET.SubElement(feed, f"{namespace}updated").text = updated
    ET.SubElement(feed, f"{namespace}generator").text = "build_data_products"
    for record in sorted(records, key=_record_sort_key, reverse=True)[:feed_limit]:
        _append_feed_entry(feed, record, site_url, base_path, updated)
    ET.indent(feed, space="  ")
    body = ET.tostring(feed, encoding="unicode")
    return f'<?xml version="1.0" encoding="utf-8"?>\n{body}\n'


def build_data_products(
    records: dict[str, dict[str, Any]],
    output_root: Path,
    updated_at: datetime.datetime,
    latest_limit: int = 100,
    clean: bool = False,
    *,
    site_url: str = "",
    base_path: str = "/",
    feed_topics: list[str] | None = None,
    feed_limit: int = 50,
) -> dict[str, Any]:
    if latest_limit < 1:
        raise ValueError("latest_limit must be at least 1")
    if feed_limit < 1:
        raise ValueError("feed_limit must be at least 1")
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

    record_values = list(records.values())
    home_url = _site_href(site_url, base_path)
    feeds_base = _site_href(site_url, base_path, "data", "feeds").rstrip("/")
    expected_feed_paths: set[Path] = set()

    def _write_feed(
        relative_name: str,
        title: str,
        subtitle: str,
        feed_records: list[dict[str, Any]],
    ) -> str:
        relative = Path("feeds") / relative_name
        feed_xml = _render_atom_feed(
            title=title,
            subtitle=subtitle,
            self_url=f"{feeds_base}/{relative_name}",
            home_url=home_url,
            updated=updated_timestamp,
            records=feed_records,
            site_url=site_url,
            base_path=base_path,
            feed_limit=feed_limit,
        )
        target = output_root / relative
        expected_feed_paths.add(target.resolve())
        _atomic_write_text(target, feed_xml)
        return relative.as_posix()

    available_topics = sorted(
        {topic for record in record_values for topic in record.get("topics", [])}
    )
    configured_topics = available_topics if feed_topics is None else list(feed_topics)
    feed_products: dict[str, Any] = {
        "all": _write_feed(
            "all.xml",
            "ASR-TTS Paper Daily",
            "Newest speech and language research across all topics.",
            record_values,
        ),
        "topics": {},
    }
    used_slugs: dict[str, str] = {}
    for topic in configured_topics:
        slug = _slugify_topic(topic)
        if slug in used_slugs:
            raise ValueError(
                f"Topic feed slug collision between {used_slugs[slug]!r} and {topic!r}"
            )
        used_slugs[slug] = topic
        topic_records = [
            record for record in record_values if topic in record.get("topics", [])
        ]
        feed_products["topics"][topic] = _write_feed(
            f"topic-{slug}.xml",
            f"ASR-TTS Paper Daily \u2014 {topic}",
            f"Newest papers classified under {topic}.",
            topic_records,
        )

    if clean:
        feeds_root = output_root / "feeds"
        if feeds_root.exists():
            for path in feeds_root.glob("*.xml"):
                if path.resolve() not in expected_feed_paths:
                    path.unlink()

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
        "feeds": feed_products,
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
    user_name = config.get("user_name") or ""
    repo_name = config.get("repo_name") or ""
    records = load_canonical_archive(args.input)
    manifest = build_data_products(
        records=records,
        output_root=args.output or Path(product_config["output_path"]),
        updated_at=_parse_timestamp(args.updated_at),
        latest_limit=args.latest_limit or int(product_config.get("latest_limit", 100)),
        clean=args.clean,
        site_url=f"https://{user_name}.github.io" if user_name else "",
        base_path=f"/{repo_name}" if repo_name else "/",
        feed_topics=list(config.get("keywords", {}).keys()) or None,
        feed_limit=int(product_config.get("feed_limit", 50)),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())