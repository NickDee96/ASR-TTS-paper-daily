import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_EXPLORER_URL = "https://nickdee96.github.io/ASR-TTS-paper-daily/"
DEFAULT_RECENT_LIMIT = 50
DEFAULT_STALE_AFTER_HOURS = 36


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def _parse_timestamp(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _markdown_cell(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _paper_row(paper: dict[str, Any]) -> str:
    title = _markdown_cell(paper.get("title") or "Untitled paper")
    links = paper.get("links") if isinstance(paper.get("links"), dict) else {}
    abstract_url = links.get("abstract")
    title_cell = f"[{title}]({abstract_url})" if abstract_url else title
    authors = [
        _markdown_cell(author)
        for author in paper.get("authors", [])
        if isinstance(author, str) and author.strip()
    ]
    author_cell = ", ".join(authors[:3])
    if len(authors) > 3:
        author_cell += " et al."
    topics = ", ".join(
        _markdown_cell(topic)
        for topic in paper.get("topics", [])
        if isinstance(topic, str) and topic.strip()
    )
    code_url = paper.get("code_url")
    code_status = paper.get("code_status")
    code = f"[Verified code]({code_url})" if code_status == "verified" and code_url else "-"
    date_value = paper.get("updated") or paper.get("published") or "Unknown"
    return f"| {_markdown_cell(date_value)} | {title_cell} | {author_cell or '-'} | {topics or '-'} | {code} |"


def _collection_message(
    latest: dict[str, Any],
    site_card: dict[str, Any],
    run_status: dict[str, Any] | None,
    now: datetime.datetime,
    stale_after_hours: int,
) -> str:
    papers = latest.get("papers")
    if not isinstance(papers, list) or not papers:
        return "> Collection status: No papers are currently available. The next successful collection will repopulate this digest."

    state = run_status.get("state") if run_status else None
    if state == "failed":
        return "> Collection status: The latest collection attempt failed. Data from the last successful update remains available."

    updated_at = site_card.get("updated_at") or latest.get("updated_at")
    if not isinstance(updated_at, str):
        return "> Collection status: Update time is unavailable. The archive may be stale."
    try:
        age = now.astimezone(datetime.timezone.utc) - _parse_timestamp(updated_at)
    except ValueError:
        return "> Collection status: Update time is invalid. The archive may be stale."
    if age > datetime.timedelta(hours=stale_after_hours):
        return f"> Collection status: Delayed. The last successful update was {updated_at}."
    return f"> Collection status: Current as of {updated_at}."


def render_readme(
    latest: dict[str, Any],
    site_card: dict[str, Any],
    *,
    run_status: dict[str, Any] | None = None,
    now: datetime.datetime | None = None,
    explorer_url: str = DEFAULT_EXPLORER_URL,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    stale_after_hours: int = DEFAULT_STALE_AFTER_HOURS,
) -> str:
    if recent_limit < 1 or recent_limit > DEFAULT_RECENT_LIMIT:
        raise ValueError(f"recent_limit must be between 1 and {DEFAULT_RECENT_LIMIT}")
    if stale_after_hours < 1:
        raise ValueError("stale_after_hours must be at least 1")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    papers = latest.get("papers")
    if not isinstance(papers, list):
        raise ValueError("latest.papers must be an array")
    topics = site_card.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("site_card.topics must be an object")

    lines = [
        "# ASR-TTS Paper Daily",
        "",
        "A daily-updated collection of speech and language research from arXiv.",
        "",
        f"[Open the searchable paper explorer]({explorer_url})",
        "",
        _collection_message(latest, site_card, run_status, now, stale_after_hours),
        "",
        "## Archive",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Unique papers | {int(site_card.get('unique_papers', 0)):,} |",
        f"| Topic assignments | {int(site_card.get('topic_assignments', 0)):,} |",
        f"| Papers with verified code | {int(site_card.get('verified_code', 0)):,} |",
        "",
        "## Topics",
        "",
    ]
    if topics:
        lines.extend(
            f"- **{_markdown_cell(topic)}:** {int(count):,} papers"
            for topic, count in sorted(topics.items())
        )
    else:
        lines.append("No topic statistics are currently available.")

    lines.extend([
        "",
        "## Recent Papers",
        "",
    ])
    if papers:
        lines.extend([
            f"Showing the {min(len(papers), recent_limit)} most recent papers. Use the [paper explorer]({explorer_url}) to search and filter the full archive.",
            "",
            "| Updated | Paper | Authors | Topics | Code |",
            "| --- | --- | --- | --- | --- |",
        ])
        lines.extend(_paper_row(paper) for paper in papers[:recent_limit])
    else:
        lines.append("No recent papers are available yet.")

    lines.extend([
        "",
        "## Contributing",
        "",
        "Issues and pull requests for topic rules, metadata corrections, and product improvements are welcome.",
        "",
        "This README is generated from the canonical archive. Do not edit its paper list manually.",
        "",
    ])
    rendered = "\n".join(lines)
    if len(rendered.encode("utf-8")) >= 100_000:
        raise ValueError("Rendered README exceeds the 100 KB size budget")
    return rendered


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a bounded repository README")
    parser.add_argument("--latest", type=Path, default=Path("site/public/data/latest.json"))
    parser.add_argument("--site-card", type=Path, default=Path("site/public/data/site-card.json"))
    parser.add_argument("--status", type=Path)
    parser.add_argument("--output", type=Path, default=Path("README.md"))
    parser.add_argument("--explorer-url", default=DEFAULT_EXPLORER_URL)
    parser.add_argument("--recent-limit", type=int, default=DEFAULT_RECENT_LIMIT)
    parser.add_argument("--stale-after-hours", type=int, default=DEFAULT_STALE_AFTER_HOURS)
    args = parser.parse_args()

    run_status = _load_json(args.status) if args.status and args.status.exists() else None
    rendered = render_readme(
        _load_json(args.latest),
        _load_json(args.site_card),
        run_status=run_status,
        explorer_url=args.explorer_url,
        recent_limit=args.recent_limit,
        stale_after_hours=args.stale_after_hours,
    )
    atomic_write_text(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())