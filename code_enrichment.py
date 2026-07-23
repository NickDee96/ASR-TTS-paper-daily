import argparse
import datetime
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import requests

from paper_collector import UTC, _format_timestamp, _parse_timestamp
from paper_schema import load_schema, validate_paper
from paper_store import atomic_write_json, load_canonical_archive, write_canonical_archive


@dataclass(frozen=True)
class CodeCandidate:
    url: str
    source: str
    evidence: tuple[str, ...]
    authoritative: bool = False
    exact_arxiv_id: bool = False
    available: bool = True


class CodeLookupSource(Protocol):
    name: str

    def lookup(self, paper: dict[str, Any]) -> Iterable[CodeCandidate]: ...


class PapersWithCodeLookup:
    name = "papers-with-code"

    def __init__(
        self,
        base_url: str = "https://arxiv.paperswithcode.com/api/v0/papers/",
        timeout_seconds: float = 10,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def lookup(self, paper: dict[str, Any]) -> Iterable[CodeCandidate]:
        response = self.session.get(
            self.base_url + paper["id"],
            timeout=self.timeout_seconds,
            headers={"User-Agent": "asr-tts-paper-daily-bot"},
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        official = payload.get("official") if isinstance(payload, dict) else None
        url = official.get("url") if isinstance(official, dict) else None
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            return []
        return [CodeCandidate(
            url=url,
            source=self.name,
            authoritative=True,
            exact_arxiv_id=True,
            evidence=("authoritative paper-to-code mapping for exact arXiv ID",),
        )]


class GitHubRepositoryLookup:
    name = "github-repository-search"

    def __init__(
        self,
        timeout_seconds: float = 10,
        token: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self.session = session or requests.Session()

    def _search(self, query: str) -> list[dict[str, Any]]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "asr-tts-paper-daily-bot",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self.session.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("items", []) if isinstance(payload, dict) else []

    def lookup(self, paper: dict[str, Any]) -> Iterable[CodeCandidate]:
        paper_id = paper["id"]
        items = self._search(paper_id)
        used_title_fallback = False
        if not items and paper.get("title"):
            items = self._search(str(paper["title"]))
            used_title_fallback = True

        candidates: list[CodeCandidate] = []
        id_pattern = re.compile(rf"(?<!\d){re.escape(paper_id)}(?!\d)", re.IGNORECASE)
        for item in items:
            if not isinstance(item, dict) or item.get("archived"):
                continue
            url = item.get("html_url")
            if not isinstance(url, str) or not url.startswith("https://github.com/"):
                continue
            metadata = " ".join(
                str(item.get(field) or "")
                for field in ("name", "full_name", "description", "homepage")
            )
            exact_id = id_pattern.search(metadata) is not None
            evidence = (
                "exact arXiv ID appears in GitHub repository metadata"
                if exact_id
                else "title-only GitHub repository search result"
            )
            if used_title_fallback and exact_id:
                evidence += " after title fallback"
            candidates.append(CodeCandidate(
                url=url,
                source=self.name,
                exact_arxiv_id=exact_id,
                evidence=(evidence,),
            ))
        return candidates


def _candidate_from_json(value: dict[str, Any]) -> CodeCandidate:
    return CodeCandidate(
        url=value["url"],
        source=value["source"],
        evidence=tuple(value.get("evidence", [])),
        authoritative=bool(value.get("authoritative", False)),
        exact_arxiv_id=bool(value.get("exact_arxiv_id", False)),
        available=bool(value.get("available", True)),
    )


def _fresh(checked_at: str, now: datetime.datetime, max_age: datetime.timedelta) -> bool:
    try:
        return now - _parse_timestamp(checked_at) <= max_age
    except (TypeError, ValueError):
        return False


def _lookup_source(
    source: CodeLookupSource,
    paper: dict[str, Any],
    cache: dict[str, Any],
    now: datetime.datetime,
    max_age: datetime.timedelta,
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[list[CodeCandidate], str, int]:
    cache_key = f"{source.name}:{paper['id']}"
    cached = cache["entries"].get(cache_key)
    if isinstance(cached, dict) and _fresh(cached.get("checked_at"), now, max_age):
        return [
            _candidate_from_json(candidate)
            for candidate in cached.get("candidates", [])
        ], "fresh-cache", 0

    for attempt in range(1, max_attempts + 1):
        try:
            candidates = list(source.lookup(paper))
            cache["entries"][cache_key] = {
                "checked_at": _format_timestamp(now),
                "candidates": [asdict(candidate) for candidate in candidates],
            }
            return candidates, "live", attempt
        except Exception:
            if attempt == max_attempts:
                if isinstance(cached, dict):
                    return [
                        _candidate_from_json(candidate)
                        for candidate in cached.get("candidates", [])
                    ], "stale-cache", attempt
                raise
            delay = min(retry_base_seconds * (2 ** (attempt - 1)), retry_max_seconds)
            sleep(delay)
    raise AssertionError("unreachable")


def resolve_code(candidates: Iterable[CodeCandidate], now: datetime.datetime) -> dict[str, Any]:
    usable = [candidate for candidate in candidates if candidate.available and candidate.url]
    ranked = sorted(
        usable,
        key=lambda candidate: (
            not candidate.authoritative,
            not candidate.exact_arxiv_id,
            candidate.source,
            candidate.url,
        ),
    )
    if not ranked:
        return {
            "status": "missing",
            "url": None,
            "source": None,
            "confidence": None,
            "checked_at": _format_timestamp(now),
            "evidence": [],
        }
    selected = ranked[0]
    verified = selected.authoritative or selected.exact_arxiv_id
    return {
        "status": "verified" if verified else "candidate",
        "url": selected.url,
        "source": selected.source,
        "confidence": 1 if selected.authoritative else 0.9 if selected.exact_arxiv_id else 0.35,
        "checked_at": _format_timestamp(now),
        "evidence": list(selected.evidence),
    }


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cache_version": 1, "entries": {}}
    cache = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(cache, dict)
        or cache.get("cache_version") != 1
        or not isinstance(cache.get("entries"), dict)
    ):
        raise ValueError("Invalid code lookup cache")
    return cache


def enrich_code_links(
    records: dict[str, dict[str, Any]],
    sources: list[CodeLookupSource],
    cache_path: Path,
    now: datetime.datetime,
    refresh_after: datetime.timedelta = datetime.timedelta(days=7),
    max_attempts: int = 3,
    retry_base_seconds: float = 2,
    retry_max_seconds: float = 20,
    sleep: Callable[[float], None] = time.sleep,
    max_papers: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    updated = json.loads(json.dumps(records))
    cache = _load_cache(cache_path)
    schema = load_schema()
    outcomes = {"verified": 0, "candidate": 0, "missing": 0, "preserved_on_failure": 0}
    cache_hits = 0
    live_lookups = 0
    stale_cache_uses = 0
    failed_sources: list[dict[str, str]] = []
    processed = 0

    for paper_id in sorted(updated):
        if max_papers is not None and processed >= max_papers:
            break
        paper = updated[paper_id]
        all_candidates: list[CodeCandidate] = []
        paper_failed = False
        for source in sources:
            try:
                candidates, lookup_type, _ = _lookup_source(
                    source,
                    paper,
                    cache,
                    now,
                    refresh_after,
                    max_attempts,
                    retry_base_seconds,
                    retry_max_seconds,
                    sleep,
                )
                all_candidates.extend(candidates)
                if lookup_type == "fresh-cache":
                    cache_hits += 1
                elif lookup_type == "stale-cache":
                    stale_cache_uses += 1
                else:
                    live_lookups += 1
            except Exception as error:
                paper_failed = True
                failed_sources.append({
                    "id": paper_id,
                    "source": source.name,
                    "error": type(error).__name__,
                })

        if all_candidates or not paper_failed:
            paper["code"] = resolve_code(all_candidates, now)
            outcomes[paper["code"]["status"]] += 1
        else:
            outcomes["preserved_on_failure"] += 1
        issues = validate_paper(paper, schema)
        if issues:
            raise ValueError(f"Code-enriched record {paper_id} is invalid: {issues}")
        processed += 1

    atomic_write_json(cache_path, cache)
    report = {
        "report_version": 1,
        "processed_papers": processed,
        "pending_papers": len(updated) - processed,
        "outcomes": outcomes,
        "cache_hits": cache_hits,
        "live_lookups": live_lookups,
        "stale_cache_uses": stale_cache_uses,
        "failed_sources": failed_sources,
    }
    return updated, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich canonical papers with code links.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-papers", type=int)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    enrichment_config = config.get("code_enrichment", {})
    timeout = float(enrichment_config.get("timeout_seconds", 10))
    records = load_canonical_archive(args.input)
    cache_path = args.cache or Path(enrichment_config["cache_path"])
    updated, report = enrich_code_links(
        records=records,
        sources=[
            PapersWithCodeLookup(timeout_seconds=timeout),
            GitHubRepositoryLookup(timeout_seconds=timeout),
        ],
        cache_path=cache_path,
        now=datetime.datetime.now(UTC),
        refresh_after=datetime.timedelta(
            days=float(enrichment_config.get("refresh_days", 7))
        ),
        max_attempts=int(enrichment_config.get("max_attempts", 3)),
        retry_base_seconds=float(enrichment_config.get("retry_base_seconds", 2)),
        retry_max_seconds=float(enrichment_config.get("retry_max_seconds", 20)),
        max_papers=args.max_papers,
    )
    write_canonical_archive(args.output, updated)
    report_path = args.report or Path(enrichment_config["report_path"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())