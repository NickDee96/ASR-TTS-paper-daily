import argparse
import json
from pathlib import Path
from typing import Any

from generate_service_worker import (
    CACHE_PREFIX,
    collect_precache_paths,
    compute_precache_version,
    parse_service_worker_manifest,
    resolve_precache_path,
)


def verify_site_build(canonical_path: Path, dist_root: Path) -> dict[str, Any]:
    with canonical_path.open("r", encoding="utf-8") as handle:
        canonical = json.load(handle)
    if not isinstance(canonical, dict):
        raise ValueError("Canonical archive must be a JSON object")

    missing_routes = [
        paper_id
        for paper_id in canonical
        if not (dist_root / "papers" / paper_id / "index.html").is_file()
    ]
    pagefind_root = dist_root / "pagefind"
    required_index_files = [
        pagefind_root / "pagefind.js",
        pagefind_root / "pagefind-ui.js",
        pagefind_root / "pagefind-entry.json",
    ]
    missing_index_files = [
        path.name for path in required_index_files if not path.is_file()
    ]
    detail_pages = list((dist_root / "papers").glob("*/index.html"))
    required_shell_files = [
        dist_root / "bookmarks" / "index.html",
        dist_root / "offline" / "index.html",
        dist_root / "sw.js",
    ]
    missing_shell_files = [
        path.relative_to(dist_root).as_posix()
        for path in required_shell_files
        if not path.is_file()
    ]
    if (
        missing_routes
        or missing_index_files
        or missing_shell_files
        or len(detail_pages) != len(canonical)
    ):
        raise ValueError(
            "Static site verification failed: "
            f"canonical={len(canonical)}, routes={len(detail_pages)}, "
            f"missing_routes={len(missing_routes)}, "
            f"missing_index_files={missing_index_files}, "
            f"missing_shell_files={missing_shell_files}"
        )

    worker_source = (dist_root / "sw.js").read_text(encoding="utf-8")
    worker_version, precache_paths = parse_service_worker_manifest(worker_source)
    expected_precache_paths = collect_precache_paths(dist_root)
    duplicate_precache_paths = sorted({
        path for path in precache_paths if precache_paths.count(path) > 1
    })
    missing_precache_files = sorted({
        path for path in precache_paths
        if not resolve_precache_path(dist_root, path).is_file()
    })
    omitted_precache_paths = sorted(set(expected_precache_paths) - set(precache_paths))
    unexpected_precache_paths = sorted(set(precache_paths) - set(expected_precache_paths))
    expected_worker_version = compute_precache_version(dist_root, expected_precache_paths)
    required_worker_markers = (
        f"const CACHE_PREFIX = '{CACHE_PREFIX}';",
        "name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME",
        "await cache.addAll(PRECACHE_URLS)",
        "if (isBookmarksNavigation(url))",
        "cache.match(OFFLINE_URL)",
    )
    missing_worker_markers = [
        marker for marker in required_worker_markers if marker not in worker_source
    ]
    if (
        duplicate_precache_paths
        or missing_precache_files
        or omitted_precache_paths
        or unexpected_precache_paths
        or worker_version != expected_worker_version
        or missing_worker_markers
    ):
        raise ValueError(
            "Offline precache verification failed: "
            f"duplicates={duplicate_precache_paths}, "
            f"missing_files={missing_precache_files}, "
            f"omitted={omitted_precache_paths}, "
            f"unexpected={unexpected_precache_paths}, "
            f"version_match={worker_version == expected_worker_version}, "
            f"missing_worker_markers={missing_worker_markers}"
        )

    sample_path = dist_root / "papers" / next(iter(sorted(canonical))) / "index.html"
    sample_html = sample_path.read_text(encoding="utf-8")
    required_attributes = (
        "data-pagefind-body",
        'data-pagefind-meta="title"',
        'data-pagefind-filter="topic"',
        'data-pagefind-filter="status"',
        'data-pagefind-filter="code:',
        'data-pagefind-filter="record_status:',
        'data-pagefind-filter="year:',
        'data-pagefind-filter="publication_date:',
        'data-pagefind-sort="updated"',
        'name="citation_title"',
        'name="citation_arxiv_id"',
    )
    missing_attributes = [
        attribute for attribute in required_attributes if attribute not in sample_html
    ]
    if missing_attributes:
        raise ValueError(
            f"Sample paper page is missing Pagefind attributes: {missing_attributes}"
        )
    category_papers = [
        paper_id
        for paper_id, record in canonical.items()
        if isinstance(record, dict) and record.get("arxiv_categories")
    ]
    if category_papers:
        category_sample = (
            dist_root / "papers" / category_papers[0] / "index.html"
        ).read_text(encoding="utf-8")
        if 'data-pagefind-filter="category"' not in category_sample:
            raise ValueError("Categorized paper page is missing the category facet")
    published_papers = [
        paper_id
        for paper_id, record in canonical.items()
        if isinstance(record, dict) and record.get("published")
    ]
    if published_papers:
        published_sample = (
            dist_root / "papers" / published_papers[0] / "index.html"
        ).read_text(encoding="utf-8")
        if 'data-pagefind-sort="published"' not in published_sample:
            raise ValueError("Published paper page is missing the published sort")
    return {
        "canonical_papers": len(canonical),
        "paper_routes": len(detail_pages),
        "pagefind_ready": True,
        "offline_shell_ready": True,
        "precache_files": len(precache_paths),
        "precache_version": worker_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify static paper routes and search index")
    parser.add_argument(
        "--canonical", type=Path, default=Path(".generated/canonical.json")
    )
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(json.dumps(verify_site_build(args.canonical, args.dist), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())