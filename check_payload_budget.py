import argparse
import json
from pathlib import Path
from typing import Any

# Budgets are deliberately set above current output with headroom so the gate
# catches regressions (a runaway bundle, or the full archive leaking into a page)
# without failing on normal growth. Sizes are uncompressed kilobytes.
DEFAULT_BUDGETS: dict[str, float] = {
    "max_js_chunk_kb": 220,
    "max_total_js_kb": 720,
    "max_html_kb": 80,
    "max_initial_data_kb": 16,
}


def _kb(path: Path) -> float:
    return path.stat().st_size / 1024


def evaluate_payload_budget(
    dist_root: Path,
    budgets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Enforce initial JavaScript, page, and eager-data budgets on a built site.

    The feed and other entry pages are rendered statically, so a bounded page size
    is the guarantee that opening the feed never transfers the full archive.
    """
    effective = {**DEFAULT_BUDGETS, **(budgets or {})}
    issues: list[str] = []

    astro_root = dist_root / "_astro"
    js_files = sorted(astro_root.glob("*.js")) if astro_root.is_dir() else []
    if not js_files:
        raise ValueError("No _astro JavaScript chunks found; build the site first")

    total_js = 0.0
    for chunk in js_files:
        size = _kb(chunk)
        total_js += size
        if size > effective["max_js_chunk_kb"]:
            issues.append(
                f"JS chunk {chunk.name} is {size:.0f}KB "
                f"(budget {effective['max_js_chunk_kb']:.0f}KB)"
            )
    if total_js > effective["max_total_js_kb"]:
        issues.append(
            f"Total JS is {total_js:.0f}KB (budget {effective['max_total_js_kb']:.0f}KB)"
        )

    for name, relative in (
        ("feed", Path("index.html")),
        ("search", Path("search") / "index.html"),
        ("status", Path("status") / "index.html"),
        ("bookmarks", Path("bookmarks") / "index.html"),
    ):
        page = dist_root / relative
        if page.is_file() and _kb(page) > effective["max_html_kb"]:
            issues.append(
                f"{name} page is {_kb(page):.0f}KB (budget {effective['max_html_kb']:.0f}KB)"
            )

    manifest = dist_root / "data" / "manifest.json"
    if manifest.is_file() and _kb(manifest) > effective["max_initial_data_kb"]:
        issues.append(
            f"manifest.json is {_kb(manifest):.0f}KB "
            f"(budget {effective['max_initial_data_kb']:.0f}KB); it is fetched on every page"
        )

    return {
        "healthy": not issues,
        "issues": issues,
        "js_chunks": len(js_files),
        "total_js_kb": round(total_js, 1),
        "largest_js_kb": round(max(_kb(chunk) for chunk in js_files), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail the build when initial payload budgets are exceeded."
    )
    parser.add_argument("--dist", type=Path, default=Path("site/dist"))
    parser.add_argument("--max-js-chunk-kb", type=float)
    parser.add_argument("--max-total-js-kb", type=float)
    parser.add_argument("--max-html-kb", type=float)
    parser.add_argument("--max-initial-data-kb", type=float)
    args = parser.parse_args()

    overrides = {
        "max_js_chunk_kb": args.max_js_chunk_kb,
        "max_total_js_kb": args.max_total_js_kb,
        "max_html_kb": args.max_html_kb,
        "max_initial_data_kb": args.max_initial_data_kb,
    }
    budgets = {key: value for key, value in overrides.items() if value is not None}

    result = evaluate_payload_budget(args.dist, budgets)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
