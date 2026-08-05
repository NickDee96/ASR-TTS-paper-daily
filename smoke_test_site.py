import argparse
import json
from pathlib import Path
from typing import Any


def _read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def evaluate_route_coverage(baseline_ids: set[str], dist_root: Path) -> dict[str, Any]:
    """Verify every migrated paper ID has a published route in the built site."""
    missing = sorted(
        paper_id
        for paper_id in baseline_ids
        if not (dist_root / "papers" / paper_id / "index.html").is_file()
    )
    return {
        "healthy": not missing,
        "expected": len(baseline_ids),
        "present": len(baseline_ids) - len(missing),
        "missing_count": len(missing),
        "missing_sample": missing[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test that the built site covers the complete migrated ID set."
    )
    parser.add_argument(
        "--baseline-ids", type=Path, default=Path("migration/baseline-paper-ids.txt")
    )
    parser.add_argument("--dist", type=Path, default=Path("site/dist"))
    args = parser.parse_args()

    baseline_ids = _read_ids(args.baseline_ids)
    if not baseline_ids:
        raise ValueError(f"No baseline IDs found in {args.baseline_ids}")

    result = evaluate_route_coverage(baseline_ids, args.dist)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
