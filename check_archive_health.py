import argparse
import json
from pathlib import Path
from typing import Any


def _unique_papers(manifest_path: Path) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = data.get("unique_papers")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Manifest {manifest_path} has no valid unique_papers count")
    return value


def evaluate_archive_health(
    current: int,
    previous: int | None,
    max_drop_fraction: float = 0.1,
    min_papers: int = 1,
) -> dict[str, Any]:
    """Judge whether a freshly built archive is safe to publish.

    A build is unhealthy when it collapses to fewer than ``min_papers`` records or
    shrinks past ``max_drop_fraction`` of the previously published archive. Growth and
    small fluctuations are always allowed.
    """
    if not 0 <= max_drop_fraction < 1:
        raise ValueError("max_drop_fraction must be in the range [0, 1)")
    if min_papers < 1:
        raise ValueError("min_papers must be at least 1")

    if current < min_papers:
        return {
            "healthy": False,
            "reason": f"Archive has {current} papers, below the minimum of {min_papers}",
            "current": current,
            "previous": previous,
        }
    if previous is not None and previous > 0:
        floor = previous * (1 - max_drop_fraction)
        if current < floor:
            return {
                "healthy": False,
                "reason": (
                    f"Archive shrank from {previous} to {current} papers, "
                    f"below the allowed floor of {floor:.0f}"
                ),
                "current": current,
                "previous": previous,
            }
    return {
        "healthy": True,
        "reason": "Archive size is within the expected range",
        "current": current,
        "previous": previous,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail the build when a new archive is empty or drops sharply."
    )
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--max-drop-fraction", type=float, default=0.1)
    parser.add_argument("--min-papers", type=int, default=1)
    args = parser.parse_args()

    current = _unique_papers(args.current)
    previous: int | None = None
    if args.previous is not None and args.previous.is_file():
        previous = _unique_papers(args.previous)

    result = evaluate_archive_health(
        current,
        previous,
        max_drop_fraction=args.max_drop_fraction,
        min_papers=args.min_papers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
