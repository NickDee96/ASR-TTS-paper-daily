import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from build_data_products import build_data_products
from migrate_legacy import migrate_archive
from paper_collector import UTC, _parse_timestamp
from paper_store import atomic_write_json


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def _read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _derive_updated_at(records: dict[str, dict[str, Any]]) -> datetime.datetime:
    timestamps: list[datetime.datetime] = []
    for record in records.values():
        value = record.get("updated") or record.get("published")
        if not isinstance(value, str) or not value:
            continue
        try:
            timestamps.append(_parse_timestamp(value))
        except ValueError:
            continue
    if not timestamps:
        raise ValueError("Cannot derive an archive update time from the paper records")
    return max(timestamps).astimezone(UTC)


def _atomic_write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(f"{value}\n" for value in values))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_site_content(
    archive_path: Path,
    baseline_ids_path: Path,
    generated_root: Path,
    public_data_root: Path,
    *,
    updated_at: datetime.datetime | None = None,
    latest_limit: int = 100,
    site_url: str = "",
    base_path: str = "/",
    feed_topics: list[str] | None = None,
    feed_limit: int = 50,
    run_manifest_path: Path | None = None,
    stale_after_hours: int = 36,
) -> dict[str, Any]:
    records, migration_report, backfill_ids = migrate_archive(
        _read_json_object(archive_path),
        expected_ids=_read_ids(baseline_ids_path),
    )
    if not migration_report["valid"]:
        raise ValueError(
            "Site content migration failed reconciliation: "
            f"missing={len(migration_report['missing_expected_ids'])}, "
            f"unexpected={len(migration_report['unexpected_ids'])}, "
            f"schema_issues={len(migration_report['schema_issues'])}"
        )

    run_manifest: dict[str, Any] | None = None
    if run_manifest_path is not None and run_manifest_path.is_file():
        run_manifest = _read_json_object(run_manifest_path)

    effective_updated_at = updated_at or _derive_updated_at(records)
    canonical_path = generated_root / "canonical.json"
    atomic_write_json(canonical_path, records)
    atomic_write_json(generated_root / "migration-report.json", migration_report)
    _atomic_write_lines(generated_root / "backfill-paper-ids.txt", backfill_ids)
    manifest = build_data_products(
        records,
        public_data_root,
        effective_updated_at,
        latest_limit=latest_limit,
        clean=True,
        site_url=site_url,
        base_path=base_path,
        feed_topics=feed_topics,
        feed_limit=feed_limit,
        run_manifest=run_manifest,
        stale_after_hours=stale_after_hours,
    )
    summary = {
        "canonical_path": canonical_path.as_posix(),
        "paper_count": len(records),
        "backfill_required": len(backfill_ids),
        "updated_at": manifest["updated_at"],
        "shard_count": len(manifest["shards"]),
    }
    atomic_write_json(generated_root / "site-content-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare canonical and derived paper data for the static site"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "asr-arxiv-daily.json",
    )
    parser.add_argument(
        "--baseline-ids",
        type=Path,
        default=REPOSITORY_ROOT / "migration" / "baseline-paper-ids.txt",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=REPOSITORY_ROOT / "site" / ".generated",
    )
    parser.add_argument(
        "--public-data-root",
        type=Path,
        default=REPOSITORY_ROOT / "site" / "public" / "data",
    )
    parser.add_argument("--updated-at")
    parser.add_argument("--latest-limit", type=int, default=100)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.yaml")
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    user_name = config.get("user_name") or ""
    repo_name = config.get("repo_name") or ""
    product_config = config.get("data_products", {})

    updated_at = _parse_timestamp(args.updated_at) if args.updated_at else None
    summary = prepare_site_content(
        args.archive,
        args.baseline_ids,
        args.generated_root,
        args.public_data_root,
        updated_at=updated_at,
        latest_limit=args.latest_limit,
        site_url=f"https://{user_name}.github.io" if user_name else "",
        base_path=f"/{repo_name}" if repo_name else "/",
        feed_topics=list(config.get("keywords", {}).keys()) or None,
        feed_limit=int(product_config.get("feed_limit", 50)),
        run_manifest_path=args.run_manifest,
        stale_after_hours=int(product_config.get("stale_after_hours", 36)),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())