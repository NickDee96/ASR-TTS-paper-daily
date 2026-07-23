import copy
import json
import tempfile
import unittest
from pathlib import Path

from paper_store import (
    DuplicateKeyError,
    atomic_write_json,
    build_run_manifest,
    canonical_shard_path,
    load_canonical_archive,
    load_json,
    merge_canonical_records,
    reconcile_archive,
    validate_run_manifest,
    write_canonical_archive,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_paper.json"


class PaperStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paper = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicates.json"
            path.write_text('{"2607.00001": {}, "2607.00001": {}}', encoding="utf-8")

            with self.assertRaisesRegex(DuplicateKeyError, "2607.00001"):
                load_json(path)

    def test_validated_atomic_write_round_trips(self) -> None:
        records = {self.paper["id"]: self.paper}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "papers.json"

            write_canonical_archive(path, records)

            self.assertEqual(load_canonical_archive(path), records)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_schema_error_and_impossible_date_block_replacement(self) -> None:
        records = {self.paper["id"]: self.paper}
        invalid = copy.deepcopy(records)
        invalid[self.paper["id"]]["updated"] = "2025-01-01"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "papers.json"
            write_canonical_archive(path, records)
            original = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "date_order"):
                write_canonical_archive(path, invalid)

            self.assertEqual(path.read_bytes(), original)

    def test_unexpected_count_loss_is_blocked(self) -> None:
        second = copy.deepcopy(self.paper)
        second["id"] = "2607.00002"
        second["links"]["abstract"] = "https://arxiv.org/abs/2607.00002"
        second["links"]["pdf"] = "https://arxiv.org/pdf/2607.00002"
        records = {self.paper["id"]: self.paper, second["id"]: second}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "papers.json"
            write_canonical_archive(path, records)

            with self.assertRaisesRegex(ValueError, "unexpectedly remove"):
                write_canonical_archive(path, {self.paper["id"]: self.paper})

            write_canonical_archive(
                path,
                {self.paper["id"]: self.paper},
                allowed_removed_ids={second["id"]},
            )
            self.assertEqual(set(load_canonical_archive(path)), {self.paper["id"]})

    def test_interrupted_replace_keeps_previous_file_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.json"
            path.write_text('{"state": "old"}\n', encoding="utf-8")
            original = path.read_bytes()

            def fail_replace(source, destination):
                raise OSError("interrupted")

            with self.assertRaisesRegex(OSError, "interrupted"):
                atomic_write_json(path, {"state": "new"}, replace=fail_replace)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_shard_path_uses_valid_year_and_month(self) -> None:
        root = Path("data/papers")
        self.assertEqual(
            canonical_shard_path(root, "2026-07-23"),
            (root / "2026" / "07.json").resolve(),
        )
        with self.assertRaisesRegex(ValueError, "Invalid publication date"):
            canonical_shard_path(root, "2026-13-23")

    def test_merge_reports_inserted_updated_and_unchanged(self) -> None:
        existing = {self.paper["id"]: self.paper}
        changed = copy.deepcopy(self.paper)
        changed["title"] = "Changed title"
        second = copy.deepcopy(self.paper)
        second["id"] = "2607.00002"
        second["links"]["abstract"] = "https://arxiv.org/abs/2607.00002"
        second["links"]["pdf"] = "https://arxiv.org/pdf/2607.00002"

        merged, counts = merge_canonical_records(
            existing,
            {changed["id"]: changed, second["id"]: second},
        )

        self.assertEqual(counts, {"inserted": 1, "updated": 1, "unchanged": 0})
        self.assertEqual(len(merged), 2)

    def test_manifest_rejects_empty_and_inconsistent_runs(self) -> None:
        batch = {
            "window_end": "2026-07-23T12:00:00Z",
            "topics": {
                "ASR": {"fetched": 5, "accepted": 3, "rejected": 2}
            },
        }
        manifest = build_run_manifest(
            batch,
            {"inserted": 1, "updated": 1, "unchanged": 1},
            failed_enrichments=2,
        )

        validate_run_manifest(manifest)
        self.assertEqual(manifest["totals"]["fetched"], 5)
        empty = build_run_manifest(
            {"window_end": batch["window_end"], "topics": {}},
            {},
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_run_manifest(empty)
        validate_run_manifest(empty, allow_empty=True)
        manifest["totals"]["accepted"] = 4
        with self.assertRaisesRegex(ValueError, "accepted plus rejected"):
            validate_run_manifest(manifest)

    def test_reconciliation_repairs_known_missing_records(self) -> None:
        second = copy.deepcopy(self.paper)
        second["id"] = "2607.00002"
        second["links"]["abstract"] = "https://arxiv.org/abs/2607.00002"
        second["links"]["pdf"] = "https://arxiv.org/pdf/2607.00002"

        repaired, report = reconcile_archive(
            {self.paper["id"]: self.paper},
            {self.paper["id"], second["id"], "2607.00003"},
            {second["id"]: second},
        )

        self.assertEqual(set(repaired), {self.paper["id"], second["id"]})
        self.assertEqual(report["repaired_ids"], [second["id"]])
        self.assertEqual(report["unresolved_ids"], ["2607.00003"])


if __name__ == "__main__":
    unittest.main()