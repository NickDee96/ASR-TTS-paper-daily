import datetime
import json
import tempfile
import unittest
from pathlib import Path

from prepare_site_content import prepare_site_content


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "legacy_archive.json"


class PrepareSiteContentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.archive_path = self.root / "legacy.json"
        self.archive_path.write_text(
            FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        archive = json.loads(self.archive_path.read_text(encoding="utf-8"))
        paper_ids = sorted(
            {
                paper_id
                for topic_rows in archive.values()
                for paper_id in topic_rows
            }
        )
        self.baseline_path = self.root / "baseline.txt"
        self.baseline_path.write_text(
            "".join(f"{paper_id}\n" for paper_id in paper_ids),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_prepares_reconciled_canonical_and_derived_outputs(self):
        generated = self.root / "generated"
        public = self.root / "public"

        summary = prepare_site_content(
            self.archive_path,
            self.baseline_path,
            generated,
            public,
            latest_limit=2,
        )

        canonical = json.loads(
            (generated / "canonical.json").read_text(encoding="utf-8")
        )
        latest = json.loads((public / "latest.json").read_text(encoding="utf-8"))
        manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(6, summary["paper_count"])
        self.assertEqual(6, summary["backfill_required"])
        self.assertEqual("2026-07-06T00:00:00Z", summary["updated_at"])
        self.assertEqual(set(canonical), set(self.baseline_path.read_text().splitlines()))
        self.assertEqual(2, len(latest["papers"]))
        self.assertEqual(6, manifest["unique_papers"])
        self.assertTrue((generated / "migration-report.json").exists())
        self.assertTrue((generated / "backfill-paper-ids.txt").exists())

    def test_explicit_update_time_is_used_deterministically(self):
        timestamp = datetime.datetime(2026, 8, 1, 9, 30, tzinfo=datetime.timezone.utc)

        first = prepare_site_content(
            self.archive_path,
            self.baseline_path,
            self.root / "generated-a",
            self.root / "public-a",
            updated_at=timestamp,
        )
        second = prepare_site_content(
            self.archive_path,
            self.baseline_path,
            self.root / "generated-b",
            self.root / "public-b",
            updated_at=timestamp,
        )

        self.assertEqual("2026-08-01T09:30:00Z", first["updated_at"])
        self.assertEqual(first["updated_at"], second["updated_at"])
        self.assertEqual(
            (self.root / "public-a" / "manifest.json").read_bytes(),
            (self.root / "public-b" / "manifest.json").read_bytes(),
        )

    def test_baseline_mismatch_blocks_all_outputs(self):
        self.baseline_path.write_text("9999.99999\n", encoding="utf-8")
        generated = self.root / "generated"
        public = self.root / "public"

        with self.assertRaisesRegex(ValueError, "failed reconciliation"):
            prepare_site_content(
                self.archive_path,
                self.baseline_path,
                generated,
                public,
            )

        self.assertFalse(generated.exists())
        self.assertFalse(public.exists())

    def test_clean_rebuild_removes_stale_public_shards(self):
        generated = self.root / "generated"
        public = self.root / "public"
        prepare_site_content(
            self.archive_path, self.baseline_path, generated, public
        )
        stale = public / "papers" / "1900-01.json"
        stale.write_text("{}", encoding="utf-8")

        prepare_site_content(
            self.archive_path, self.baseline_path, generated, public
        )

        self.assertFalse(stale.exists())

    def test_persisted_canonical_overlay_preserves_enrichment_and_adds_new_ids(self):
        generated = self.root / "generated"
        public = self.root / "public"
        persisted_path = self.root / "canonical-archive.json"

        first = prepare_site_content(
            self.archive_path,
            self.baseline_path,
            generated,
            public,
            persisted_canonical_path=persisted_path,
        )
        self.assertEqual(first["backfill_required"], 6)
        persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
        some_id = next(iter(persisted))
        persisted[some_id]["record_status"] = "complete"
        persisted[some_id]["abstract"] = "A fully enriched abstract."
        persisted[some_id]["published"] = "2026-07-01"
        persisted[some_id]["arxiv_categories"] = ["cs.CL"]
        persisted[some_id]["primary_category"] = "cs.CL"
        persisted_path.write_text(
            json.dumps(persisted, ensure_ascii=False), encoding="utf-8"
        )

        second = prepare_site_content(
            self.archive_path,
            self.baseline_path,
            generated,
            public,
            persisted_canonical_path=persisted_path,
        )

        canonical = json.loads(
            (generated / "canonical.json").read_text(encoding="utf-8")
        )
        self.assertEqual(canonical[some_id]["abstract"], "A fully enriched abstract.")
        self.assertEqual(canonical[some_id]["record_status"], "complete")
        self.assertEqual(second["backfill_required"], 5)
        reloaded_persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
        self.assertEqual(set(reloaded_persisted), set(canonical))


if __name__ == "__main__":
    unittest.main()