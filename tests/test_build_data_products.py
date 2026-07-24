import copy
import datetime
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from build_data_products import build_data_products
from paper_collector import UTC


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_paper.json"
UPDATED_AT = datetime.datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
ATOM = "{http://www.w3.org/2005/Atom}"
SITE_URL = "https://nickdee96.github.io"
BASE_PATH = "/ASR-TTS-paper-daily"


def _read_feed(path: Path) -> tuple[ET.Element, list[ET.Element]]:
    root = ET.parse(path).getroot()
    return root, root.findall(f"{ATOM}entry")


def _entry_ids(entries: list[ET.Element]) -> list[str]:
    return [entry.find(f"{ATOM}id").text for entry in entries]


def paper_for(paper_id: str, published: str | None, topics: list[str]):
    paper = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    paper["id"] = paper_id
    paper["published"] = published
    paper["updated"] = published or "2026-07-23"
    paper["topics"] = topics
    paper["classification"]["matches"] = [
        {
            **paper["classification"]["matches"][0],
            "topic": topic,
        }
        for topic in topics
    ]
    paper["links"]["abstract"] = f"https://arxiv.org/abs/{paper_id}"
    paper["links"]["pdf"] = f"https://arxiv.org/pdf/{paper_id}"
    if published is None:
        paper["record_status"] = "partial"
        paper["abstract"] = None
        paper["published"] = None
        paper["arxiv_categories"] = []
        paper["primary_category"] = None
    return paper


class DataProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = {
            "2607.00001": paper_for("2607.00001", "2026-07-01", ["ASR"]),
            "2606.00002": paper_for(
                "2606.00002", "2026-06-15", ["TTS", "Synthetic Generation"]
            ),
            "2501.00003": paper_for("2501.00003", "2025-01-02", ["ASR"]),
            "1307.4048": paper_for("1307.4048", None, ["Machine Translation"]),
        }

    def _files(self, root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*.json")
        }

    def test_build_partitions_once_and_bounds_latest_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            manifest = build_data_products(
                self.records, root, UPDATED_AT, latest_limit=2
            )

            self.assertEqual(manifest["unique_papers"], 4)
            self.assertEqual(sum(shard["count"] for shard in manifest["shards"]), 4)
            self.assertEqual(
                {shard["path"] for shard in manifest["shards"]},
                {
                    "papers/2026/07.json",
                    "papers/2026/06.json",
                    "papers/2025/01.json",
                    "papers/undated.json",
                },
            )
            latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(latest["papers"]), 2)
            self.assertLess((root / "latest.json").stat().st_size, 10000)

    def test_manifest_checksums_match_exact_shard_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = build_data_products(self.records, root, UPDATED_AT)

            for shard in manifest["shards"]:
                digest = hashlib.sha256((root / shard["path"]).read_bytes()).hexdigest()
                self.assertEqual(digest, shard["sha256"])

    def test_facets_and_statistics_distinguish_papers_and_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_data_products(self.records, root, UPDATED_AT)

            facets = json.loads((root / "facets.json").read_text(encoding="utf-8"))
            statistics = json.loads(
                (root / "statistics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(facets["facets"]["topics"]["ASR"], 2)
        self.assertEqual(facets["facets"]["years"]["Unknown"], 1)
        self.assertEqual(statistics["unique_papers"], 4)
        self.assertEqual(statistics["topic_assignments"], 5)

    def test_clean_and_incremental_builds_are_byte_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            clean_root = root / "clean"
            incremental_root = root / "incremental"
            stale = incremental_root / "papers" / "1999" / "01.json"
            stale.parent.mkdir(parents=True)
            stale.write_text("{}", encoding="utf-8")

            build_data_products(
                self.records, clean_root, UPDATED_AT, latest_limit=3, clean=True
            )
            build_data_products(
                self.records, incremental_root, UPDATED_AT, latest_limit=3, clean=True
            )

            self.assertFalse(stale.exists())
            self.assertEqual(self._files(clean_root), self._files(incremental_root))

    def test_rebuild_removes_stale_shard_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_data_products(self.records, root, UPDATED_AT, clean=True)
            reduced = copy.deepcopy(self.records)
            del reduced["2501.00003"]

            manifest = build_data_products(reduced, root, UPDATED_AT, clean=True)

            self.assertFalse((root / "papers" / "2025" / "01.json").exists())
            self.assertEqual(sum(shard["count"] for shard in manifest["shards"]), 3)

    def test_feeds_cover_all_papers_and_each_configured_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = build_data_products(
                self.records,
                root,
                UPDATED_AT,
                site_url=SITE_URL,
                base_path=BASE_PATH,
                feed_topics=["ASR", "TTS", "Machine Translation", "Small Language Models"],
            )

            self.assertEqual(manifest["feeds"]["all"], "feeds/all.xml")
            self.assertEqual(
                set(manifest["feeds"]["topics"]),
                {"ASR", "TTS", "Machine Translation", "Small Language Models"},
            )

            root_feed, entries = _read_feed(root / "feeds" / "all.xml")
            self.assertEqual(root_feed.tag, f"{ATOM}feed")
            self.assertEqual(
                set(_entry_ids(entries)),
                {
                    "https://arxiv.org/abs/2607.00001",
                    "https://arxiv.org/abs/2606.00002",
                    "https://arxiv.org/abs/2501.00003",
                    "https://arxiv.org/abs/1307.4048",
                },
            )
            first_link = entries[0].find(f"{ATOM}link").get("href")
            self.assertTrue(first_link.startswith(f"{SITE_URL}{BASE_PATH}/papers/"))

            _, asr_entries = _read_feed(root / "feeds" / "topic-asr.xml")
            self.assertEqual(
                set(_entry_ids(asr_entries)),
                {"https://arxiv.org/abs/2607.00001", "https://arxiv.org/abs/2501.00003"},
            )
            _, empty_entries = _read_feed(
                root / "feeds" / "topic-small-language-models.xml"
            )
            self.assertEqual(empty_entries, [])

    def test_feed_entries_are_bounded_and_sorted_by_recency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_data_products(
                self.records,
                root,
                UPDATED_AT,
                site_url=SITE_URL,
                base_path=BASE_PATH,
                feed_topics=["ASR"],
                feed_limit=2,
            )

            _, entries = _read_feed(root / "feeds" / "all.xml")
            self.assertEqual(
                _entry_ids(entries),
                ["https://arxiv.org/abs/1307.4048", "https://arxiv.org/abs/2607.00001"],
            )
            updated = entries[0].find(f"{ATOM}updated").text
            self.assertRegex(updated, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_feed_escapes_markup_and_preserves_unicode(self) -> None:
        record = paper_for("2607.09999", "2026-07-05", ["ASR"])
        record["title"] = "Speech & <Language> Models"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_data_products(
                {record["id"]: record},
                root,
                UPDATED_AT,
                site_url=SITE_URL,
                base_path=BASE_PATH,
                feed_topics=["ASR"],
            )
            raw = (root / "feeds" / "all.xml").read_text(encoding="utf-8")
            self.assertNotIn("<Language>", raw)
            self.assertIn("&amp;", raw)

            _, entries = _read_feed(root / "feeds" / "all.xml")
            self.assertEqual(
                entries[0].find(f"{ATOM}title").text, "Speech & <Language> Models"
            )
            author = entries[0].find(f"{ATOM}author/{ATOM}name").text
            self.assertEqual(author, "Māia Example")


if __name__ == "__main__":
    unittest.main()