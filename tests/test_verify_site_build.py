import json
import tempfile
import unittest
from pathlib import Path

from verify_site_build import verify_site_build


class VerifySiteBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.canonical = self.root / "canonical.json"
        self.dist = self.root / "dist"
        self.canonical.write_text(
            json.dumps({"2607.00001": {"id": "2607.00001"}}), encoding="utf-8"
        )
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.parent.mkdir(parents=True)
        route.write_text(
            '<article data-pagefind-body><h1 data-pagefind-meta="title">Title</h1>'
            '<span data-pagefind-filter="topic">ASR</span>'
            '<time data-pagefind-sort="date">2026-07-01</time></article>',
            encoding="utf-8",
        )
        pagefind = self.dist / "pagefind"
        pagefind.mkdir()
        for name in ("pagefind.js", "pagefind-ui.js", "pagefind-entry.json"):
            (pagefind / name).write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_accepts_matching_routes_and_pagefind_bundle(self):
        report = verify_site_build(self.canonical, self.dist)
        self.assertEqual(1, report["canonical_papers"])
        self.assertEqual(1, report["paper_routes"])
        self.assertTrue(report["pagefind_ready"])

    def test_rejects_missing_routes(self):
        (self.dist / "papers" / "2607.00001" / "index.html").unlink()
        with self.assertRaisesRegex(ValueError, "missing_routes=1"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_index_files(self):
        (self.dist / "pagefind" / "pagefind.js").unlink()
        with self.assertRaisesRegex(ValueError, "pagefind.js"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_pagefind_attributes(self):
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.write_text("<h1>Title</h1>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Pagefind attributes"):
            verify_site_build(self.canonical, self.dist)


if __name__ == "__main__":
    unittest.main()