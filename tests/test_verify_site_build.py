import json
import tempfile
import unittest
from pathlib import Path

from generate_service_worker import generate_service_worker
from verify_site_build import verify_site_build


class VerifySiteBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.canonical = self.root / "canonical.json"
        self.dist = self.root / "dist"
        self.canonical.write_text(json.dumps({
            "2607.00001": {
                "id": "2607.00001",
                "arxiv_categories": ["cs.CL"],
                "published": "2026-07-01",
            },
        }), encoding="utf-8")
        self.dist.mkdir()
        (self.dist / "index.html").write_text("<h1>Feed</h1>", encoding="utf-8")
        (self.dist / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.parent.mkdir(parents=True)
        route.write_text(
            '<meta name="citation_title" content="Title">'
            '<meta name="citation_arxiv_id" content="2607.00001">'
            '<article data-pagefind-body><h1 data-pagefind-meta="title">Title</h1>'
            '<span data-pagefind-filter="topic">ASR</span>'
            '<span data-pagefind-filter="category">cs.CL</span>'
            '<span data-pagefind-filter="status">new</span>'
            '<span data-pagefind-filter="code:missing"></span>'
            '<span data-pagefind-filter="record_status:partial"></span>'
            '<span data-pagefind-filter="year:2026"></span>'
            '<span data-pagefind-filter="publication_date:known"></span>'
            '<time data-pagefind-sort="published">2026-07-01</time>'
            '<time data-pagefind-sort="updated">2026-07-02</time></article>',
            encoding="utf-8",
        )
        pagefind = self.dist / "pagefind"
        pagefind.mkdir()
        for name in ("pagefind.js", "pagefind-ui.js", "pagefind-entry.json"):
            (pagefind / name).write_text("{}", encoding="utf-8")
        bookmarks = self.dist / "bookmarks" / "index.html"
        bookmarks.parent.mkdir()
        bookmarks.write_text("<h1>Bookmarks</h1>", encoding="utf-8")
        offline = self.dist / "offline" / "index.html"
        offline.parent.mkdir()
        offline.write_text("<h1>Offline</h1>", encoding="utf-8")
        assets = self.dist / "_astro"
        assets.mkdir()
        (assets / "client.hash.js").write_text("// client", encoding="utf-8")
        generate_service_worker(self.dist)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_accepts_matching_routes_and_pagefind_bundle(self):
        report = verify_site_build(self.canonical, self.dist)
        self.assertEqual(1, report["canonical_papers"])
        self.assertEqual(1, report["paper_routes"])
        self.assertTrue(report["pagefind_ready"])
        self.assertTrue(report["offline_shell_ready"])
        self.assertEqual(5, report["precache_files"])

    def test_rejects_missing_routes(self):
        (self.dist / "papers" / "2607.00001" / "index.html").unlink()
        with self.assertRaisesRegex(ValueError, "missing_routes=1"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_index_files(self):
        (self.dist / "pagefind" / "pagefind.js").unlink()
        with self.assertRaisesRegex(ValueError, "pagefind.js"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_offline_shell_files(self):
        (self.dist / "sw.js").unlink()
        with self.assertRaisesRegex(ValueError, "sw.js"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_precache_files(self):
        (self.dist / "_astro" / "client.hash.js").unlink()
        with self.assertRaisesRegex(ValueError, "missing_files"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_assets_omitted_from_precache_manifest(self):
        worker = self.dist / "sw.js"
        worker.write_text(
            worker.read_text(encoding="utf-8").replace(
                ',\n  "./_astro/client.hash.js"\n', "\n"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "omitted"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_unscoped_cache_cleanup(self):
        worker = self.dist / "sw.js"
        worker.write_text(
            worker.read_text(encoding="utf-8").replace(
                "name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME",
                "name !== CACHE_NAME",
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "missing_worker_markers"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_pagefind_attributes(self):
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.write_text("<h1>Title</h1>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Pagefind attributes"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_category_facet_for_categorized_paper(self):
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.write_text(
            route.read_text(encoding="utf-8").replace(
                '<span data-pagefind-filter="category">cs.CL</span>', ""
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "category facet"):
            verify_site_build(self.canonical, self.dist)

    def test_rejects_missing_published_sort_for_published_paper(self):
        route = self.dist / "papers" / "2607.00001" / "index.html"
        route.write_text(
            route.read_text(encoding="utf-8").replace(
                '<time data-pagefind-sort="published">2026-07-01</time>', ""
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "published sort"):
            verify_site_build(self.canonical, self.dist)


if __name__ == "__main__":
    unittest.main()