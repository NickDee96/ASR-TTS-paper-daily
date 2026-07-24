import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence


CACHE_PREFIX = "asr-tts-reader-"
MANIFEST_START = "/* precache-manifest:start */"
MANIFEST_END = "/* precache-manifest:end */"
REQUIRED_SHELL_PATHS = ("./", "./bookmarks/", "./offline/", "./favicon.svg")


def resolve_precache_path(dist_root: Path, path: str) -> Path:
    if not path.startswith("./") or "?" in path or "#" in path:
        raise ValueError(f"Invalid precache path: {path}")
    relative = path[2:]
    parts = PurePosixPath(relative).parts
    if ".." in parts:
        raise ValueError(f"Precache path escapes the build root: {path}")
    if path.endswith("/"):
        parts = (*parts, "index.html")
    return dist_root.joinpath(*parts)


def collect_precache_paths(dist_root: Path) -> list[str]:
    paths = list(REQUIRED_SHELL_PATHS)
    asset_root = dist_root / "_astro"
    if not asset_root.is_dir():
        raise ValueError("Astro asset directory is missing")
    paths.extend(
        f"./{asset.relative_to(dist_root).as_posix()}"
        for asset in sorted(asset_root.rglob("*"))
        if asset.is_file()
    )
    missing = [
        path for path in paths if not resolve_precache_path(dist_root, path).is_file()
    ]
    if missing:
        raise ValueError(f"Precache inputs are missing: {missing}")
    return paths


def compute_precache_version(dist_root: Path, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolve_precache_path(dist_root, path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def render_service_worker(paths: Sequence[str], version: str) -> str:
    manifest = json.dumps(list(paths), indent=2)
    return f"""const CACHE_PREFIX = '{CACHE_PREFIX}';
const PRECACHE_VERSION = '{version}';
const CACHE_NAME = `${{CACHE_PREFIX}}${{PRECACHE_VERSION}}`;
const PRECACHE_PATHS =
{MANIFEST_START}
{manifest}
{MANIFEST_END};
const SCOPE_URL = new URL(self.registration.scope);
const PRECACHE_URLS = PRECACHE_PATHS.map((path) => new URL(path, SCOPE_URL).href);
const BOOKMARKS_URL = new URL('./bookmarks/', SCOPE_URL);
const OFFLINE_URL = new URL('./offline/', SCOPE_URL).href;

function isBookmarksNavigation(url) {{
  return url.pathname === BOOKMARKS_URL.pathname
    || `${{url.pathname}}/` === BOOKMARKS_URL.pathname;
}}

async function cacheNetworkResponse(cache, request, response) {{
  if (response.ok) await cache.put(request, response.clone());
  return response;
}}

self.addEventListener('install', (event) => {{
  event.waitUntil((async () => {{
    const cache = await caches.open(CACHE_NAME);
    await cache.addAll(PRECACHE_URLS);
    await self.skipWaiting();
  }})());
}});

self.addEventListener('activate', (event) => {{
  event.waitUntil((async () => {{
    const names = await caches.keys();
    await Promise.all(names
      .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  }})());
}});

self.addEventListener('fetch', (event) => {{
  const {{ request }} = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== SCOPE_URL.origin || !url.pathname.startsWith(SCOPE_URL.pathname)) return;

  if (request.mode === 'navigate') {{
    event.respondWith((async () => {{
      const cache = await caches.open(CACHE_NAME);
      try {{
        return await cacheNetworkResponse(cache, request, await fetch(request));
      }} catch {{
        const exact = await cache.match(request, {{ ignoreSearch: true }});
        if (exact) return exact;
        if (isBookmarksNavigation(url)) {{
          const bookmarks = await cache.match(BOOKMARKS_URL.href);
          if (bookmarks) return bookmarks;
        }}
        return await cache.match(OFFLINE_URL) || Response.error();
      }}
    }})());
    return;
  }}

  if (['script', 'style', 'font', 'image'].includes(request.destination)) {{
    event.respondWith((async () => {{
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      return cacheNetworkResponse(cache, request, await fetch(request));
    }})());
  }}
}});
"""


def parse_service_worker_manifest(source: str) -> tuple[str, list[str]]:
    version_prefix = "const PRECACHE_VERSION = '"
    version_start = source.find(version_prefix)
    if version_start < 0:
        raise ValueError("Service worker precache version is missing")
    version_start += len(version_prefix)
    version_end = source.find("';", version_start)
    if version_end < 0:
        raise ValueError("Service worker precache version is malformed")
    manifest_start = source.find(MANIFEST_START)
    manifest_end = source.find(MANIFEST_END, manifest_start + len(MANIFEST_START))
    if manifest_start < 0 or manifest_end < 0:
        raise ValueError("Service worker precache manifest is missing")
    raw_manifest = source[manifest_start + len(MANIFEST_START):manifest_end].strip()
    paths = json.loads(raw_manifest)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("Service worker precache manifest is malformed")
    return source[version_start:version_end], paths


def generate_service_worker(dist_root: Path) -> dict[str, object]:
    paths = collect_precache_paths(dist_root)
    version = compute_precache_version(dist_root, paths)
    output = dist_root / "sw.js"
    output.write_text(render_service_worker(paths, version), encoding="utf-8")
    return {"version": version, "precache_files": len(paths), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the static offline service worker")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(json.dumps(generate_service_worker(args.dist), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())