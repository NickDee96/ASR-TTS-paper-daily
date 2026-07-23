# Paper Explorer Frontend

Static Astro application for browsing the canonical ASR-TTS paper archive.

## Commands

Run from `site/`:

| Command | Action |
| :--- | :--- |
| `npm install` | Install dependencies. |
| `npm run dev` | Start the development server. |
| `npm run check` | Validate Astro and TypeScript source. |
| `npm run prepare:data` | Reconcile the legacy archive and generate canonical site data. |
| `npm run build:astro` | Build Astro routes without rebuilding data or the search index. |
| `npm run build:index` | Generate the Pagefind index from built paper pages. |
| `npm run build` | Prepare data, build all routes, index them, and verify parity. |
| `npm run preview` | Preview the production build. |

See [UI_DECISIONS.md](UI_DECISIONS.md) for the Astro, React, Fluent UI, and
project-owned CSS boundaries.
