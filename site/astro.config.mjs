// @ts-check
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';

// Sitemap lastmod per paper route uses the canonical archive's real "updated"
// date when it is available at build time; falls back to Astro's file-mtime
// default when the archive has not been prepared yet (e.g. `astro check`).
const canonicalPath = fileURLToPath(new URL('.generated/canonical.json', import.meta.url));
const paperUpdatedAt = new Map();
if (existsSync(canonicalPath)) {
	const records = JSON.parse(readFileSync(canonicalPath, 'utf8'));
	for (const record of Object.values(records)) {
		const updated = record?.updated || record?.published;
		if (record?.id && updated) paperUpdatedAt.set(record.id, updated);
	}
}

// https://astro.build/config
export default defineConfig({
	site: 'https://nickdee96.github.io',
	base: '/ASR-TTS-paper-daily',
	output: 'static',
	trailingSlash: 'always',
	integrations: [
		react(),
		sitemap({
			filter: (page) => !page.includes('/offline/'),
			serialize(item) {
				const match = /\/papers\/([^/]+)\/$/.exec(item.url);
				const updated = match ? paperUpdatedAt.get(decodeURIComponent(match[1])) : undefined;
				if (updated) item.lastmod = updated;
				return item;
			},
		}),
	],
});
