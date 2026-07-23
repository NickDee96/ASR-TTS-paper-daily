// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
	site: 'https://nickdee96.github.io',
	base: '/ASR-TTS-paper-daily',
	output: 'static',
	trailingSlash: 'always',
	integrations: [react()],
});
