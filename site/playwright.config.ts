import { defineConfig, devices } from '@playwright/test';

// The preview server serves the built `dist/` under the GitHub Pages base path.
const PORT = Number(process.env.PREVIEW_PORT ?? 4321);
const HOST = `http://localhost:${PORT}`;
const BASE_PATH = '/ASR-TTS-paper-daily';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: HOST,
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } },
    },
    {
      name: 'mobile',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: `npm run preview -- --port ${PORT}`,
    url: `${HOST}${BASE_PATH}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
