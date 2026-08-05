import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// Every primary route must be free of serious and critical accessibility issues.
const ROUTES: Array<{ name: string; path: string; ready: string }> = [
  { name: 'feed', path: '/ASR-TTS-paper-daily/', ready: 'Latest papers' },
  { name: 'search', path: '/ASR-TTS-paper-daily/search/', ready: 'papers' },
  { name: 'status', path: '/ASR-TTS-paper-daily/status/', ready: 'Archive status' },
  { name: 'bookmarks', path: '/ASR-TTS-paper-daily/bookmarks/', ready: 'Bookmarked papers' },
  { name: 'paper', path: '/ASR-TTS-paper-daily/papers/2607.18064/', ready: 'Paper metadata' },
];

for (const route of ROUTES) {
  test(`no serious or critical accessibility issues on the ${route.name} route`, async ({ page }) => {
    await page.goto(route.path);
    await page.getByText(route.ready, { exact: false }).first().waitFor();

    // Tabster (Fluent UI's focus manager) injects aria-hidden focus sentinels with
    // tabindex="0"; they are framework internals invisible to assistive tech, so they
    // are excluded from the authored-content scan.
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .exclude('[data-tabster-dummy]')
      .analyze();
    const blocking = results.violations.filter(
      (violation) => violation.impact === 'serious' || violation.impact === 'critical',
    );

    expect(
      blocking,
      blocking.map((violation) => `${violation.id}: ${violation.help}`).join('\n'),
    ).toEqual([]);
  });
}
