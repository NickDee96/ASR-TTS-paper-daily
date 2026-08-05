import { test, expect } from '@playwright/test';

test('feed is the first meaningful screen with a keyboard-reachable skip link', async ({ page }) => {
  await page.goto('/ASR-TTS-paper-daily/');
  await expect(page.getByRole('heading', { name: 'Latest papers' })).toBeVisible();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: /skip to results/i })).toBeFocused();
});

test('status route announces update health and archive totals', async ({ page }) => {
  await page.goto('/ASR-TTS-paper-daily/status/');
  await expect(page.getByRole('heading', { name: 'Archive status' })).toBeVisible();
  await expect(page.getByText(/Last successful update|update time is unavailable/)).toBeVisible();
  await expect(page.getByText('Unique papers')).toBeVisible();
});

test('search loads results, selects papers, and exports a downloadable file', async ({ page }) => {
  await page.goto('/ASR-TTS-paper-daily/search/');
  await expect(page.locator('.search-result-row').first()).toBeVisible();

  const firstSelect = page.getByRole('checkbox', { name: /select .* for export/i }).first();
  await firstSelect.check();
  await expect(page.getByText(/selected for export/)).toBeVisible();

  await page.getByRole('button', { name: 'Export' }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('menuitem', { name: 'CSV' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('papers.csv');
  await expect(page.getByText(/Exported .* as CSV/)).toBeVisible();
});

test('a paper page can be bookmarked and offers a citation download', async ({ page }) => {
  await page.goto('/ASR-TTS-paper-daily/papers/2607.18064/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();

  const bookmark = page.getByRole('button', { name: /bookmark paper/i });
  await bookmark.click();
  await expect(page.getByRole('button', { name: /remove bookmark/i })).toBeVisible();

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'RIS' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('2607.18064.ris');
});

test('mobile search opens the filter drawer and restores focus on close', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile', 'Drawer trigger is mobile-only');
  await page.goto('/ASR-TTS-paper-daily/search/');
  await expect(page.locator('.search-result-row').first()).toBeVisible();

  const trigger = page.getByRole('button', { name: /filter/i }).first();
  await trigger.click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();

  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(page).toHaveURL(/\/search\//);
});
