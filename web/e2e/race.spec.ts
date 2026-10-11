// The CPU race: a single-thread benchmark of the t-line plus B0, then a live race whose
// teacher lanes are projected from thesis Table 6.5.
import { expect, test } from '@playwright/test';

test('benchmark and race on one thread', async ({ page }) => {
  test.setTimeout(400_000);
  await page.goto('/race');
  const run = page.getByRole('button', { name: 'Run the benchmark' });
  await expect(run).toBeEnabled({ timeout: 60_000 });
  await run.click();
  await expect(page.getByText('Done', { exact: true })).toBeVisible({ timeout: 300_000 });
  const rows = await page.locator('table tbody tr').allInnerTexts();
  console.log('benchmark\n' + rows.map((r) => r.replace(/\s+/g, ' ')).join('\n'));
  expect(rows.length).toBeGreaterThanOrEqual(9);

  await page.getByLabel('Challenger').selectOption('t2');
  await page.getByRole('button', { name: 'Start the race' }).click();
  const verdict = page.getByText(/In the time Pillar-0 would need for one scan/);
  await expect(verdict).toBeVisible({ timeout: 200_000 });
  console.log((await verdict.innerText()).replace(/\s+/g, ' '));
  await page.getByRole('button', { name: 'Stop' }).click();
});
