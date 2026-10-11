// Every page renders without errors and passes axe (serious and critical rules) in both colour
// schemes; spot checks that the pages carry the thesis numbers.
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

const PAGES = ['/', '/race', '/evidence', '/method', '/about'];

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  return errors;
}

async function seriousViolations(page: Page) {
  // The viewer canvases are WebGL images, not text; axe cannot judge them.
  const result = await new AxeBuilder({ page }).exclude('canvas').analyze();
  return result.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map(
      (v) =>
        `${v.id}: ${v.nodes
          .map((n) => n.target.join(' '))
          .slice(0, 3)
          .join(' | ')}`
    );
}

for (const path of PAGES) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`${path} in ${colorScheme} renders without errors and passes axe`, async ({ page }) => {
      await page.emulateMedia({ colorScheme });
      const errors = collectErrors(page);
      await page.goto(path);
      await expect(page.locator('h1')).toHaveCount(1);
      await page.waitForLoadState('networkidle');
      expect(await seriousViolations(page)).toEqual([]);
      expect(errors).toEqual([]);
    });
  }
}

test('the evidence page carries the thesis numbers', async ({ page }) => {
  // Regular expressions see the raw text, so every gap between words is \s+ (source line breaks).
  await page.goto('/evidence');
  await expect(page.getByText(/T2\s+reaches\s+70\.5\s+%/)).toBeVisible();
  await expect(
    page.getByText(/student\s+is\s+82\.6\s+times\s+faster\s+than\s+Pillar-0/)
  ).toBeVisible();
  await expect(
    page.getByText(/TMIN\s+reaches\s+68\.7\s+against\s+Curia-1's\s+65\.6/)
  ).toBeVisible();
  await expect(
    page.getByText(/1,296\s+endpoint,\s+model\s+and\s+probe\s+combinations/)
  ).toBeVisible();
});

test('the method tour steps through the figure panels', async ({ page }) => {
  await page.goto('/method');
  await page.getByRole('tab', { name: '(d)' }).click();
  await expect(page.getByRole('heading', { name: /pull to the positive/ })).toBeVisible();
  await expect(page.locator('math')).toBeVisible();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: '(e)' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('figure img')).toHaveAttribute('src', /mucodi_m2box\.svg$/);
  const tmin = page.getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'TMIN' }) });
  await expect(tmin).toContainText('0.87');
});
