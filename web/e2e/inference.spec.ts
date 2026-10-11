// The browser pipeline must reproduce the thesis pipeline: decode, window and ONNX Runtime Web
// on the bundled scan give the PyTorch golden embedding (cosine >= 0.9999).
import { expect, test, type Page } from '@playwright/test';

async function cosineShown(page: Page): Promise<number> {
  const match = page.getByText(/matches the thesis pipeline in PyTorch/);
  await expect(match).toBeVisible({ timeout: 120_000 });
  return Number((await match.innerText()).match(/cosine similarity\s+([0-9.]+)/)![1]);
}

test('TMIN encodes the bundled scan and matches the PyTorch golden', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/reading-room');
  const encode = page.getByRole('button', { name: 'Encode this scan' });
  await expect(encode).toBeEnabled({ timeout: 60_000 });
  await encode.click();
  const cosine = await cosineShown(page);
  expect(cosine).toBeGreaterThanOrEqual(0.9999);
  await page.getByText('Details', { exact: true }).click();
  const forward = await page.locator('.pipeline li', { hasText: 'Forward pass' }).innerText();
  console.log(`TMIN forward pass, 1 thread: ${forward.replace(/\s+/g, ' ')}; cosine ${cosine}`);
  expect(errors).toEqual([]);
});

test('the full native pipeline in the browser also matches the golden', async ({ page }) => {
  await page.goto('/reading-room');
  await expect(page.getByRole('button', { name: 'Encode this scan' })).toBeEnabled({
    timeout: 60_000
  });
  await page.getByText('Scan details', { exact: true }).click();
  await page.getByLabel(/Preprocess the original scan in this tab/).check();
  await expect(page.getByRole('button', { name: 'Encode this scan' })).toBeEnabled({
    timeout: 120_000
  });
  await page.getByRole('button', { name: 'Encode this scan' }).click();
  const cosine = await cosineShown(page);
  await page.getByText('Details', { exact: true }).click();
  await expect(page.locator('.pipeline li', { hasText: 'Resample' })).toBeVisible();
  const steps = (await page.locator('.pipeline').innerText()).replace(/\s+/g, ' ');
  console.log(`native pipeline: ${steps}; cosine ${cosine}`);
  expect(cosine).toBeGreaterThanOrEqual(0.9999);
});

test('the overview button opens the reading room and encodes', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Encode a scan on this computer' }).click();
  await expect(page).toHaveURL(/reading-room\?encode=1/);
  expect(await cosineShown(page)).toBeGreaterThanOrEqual(0.9999);
});
