// The browser pipeline must reproduce the thesis pipeline: decode, window and ONNX Runtime Web
// on the bundled scan give the PyTorch golden embedding (cosine >= 0.9999).
import { expect, test } from '@playwright/test';

test('TMIN encodes the bundled scan and matches the PyTorch golden', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  const encode = page.getByRole('button', { name: 'Encode on this CPU' });
  await expect(encode).toBeEnabled({ timeout: 60_000 });
  await encode.click();
  const match = page.getByText(/Matches the thesis pipeline in PyTorch: cosine/);
  await expect(match).toBeVisible({ timeout: 120_000 });
  const cosine = Number((await match.innerText()).match(/cosine\s+([0-9.]+)/)![1]);
  expect(cosine).toBeGreaterThanOrEqual(0.9999);
  const forward = await page.locator('.pipeline li', { hasText: 'Forward pass' }).innerText();
  console.log(`TMIN forward pass, 1 thread: ${forward.replace(/\s+/g, ' ')}; cosine ${cosine}`);
  expect(errors).toEqual([]);
});

test('the full native pipeline in the browser also matches the golden', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Encode on this CPU' })).toBeEnabled({
    timeout: 60_000
  });
  await page.getByLabel(/Run the full preprocessing in this tab/).check();
  await expect(page.locator('.pipeline li', { hasText: 'Resample' })).toBeVisible({
    timeout: 120_000
  });
  await page.getByRole('button', { name: 'Encode on this CPU' }).click();
  const match = page.getByText(/Matches the thesis pipeline in PyTorch: cosine/);
  await expect(match).toBeVisible({ timeout: 120_000 });
  const cosine = Number((await match.innerText()).match(/cosine\s+([0-9.]+)/)![1]);
  const steps = (await page.locator('.pipeline').innerText()).replace(/\s+/g, ' ');
  console.log(`native pipeline: ${steps}; cosine ${cosine}`);
  expect(cosine).toBeGreaterThanOrEqual(0.9999);
});
