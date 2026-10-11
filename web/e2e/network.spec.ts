// The network view: stage maps of T2 on the bundled scan, PCA colours and a similarity map.
import { expect, test } from '@playwright/test';

test('network view shows PCA colours and a similarity map', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Encode on this CPU' })).toBeEnabled({
    timeout: 60_000
  });
  await page.getByRole('button', { name: 'Compute stage maps' }).click();
  await expect(page.getByText(/first three principal components/)).toBeVisible({
    timeout: 120_000
  });
  console.log(
    (await page.getByText(/first three principal components/).innerText()).replace(/\s+/g, ' ')
  );
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'test-results/network-pca.png' });
  await page.getByRole('button', { name: '2', exact: true }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'test-results/network-pca-stage2.png' });
  await page.getByRole('button', { name: 'Similarity to a cell' }).click();
  const canvas = page.locator('.canvas canvas');
  const box = (await canvas.boundingBox())!;
  // Click inside the axial tile (bottom-left quadrant of the 2 x 2 layout).
  await page.mouse.click(box.x + box.width * 0.35, box.y + box.height * 0.72);
  await expect(page.getByText(/cosine similarity of every cell/)).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'test-results/network-similarity.png' });
  await page.getByRole('button', { name: '3D' }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'test-results/network-3d.png' });
});
