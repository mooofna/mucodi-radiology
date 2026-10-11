// The network view: stage maps of T2 on the bundled scan, PCA colours and a similarity map.
import { expect, test } from '@playwright/test';

test('network view shows PCA colours and a similarity map', async ({ page }) => {
  await page.goto('/reading-room');
  await expect(page.getByRole('button', { name: 'Encode this scan' })).toBeEnabled({
    timeout: 60_000
  });
  await page.getByRole('tab', { name: 'Network view' }).click();
  await page.getByRole('button', { name: 'Compute the feature maps' }).click();
  const pca = page.getByText(/first three principal components/);
  await expect(pca).toBeVisible({ timeout: 120_000 });
  console.log((await pca.innerText()).replace(/\s+/g, ' '));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'test-results/network-pca.png' });
  // Move the stage slider to stage 3.
  await page
    .getByRole('slider')
    .first()
    .evaluate((el: HTMLInputElement) => {
      el.value = '3';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
  await expect(page.getByText('Stage 3', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Similarity to a point' }).click();
  const canvas = page.locator('.canvas canvas');
  const box = (await canvas.boundingBox())!;
  // Click inside the axial tile (bottom-left quadrant of the 2 x 2 layout).
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.72);
  await expect(page.getByText(/Cosine similarity of every cell/)).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'test-results/network-similarity.png' });
});
