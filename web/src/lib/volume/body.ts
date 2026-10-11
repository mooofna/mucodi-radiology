// The patient's body on the model grid, for display and for the network view (never for the model
// input). Dense tissue (above -500 HU) is eroded in each axial slice, which removes thin structures
// such as the CT table top and breaks the thin contacts between table and patient. The large
// connected regions that remain (the trunk, and arms lying next to it) are grown back into the
// dense tissue, air enclosed in an axial slice (lungs, airways, bowel) is filled in, and the result
// is dilated by one voxel so that the skin stays inside.

const DENSE_HU = -500;
// The mask is computed on a grid with half the in-plane resolution (2.8 mm cells on the model
// grid), four times faster; its boundary lies in the air around the patient, where the coarser
// grid is invisible.
const ERODE = 2; // cells, in-plane; the table top is thinner than this, the trunk much thicker
const KEEP_FRACTION = 0.02; // components at least this fraction of the largest are kept

type Shape = [number, number, number];

/** One in-plane step of erosion (4-neighbourhood) or dilation, restricted to `within` if given. */
function step(src: Uint8Array, [D, H, W]: Shape, grow: boolean, within?: Uint8Array): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let d = 0; d < D; d++)
    for (let h = 0; h < H; h++)
      for (let w = 0; w < W; w++) {
        const v = (d * H + h) * W + w;
        const left = w > 0 ? src[v - 1]! : 0;
        const right = w < W - 1 ? src[v + 1]! : 0;
        const up = h > 0 ? src[v - W]! : 0;
        const down = h < H - 1 ? src[v + W]! : 0;
        if (grow) {
          const on = src[v] || left || right || up || down;
          out[v] = on && (!within || within[v]) ? 1 : 0;
        } else {
          out[v] = src[v] && left && right && up && down ? 1 : 0;
        }
      }
  return out;
}

/** Labels the 6-connected components of `mask`; returns labels (1-based) and their sizes. */
function components(mask: Uint8Array, [D, H, W]: Shape): { labels: Int32Array; sizes: number[] } {
  const n = mask.length;
  const plane = H * W;
  const labels = new Int32Array(n);
  const sizes = [0];
  const queue = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    if (!mask[s] || labels[s]) continue;
    const id = sizes.length;
    let head = 0;
    let tail = 0;
    queue[tail++] = s;
    labels[s] = id;
    while (head < tail) {
      const v = queue[head++]!;
      const w = v % W;
      const h = Math.floor(v / W) % H;
      const d = Math.floor(v / plane);
      const visit = (u: number) => {
        if (mask[u] && !labels[u]) {
          labels[u] = id;
          queue[tail++] = u;
        }
      };
      if (w > 0) visit(v - 1);
      if (w < W - 1) visit(v + 1);
      if (h > 0) visit(v - W);
      if (h < H - 1) visit(v + W);
      if (d > 0) visit(v - plane);
      if (d < D - 1) visit(v + plane);
    }
    sizes.push(tail);
  }
  return { labels, sizes };
}

/** Mask (1 = body) of a (D, H, W) C-order HU volume, on the same grid. */
export function bodyMask(hu: Float32Array, [D, FH, FW]: Shape): Uint8Array {
  // A cell is dense if any of its 2 x 2 voxels is, so thin skin and bone survive the pooling.
  const H = Math.ceil(FH / 2);
  const W = Math.ceil(FW / 2);
  const shape: Shape = [D, H, W];
  const plane = H * W;
  const n = D * plane;
  const dense = new Uint8Array(n);
  for (let d = 0; d < D; d++)
    for (let fh = 0; fh < FH; fh++)
      for (let fw = 0; fw < FW; fw++)
        if (hu[(d * FH + fh) * FW + fw]! > DENSE_HU)
          dense[d * plane + (fh >> 1) * W + (fw >> 1)] = 1;

  // Erode, keep the large components, and grow them back inside the dense tissue.
  let core: Uint8Array = dense;
  for (let k = 0; k < ERODE; k++) core = step(core, shape, false);
  const { labels, sizes } = components(core, shape);
  const largest = Math.max(0, ...sizes.slice(1));
  if (largest === 0) return new Uint8Array(D * FH * FW);
  const keep = sizes.map((s, id) => id > 0 && s >= KEEP_FRACTION * largest);
  let body: Uint8Array = new Uint8Array(n);
  for (let i = 0; i < n; i++) body[i] = keep[labels[i]!] ? 1 : 0;
  for (let k = 0; k < ERODE + 1; k++) body = step(body, shape, true, dense);

  // Fill enclosed air slice by slice: flood the outside from the slice border; the rest is body.
  const outside = new Uint8Array(plane);
  const stack = new Int32Array(plane);
  for (let d = 0; d < D; d++) {
    const base = d * plane;
    outside.fill(0);
    let top = 0;
    const push = (p: number) => {
      if (!outside[p] && !body[base + p]) {
        outside[p] = 1;
        stack[top++] = p;
      }
    };
    for (let w = 0; w < W; w++) (push(w), push((H - 1) * W + w));
    for (let h = 0; h < H; h++) (push(h * W), push(h * W + W - 1));
    while (top > 0) {
      const p = stack[--top]!;
      const w = p % W;
      const h = (p - w) / W;
      if (w > 0) push(p - 1);
      if (w < W - 1) push(p + 1);
      if (h > 0) push(p - W);
      if (h < H - 1) push(p + W);
    }
    for (let p = 0; p < plane; p++) if (!outside[p]) body[base + p] = 1;
  }

  // A margin of two cells in the slice and two slices along the scan axis keeps the skin inside,
  // also where the body surface runs obliquely through the slices; it only adds air.
  const grown = step(step(body, shape, true), shape, true);
  let coarse: Uint8Array = grown;
  for (let k = 0; k < 2; k++) {
    const next = coarse.slice();
    for (let d = 0; d < D; d++)
      for (let p = 0; p < plane; p++)
        if (
          !coarse[d * plane + p] &&
          ((d > 0 && coarse[(d - 1) * plane + p]) || (d < D - 1 && coarse[(d + 1) * plane + p]))
        )
          next[d * plane + p] = 1;
    coarse = next;
  }
  const full = new Uint8Array(D * FH * FW);
  for (let d = 0; d < D; d++)
    for (let fh = 0; fh < FH; fh++)
      for (let fw = 0; fw < FW; fw++)
        full[(d * FH + fh) * FW + fw] = coarse[d * plane + (fh >> 1) * W + (fw >> 1)]!;
  return full;
}
