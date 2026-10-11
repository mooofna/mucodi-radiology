// Full field of view trilinear resize, matching torch.nn.functional.interpolate(
//   mode="trilinear", align_corners=False) on CPU in float32, as used by the thesis evaluation
// (rate_eval/datasets/preprocess.py::_interpolate). Arithmetic is rounded to float32 after every
// operation and nested in PyTorch's order (D outer, H, W innermost).
import type { Canonical } from './orientation';

const f = Math.fround;

export interface AxisTable {
  i0: Int32Array;
  i1: Int32Array;
  l0: Float32Array;
  l1: Float32Array;
}

/** PyTorch area_pixel_compute_source_index + guard_index_and_lambda for one axis. */
export function axisTable(nIn: number, nOut: number): AxisTable {
  const scale = f(nIn / nOut);
  const i0 = new Int32Array(nOut);
  const i1 = new Int32Array(nOut);
  const l0 = new Float32Array(nOut);
  const l1 = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let src = f(f(scale * f(o + 0.5)) - 0.5);
    if (src < 0) src = 0;
    const idx = Math.min(Math.floor(src), nIn - 1);
    const lambda = Math.min(Math.max(f(src - idx), 0), 1);
    i0[o] = idx;
    i1[o] = idx + (idx < nIn - 1 ? 1 : 0);
    l1[o] = lambda;
    l0[o] = f(1 - lambda);
  }
  return { i0, i1, l0, l1 };
}

/**
 * Resize the canonical volume (x, y, z) to the model grid (D, H, W) = (z, x, y), i.e. the
 * `transpose(2, 0, 1)` of rate_eval's reader followed by the trilinear resize.
 * Returns a float32 array in (D, H, W) C order.
 */
export function resizeToGrid(
  data: Float32Array,
  can: Canonical,
  out: [number, number, number] = [128, 256, 256]
): Float32Array {
  const [nx, ny, nz] = can.shape;
  const [D, H, W] = out;
  const td = axisTable(nz, D);
  const th = axisTable(nx, H);
  const tw = axisTable(ny, W);
  const result = new Float32Array(D * H * W);
  const [offX, offY, offZ] = can.offsets;
  for (let d = 0; d < D; d++) {
    const z0 = offZ[td.i0[d]!]!,
      z1 = offZ[td.i1[d]!]!,
      wd0 = td.l0[d]!,
      wd1 = td.l1[d]!;
    for (let h = 0; h < H; h++) {
      const x0 = offX[th.i0[h]!]!,
        x1 = offX[th.i1[h]!]!,
        wh0 = th.l0[h]!,
        wh1 = th.l1[h]!;
      const a00 = z0 + x0,
        a01 = z0 + x1,
        a10 = z1 + x0,
        a11 = z1 + x1;
      const base = (d * H + h) * W;
      for (let w = 0; w < W; w++) {
        const y0 = offY[tw.i0[w]!]!,
          y1 = offY[tw.i1[w]!]!,
          ww0 = tw.l0[w]!,
          ww1 = tw.l1[w]!;
        const t00 = f(f(data[a00 + y0]! * ww0) + f(data[a00 + y1]! * ww1));
        const t01 = f(f(data[a01 + y0]! * ww0) + f(data[a01 + y1]! * ww1));
        const t10 = f(f(data[a10 + y0]! * ww0) + f(data[a10 + y1]! * ww1));
        const t11 = f(f(data[a11 + y0]! * ww0) + f(data[a11 + y1]! * ww1));
        const t0 = f(f(t00 * wh0) + f(t01 * wh1));
        const t1 = f(f(t10 * wh0) + f(t11 * wh1));
        result[base + w] = f(f(t0 * wd0) + f(t1 * wd1));
      }
    }
  }
  return result;
}
