/// <reference lib="webworker" />
// Preprocessing off the main thread: the thesis evaluation path, step by step, with timings.
//  * 'grid': a bundled model-grid file (already resized by the thesis code) -> windows.
//  * 'native': any NIfTI -> canonical RAS -> full field of view resize -> windows.
import type { Mat4 } from './affine';
import { cleanAffine, isDegenerate } from './affine';
import { readNifti } from './nifti';
import { canonical } from './orientation';
import { resizeToGrid } from './resample';
import { applyWindows } from './window';
import { bodyMask } from './body';
import { GRID_SHAPE, gridAffine } from './grid';

export type PrepRequest = {
  id: number;
  kind: 'grid' | 'native';
  buffer: ArrayBuffer;
  name: string;
};

export interface PrepResult {
  id: number;
  /** (1, 4, 128, 256, 256) model input. */
  input: Float32Array;
  /** (128, 256, 256) HU on the model grid, (D, H, W) C order. */
  gridHu: Float32Array;
  /** World affine of the model grid stored as NIfTI (i, j, k) = (W, H, D). */
  gridAffine: Mat4;
  /** (128, 256, 256) body mask on the model grid, for display only (see body.ts). */
  body: Uint8Array;
  native: {
    shape: [number, number, number];
    zooms: [number, number, number];
    affine: Mat4;
    datatype: string;
  } | null;
  stats: { minHu: number; maxHu: number; extentMm: [number, number, number] | null };
  timings: { name: string; ms: number }[];
}

type PrepError = { id: number; error: string };

function huStats(a: Float32Array) {
  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { minHu: min, maxHu: max };
}

async function prepare(req: PrepRequest): Promise<PrepResult> {
  const timings: { name: string; ms: number }[] = [];
  let t = performance.now();
  const lap = (name: string) => {
    const now = performance.now();
    timings.push({ name, ms: now - t });
    t = now;
  };

  const vol = await readNifti(req.buffer);
  lap('Decode');

  if (req.kind === 'grid') {
    // Stored as NIfTI (W, H, D) with HU x 10 and scl_slope 0.1: its data already is (D, H, W).
    const [W, H, D] = vol.shape;
    if (D !== GRID_SHAPE[0] || H !== GRID_SHAPE[1] || W !== GRID_SHAPE[2])
      throw new Error('not a model-grid file');
    const gridHu = vol.data;
    const input = applyWindows(gridHu);
    lap('Window');
    return {
      id: req.id,
      input,
      gridHu,
      gridAffine: vol.affine,
      body: bodyMask(gridHu, GRID_SHAPE),
      native: null,
      stats: { ...huStats(gridHu), extentMm: null },
      timings
    };
  }

  const affine = isDegenerate(vol.affine) ? cleanAffine(vol.affine, vol.zooms) : vol.affine;
  const can = canonical(vol.shape, affine);
  lap('Reorient');
  const gridHu = resizeToGrid(vol.data, can, GRID_SHAPE);
  lap('Resample');
  const input = applyWindows(gridHu);
  lap('Window');
  const extentMm = can.shape.map((n, i) => {
    const col = [can.affine[0]![i]!, can.affine[1]![i]!, can.affine[2]![i]!];
    return n * Math.hypot(col[0]!, col[1]!, col[2]!);
  }) as [number, number, number];
  return {
    id: req.id,
    input,
    gridHu,
    gridAffine: gridAffine(can.affine, can.shape),
    body: bodyMask(gridHu, GRID_SHAPE),
    native: { shape: vol.shape, zooms: vol.zooms, affine: vol.affine, datatype: vol.datatype },
    stats: { ...huStats(vol.data), extentMm },
    timings
  };
}

self.onmessage = (event: MessageEvent<PrepRequest>) => {
  prepare(event.data)
    .then((r) =>
      (self as unknown as DedicatedWorkerGlobalScope).postMessage(r, [
        r.input.buffer,
        r.gridHu.buffer,
        r.body.buffer
      ])
    )
    .catch((err: unknown) =>
      (self as unknown as DedicatedWorkerGlobalScope).postMessage({
        id: event.data.id,
        error: err instanceof Error ? err.message : String(err)
      } satisfies PrepError)
    );
};
