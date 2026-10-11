/// <reference lib="webworker" />
// Holds the stage feature maps of one or more scans and turns them into overlay volumes.
import { cellVector, cosineMap, pca3, toRgba } from './pca';

export interface TapGeom {
  name: string;
  shape: [number, number, number, number]; // C, D, H, W
  stride: [number, number, number];
  offset: [number, number, number];
  rf: [number, number, number];
}

export type NetRequest =
  | {
      id: number;
      type: 'set';
      key: string;
      taps: { name: string; data: Float32Array }[];
      geometry: TapGeom[];
      gridHu: Float32Array;
      gridShape: [number, number, number];
    }
  | { id: number; type: 'pca'; key: string; tap: string; basisKey?: string }
  | { id: number; type: 'similarity'; key: string; tap: string; cell: [number, number, number]; targetKey?: string };

export type NetResponse =
  | { id: number; type: 'stored' }
  | { id: number; type: 'pca'; rgba: Uint8Array; explained: [number, number, number]; dims: [number, number, number] }
  | {
      id: number;
      type: 'similarity';
      map: Float32Array;
      dims: [number, number, number];
      best: [number, number, number];
      bestValue: number;
    }
  | { id: number; type: 'error'; message: string };

interface Scan {
  taps: Map<string, Float32Array>;
  geometry: Map<string, TapGeom>;
  masks: Map<string, Uint8Array>;
  gridHu: Float32Array;
  gridShape: [number, number, number];
  body: Uint8Array | null;
}

/**
 * Body mask on the model grid: within each axial slice, voxels between the first and last voxel
 * denser than -500 HU along both in-plane directions. This keeps the lungs (inside the chest)
 * and drops the air around the patient.
 */
function bodyMask(hu: Float32Array, [D, H, W]: [number, number, number]): Uint8Array {
  const rows = new Uint8Array(D * H * W);
  const cols = new Uint8Array(D * H * W);
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      const base = (d * H + h) * W;
      let first = -1,
        last = -1;
      for (let w = 0; w < W; w++)
        if (hu[base + w]! > -500) {
          if (first < 0) first = w;
          last = w;
        }
      for (let w = first; first >= 0 && w <= last; w++) rows[base + w] = 1;
    }
    for (let w = 0; w < W; w++) {
      let first = -1,
        last = -1;
      for (let h = 0; h < H; h++)
        if (hu[(d * H + h) * W + w]! > -500) {
          if (first < 0) first = h;
          last = h;
        }
      for (let h = first; first >= 0 && h <= last; h++) cols[(d * H + h) * W + w] = 1;
    }
  }
  for (let i = 0; i < rows.length; i++) rows[i] = rows[i]! & cols[i]!;
  return rows;
}

const scans = new Map<string, Scan>();

/** Cells whose centre voxel on the model grid lies inside the body mask. */
function mask(scan: Scan, g: TapGeom): Uint8Array {
  const hit = scan.masks.get(g.name);
  if (hit) return hit;
  scan.body ??= bodyMask(scan.gridHu, scan.gridShape);
  const [, D, H, W] = g.shape;
  const [GD, GH, GW] = scan.gridShape;
  const out = new Uint8Array(D * H * W);
  for (let d = 0; d < D; d++)
    for (let h = 0; h < H; h++)
      for (let w = 0; w < W; w++) {
        const zd = Math.min(GD - 1, Math.max(0, Math.round(g.stride[0] * d + g.offset[0])));
        const yh = Math.min(GH - 1, Math.max(0, Math.round(g.stride[1] * h + g.offset[1])));
        const xw = Math.min(GW - 1, Math.max(0, Math.round(g.stride[2] * w + g.offset[2])));
        out[(d * H + h) * W + w] = scan.body[(zd * GH + yh) * GW + xw]!;
      }
  scan.masks.set(g.name, out);
  return out;
}

function get(key: string, tap: string) {
  const scan = scans.get(key);
  const data = scan?.taps.get(tap);
  const g = scan?.geometry.get(tap);
  if (!scan || !data || !g) throw new Error(`no feature maps for ${key} ${tap}`);
  return { scan, data, g, C: g.shape[0], N: g.shape[1] * g.shape[2] * g.shape[3] };
}

function handle(msg: NetRequest): [NetResponse, Transferable[]] {
  switch (msg.type) {
    case 'set': {
      scans.set(msg.key, {
        taps: new Map(msg.taps.map((t) => [t.name, t.data])),
        geometry: new Map(msg.geometry.map((g) => [g.name, g])),
        masks: new Map(),
        gridHu: msg.gridHu,
        gridShape: msg.gridShape,
        body: null
      });
      return [{ id: msg.id, type: 'stored' }, []];
    }
    case 'pca': {
      const { scan, data, g, C, N } = get(msg.key, msg.tap);
      const m = mask(scan, g);
      const p = pca3(data, C, N, m);
      const rgba = toRgba(p, N, m, 235);
      const [, D, H, W] = g.shape;
      return [{ id: msg.id, type: 'pca', rgba, explained: p.explained, dims: [W, H, D] }, [rgba.buffer]];
    }
    case 'similarity': {
      const src = get(msg.key, msg.tap);
      const [, D, H, W] = src.g.shape;
      const [d, h, w] = msg.cell;
      const ref = cellVector(src.data, src.C, src.N, (d * H + h) * W + w);
      const target = msg.targetKey ? get(msg.targetKey, msg.tap) : src;
      const map = cosineMap(target.data, target.C, target.N, ref);
      const m = mask(target.scan, target.g);
      // Outside the body nothing is shown: those cells fall below any display threshold.
      for (let k = 0; k < target.N; k++) if (!m[k]) map[k] = -1;
      let best = 0;
      for (let k = 0; k < target.N; k++) if (map[k]! > map[best]!) best = k;
      const [, TD, TH, TW] = target.g.shape;
      const bd = Math.floor(best / (TH * TW));
      const bh = Math.floor((best % (TH * TW)) / TW);
      const bw = best % TW;
      return [
        { id: msg.id, type: 'similarity', map, dims: [TW, TH, TD], best: [bd, bh, bw], bestValue: map[best]! },
        [map.buffer]
      ];
    }
  }
}

self.onmessage = (event: MessageEvent<NetRequest>) => {
  try {
    const [res, transfer] = handle(event.data);
    (self as unknown as DedicatedWorkerGlobalScope).postMessage(res, transfer);
  } catch (err) {
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({
      id: event.data.id,
      type: 'error',
      message: err instanceof Error ? err.message : String(err)
    } satisfies NetResponse);
  }
};
