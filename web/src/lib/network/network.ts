// Page-side client for the network-view worker, plus the geometry that places feature cells.
import type { Mat4 } from '../volume/affine';
import { apply, invert, matmul } from '../volume/affine';
import type { NetRequest, NetResponse, TapGeom } from './network.worker';

export type { TapGeom };

let worker: Worker | null = null;
let next = 1;
const pending = new Map<number, (r: NetResponse) => void>();

type Body<T> = T extends unknown ? Omit<T, 'id'> : never;

export function call(msg: Body<NetRequest>, transfer: Transferable[] = []): Promise<NetResponse> {
  if (!worker) {
    worker = new Worker(new URL('./network.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<NetResponse>) => {
      pending.get(e.data.id)?.(e.data);
      pending.delete(e.data.id);
    };
  }
  const id = next++;
  return new Promise((resolve, reject) => {
    pending.set(id, (r) => (r.type === 'error' ? reject(new Error(r.message)) : resolve(r)));
    worker!.postMessage({ ...msg, id }, transfer);
  });
}

/**
 * World affine of a tap stored as NIfTI (W', H', D'): tap cell (a, b, c) sits on model-grid voxel
 * (w, h, d) = (s_w a + o_w, s_h b + o_h, s_d c + o_d); the grid affine takes it to world space.
 */
export function tapAffine(gridAffine: Mat4, g: TapGeom): Mat4 {
  const [sd, sh, sw] = g.stride;
  const [od, oh, ow] = g.offset;
  const t: Mat4 = [
    [sw, 0, 0, ow],
    [0, sh, 0, oh],
    [0, 0, sd, od],
    [0, 0, 0, 1]
  ];
  return matmul(gridAffine, t);
}

/** World mm -> nearest tap cell (d, h, w), clamped to the tap grid. */
export function cellAt(gridAffine: Mat4, g: TapGeom, mm: [number, number, number]): [number, number, number] {
  const [w, h, d] = apply(invert(gridAffine), mm);
  const [, D, H, W] = g.shape;
  const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, Math.round(v)));
  return [
    clamp((d - g.offset[0]) / g.stride[0], D),
    clamp((h - g.offset[1]) / g.stride[1], H),
    clamp((w - g.offset[2]) / g.stride[2], W)
  ];
}

/** World mm of a tap cell centre. */
export function cellCentre(gridAffine: Mat4, g: TapGeom, cell: [number, number, number]): [number, number, number] {
  const [d, h, w] = cell;
  return apply(gridAffine, [g.stride[2] * w + g.offset[2], g.stride[1] * h + g.offset[1], g.stride[0] * d + g.offset[0]]);
}
