// Port of nibabel 5.4.2 io_orientation / apply_orientation / inv_ornt_aff, i.e. what
// nib.as_closest_canonical does. Only axis permutations and flips: no resampling.
import type { Mat4 } from './affine';
import { matmul } from './affine';

/** For each stored axis i: [output (RAS) axis, +1 or -1]. */
export type Orientation = [[number, number], [number, number], [number, number]];

/** Jacobi eigen-decomposition of a symmetric 3x3 matrix: values and column eigenvectors. */
function eigSym3(m: number[][]): { values: number[]; vectors: number[][] } {
  const a = m.map((r) => r.slice());
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += a[p]![q]! ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 3; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k]![p]!,
            akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p]![k]!,
            aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k]![p]!,
            vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: [a[0]![0]!, a[1]![1]!, a[2]![2]!], vectors: v };
}

/** nibabel io_orientation(affine) for a 3D affine. */
export function ioOrientation(affine: Mat4): Orientation {
  // RZS / zooms -> RS; polar decomposition R = RS (RS^T RS)^(-1/2), the closest shear-free matrix.
  const rs: number[][] = [[], [], []];
  for (let j = 0; j < 3; j++) {
    let z = 0;
    for (let i = 0; i < 3; i++) z += affine[i]![j]! ** 2;
    z = Math.sqrt(z) || 1;
    for (let i = 0; i < 3; i++) rs[i]![j] = affine[i]![j]! / z;
  }
  const m: number[][] = [0, 1, 2].map((i) =>
    [0, 1, 2].map(
      (j) => rs[0]![i]! * rs[0]![j]! + rs[1]![i]! * rs[1]![j]! + rs[2]![i]! * rs[2]![j]!
    )
  );
  const { values, vectors } = eigSym3(m);
  const maxVal = Math.max(...values.map(Math.abs));
  const tol = Math.sqrt(maxVal) * 3 * 2.220446049250313e-16;
  const inv: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  for (let k = 0; k < 3; k++) {
    const sv = Math.sqrt(Math.max(values[k]!, 0));
    if (sv <= tol) continue; // rank-deficient direction: dropped, as nibabel's `keep`
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) inv[i]![j]! += (vectors[i]![k]! * vectors[j]![k]!) / sv;
  }
  const R: number[][] = [0, 1, 2].map((i) =>
    [0, 1, 2].map(
      (j) => rs[i]![0]! * inv[0]![j]! + rs[i]![1]! * inv[1]![j]! + rs[i]![2]! * inv[2]![j]!
    )
  );

  // Strongest input axis first (stable on ties), as nibabel 5.4.
  const strength = [0, 1, 2].map((j) => Math.max(...[0, 1, 2].map((i) => R[i]![j]! ** 2)));
  const order = [0, 1, 2].sort((x, y) => strength[y]! - strength[x]! || x - y);
  const ornt: number[][] = [
    [NaN, NaN],
    [NaN, NaN],
    [NaN, NaN]
  ];
  for (const j of order) {
    const col = [R[0]![j]!, R[1]![j]!, R[2]![j]!];
    if (col.every((v) => Math.abs(v) <= 1e-8)) continue;
    let out = 0;
    for (let i = 1; i < 3; i++) if (Math.abs(col[i]!) > Math.abs(col[out]!)) out = i;
    ornt[j] = [out, col[out]! < 0 ? -1 : 1];
    for (let k = 0; k < 3; k++) R[out]![k] = 0;
  }
  if (ornt.some((r) => Number.isNaN(r[0])))
    throw new Error('affine does not determine an orientation');
  return ornt as Orientation;
}

export interface Canonical {
  /** Canonical (RAS) array shape (x, y, z). */
  shape: [number, number, number];
  /** Canonical voxel index -> world mm (affine of as_closest_canonical). */
  affine: Mat4;
  /** Per canonical axis, the stored-array offset contributed by each coordinate (separable). */
  offsets: [Int32Array, Int32Array, Int32Array];
  /** Flat index into the stored (x-fastest) array for canonical voxel (cx, cy, cz). */
  index: (cx: number, cy: number, cz: number) => number;
}

/** as_closest_canonical: an index function into the original data plus the new affine. */
export function canonical(shape: [number, number, number], affine: Mat4): Canonical {
  const ornt = ioOrientation(affine);
  const outShape: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i++) outShape[ornt[i]![0]] = shape[i]!;
  // inv_ornt_aff: canonical voxel -> stored voxel.
  const t: Mat4 = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
  ];
  for (let i = 0; i < 3; i++) {
    const [k, flip] = ornt[i]!;
    t[i]![k] = flip;
    t[i]![3] = flip === -1 ? shape[i]! - 1 : 0;
  }
  const strides = [1, shape[0], shape[0] * shape[1]];
  // stored index along axis i = flip ? n_i - 1 - c[k] : c[k], with k = ornt[i][0]
  const offsets: [Int32Array, Int32Array, Int32Array] = [
    new Int32Array(outShape[0]),
    new Int32Array(outShape[1]),
    new Int32Array(outShape[2])
  ];
  for (let i = 0; i < 3; i++) {
    const [k, flip] = ornt[i]!;
    const table = offsets[k]!;
    for (let c = 0; c < table.length; c++) {
      table[c] = (flip === -1 ? shape[i]! - 1 - c : c) * strides[i]!;
    }
  }
  const [ox, oy, oz] = offsets;
  const index = (cx: number, cy: number, cz: number): number => ox[cx]! + oy[cy]! + oz[cz]!;
  return { shape: outShape, affine: matmul(affine, t), offsets, index };
}
