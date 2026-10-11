// 4x4 affine helpers mirroring nibabel's Nifti1Header (get_sform, get_qform, get_base_affine)
// and rate_eval's degenerate-affine repair (datasets/affine_repair.py).

export type Mat4 = number[][];

export function identity(): Mat4 {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

export function matmul(a: Mat4, b: Mat4): Mat4 {
  const out: Mat4 = [];
  for (let i = 0; i < 4; i++) {
    const row: number[] = [];
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[i]![k]! * b[k]![j]!;
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

export function apply(a: Mat4, p: [number, number, number]): [number, number, number] {
  const out: number[] = [];
  for (let i = 0; i < 3; i++)
    out.push(a[i]![0]! * p[0] + a[i]![1]! * p[1] + a[i]![2]! * p[2] + a[i]![3]!);
  return [out[0]!, out[1]!, out[2]!];
}

export function invert(m: Mat4): Mat4 {
  // Affine inverse: invert the 3x3 linear part, then the translation.
  const a = m;
  const [a00, a01, a02] = [a[0]![0]!, a[0]![1]!, a[0]![2]!];
  const [a10, a11, a12] = [a[1]![0]!, a[1]![1]!, a[1]![2]!];
  const [a20, a21, a22] = [a[2]![0]!, a[2]![1]!, a[2]![2]!];
  const det =
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
  if (Math.abs(det) < 1e-12) throw new Error('singular affine');
  const r = [
    [(a11 * a22 - a12 * a21) / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det],
    [(a12 * a20 - a10 * a22) / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det],
    [(a10 * a21 - a11 * a20) / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det]
  ];
  const t = [a[0]![3]!, a[1]![3]!, a[2]![3]!];
  const out = identity();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) out[i]![j] = r[i]![j]!;
    out[i]![3] = -(r[i]![0]! * t[0]! + r[i]![1]! * t[1]! + r[i]![2]! * t[2]!);
  }
  return out;
}

export function sformAffine(srows: number[][]): Mat4 {
  const out = identity();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) out[i]![j] = srows[i]![j]!;
  return out;
}

/** nibabel get_qform(): quaternion (b, c, d) with w filled positive, zooms with qfac on z. */
export function qformAffine(
  bcd: [number, number, number],
  offset: [number, number, number],
  pixdim: [number, number, number],
  qfacRaw: number
): Mat4 {
  const [b, c, d] = bcd;
  const w2 = 1 - (b * b + c * c + d * d);
  const w = w2 < 0 ? 0 : Math.sqrt(w2);
  // quat2mat for (w, x, y, z) = (w, b, c, d)
  const nq = w * w + b * b + c * c + d * d;
  const s = nq < 1e-12 ? 0 : 2 / nq;
  const X = b * s,
    Y = c * s,
    Z = d * s;
  const wX = w * X,
    wY = w * Y,
    wZ = w * Z;
  const xX = b * X,
    xY = b * Y,
    xZ = b * Z;
  const yY = c * Y,
    yZ = c * Z,
    zZ = d * Z;
  const R =
    nq < 1e-12
      ? [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ]
      : [
          [1 - (yY + zZ), xY - wZ, xZ + wY],
          [xY + wZ, 1 - (xX + zZ), yZ - wX],
          [xZ - wY, yZ + wX, 1 - (xX + yY)]
        ];
  const qfac = qfacRaw === -1 ? -1 : 1;
  const vox = [Math.abs(pixdim[0]), Math.abs(pixdim[1]), Math.abs(pixdim[2]) * qfac];
  const out = identity();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) out[i]![j] = R[i]![j]! * vox[j]!;
    out[i]![3] = offset[i]!;
  }
  return out;
}

/** nibabel get_base_affine(): Analyze-style centred affine with the x axis flipped. */
export function baseAffine(
  shape: [number, number, number],
  pixdim: [number, number, number]
): Mat4 {
  const zooms = [-Math.abs(pixdim[0]), Math.abs(pixdim[1]), Math.abs(pixdim[2])];
  const out = identity();
  for (let i = 0; i < 3; i++) {
    out[i]![i] = zooms[i]!;
    out[i]![3] = -((shape[i]! - 1) / 2) * zooms[i]!;
  }
  return out;
}

/** rate_eval affine_repair.affine_is_degenerate. */
export function isDegenerate(a: Mat4): boolean {
  for (const row of a) for (const v of row) if (!Number.isFinite(v)) return true;
  const det =
    a[0]![0]! * (a[1]![1]! * a[2]![2]! - a[1]![2]! * a[2]![1]!) -
    a[0]![1]! * (a[1]![0]! * a[2]![2]! - a[1]![2]! * a[2]![0]!) +
    a[0]![2]! * (a[1]![0]! * a[2]![1]! - a[1]![1]! * a[2]![0]!);
  return Math.abs(det) < 1e-12;
}

/** rate_eval affine_repair._clean_affine: diagonal from header zooms, keeping finite signs. */
export function cleanAffine(a: Mat4, zooms: [number, number, number]): Mat4 {
  const out = identity();
  for (let i = 0; i < 3; i++) {
    const z = Number.isFinite(zooms[i]!) && zooms[i] !== 0 ? Math.abs(zooms[i]!) : 1;
    const d = a[i]![i]!;
    const sign = Number.isFinite(d) && d !== 0 ? Math.sign(d) : 1;
    out[i]![i] = sign * z;
  }
  return out;
}
