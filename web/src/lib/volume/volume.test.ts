// The browser preprocessing must reproduce nibabel and the thesis code (goldens written by
// `uv run mucodi-demo web-goldens`).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { readNifti } from './nifti';
import { canonical, ioOrientation } from './orientation';
import { resizeToGrid } from './resample';
import { applyWindows } from './window';

const dir = join(import.meta.dirname, '__fixtures__');
const load = <T>(name: string): T => JSON.parse(readFileSync(join(dir, name), 'utf8')) as T;

function closeMat(a: number[][], b: number[][], tol = 1e-6) {
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(a[i]![j]!).toBeCloseTo(b[i]![j]!, -Math.log10(tol));
}

describe('io_orientation and as_closest_canonical (nibabel 5.4.2)', () => {
  const cases = load<
    { shape: [number, number, number]; affine: number[][]; ornt: number[][]; canonical_shape: number[]; canonical_affine: number[][] }[]
  >('orientation.json');

  it('covers all 48 axis codes plus oblique and sheared affines', () => {
    expect(cases.length).toBe(60);
  });

  it.each(cases.map((c, i) => [i, c] as const))('case %i', (_i, c) => {
    expect(ioOrientation(c.affine)).toEqual(c.ornt);
    const can = canonical(c.shape, c.affine);
    expect(can.shape).toEqual(c.canonical_shape);
    closeMat(can.affine, c.canonical_affine);
  });
});

describe('NIfTI reading (datatypes, endianness, qform/sform, scaling, NIfTI-2)', () => {
  const cases = load<
    { file: string; best_affine: number[][]; canonical_shape: number[]; canonical_affine: number[][]; canonical_fortran: number[] }[]
  >('nifti.json');

  it.each(cases.map((c) => [c.file, c] as const))('%s', async (_name, c) => {
    const bytes = readFileSync(join(dir, 'nifti', c.file));
    const vol = await readNifti(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    closeMat(vol.affine, c.best_affine, 1e-5);
    const can = canonical(vol.shape, vol.affine);
    expect(can.shape).toEqual(c.canonical_shape);
    closeMat(can.affine, c.canonical_affine, 1e-5);
    const [nx, ny, nz] = can.shape;
    let maxDiff = 0;
    for (let z = 0; z < nz; z++)
      for (let y = 0; y < ny; y++)
        for (let x = 0; x < nx; x++) {
          const got = vol.data[can.index(x, y, z)]!;
          const want = c.canonical_fortran[x + nx * (y + ny * z)]!;
          maxDiff = Math.max(maxDiff, Math.abs(got - want));
        }
    expect(maxDiff).toBe(0);
  });
});

describe('trilinear resize (torch F.interpolate, align_corners=False, float32)', () => {
  const cases = load<
    { shape_xyz: [number, number, number]; data_xyz_fortran: number[]; target_dhw: [number, number, number]; expected_dhw: number[] }[]
  >('resize.json');

  it.each(cases.map((c) => [c.shape_xyz.join('x') + ' -> ' + c.target_dhw.join('x'), c] as const))('%s', (_n, c) => {
    const identity = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
    const can = canonical(c.shape_xyz, identity);
    const out = resizeToGrid(Float32Array.from(c.data_xyz_fortran), can, c.target_dhw);
    let maxDiff = 0;
    for (let i = 0; i < out.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i]! - c.expected_dhw[i]!));
    // HU values up to ~1500: a float32 ulp is ~1.2e-4; allow two ulps of contraction differences.
    expect(maxDiff).toBeLessThanOrEqual(3e-4);
  });
});

describe('HU windows (rate_eval models/common.py)', () => {
  const golden = load<{ hu: number[]; expected: number[][] }>('windows.json');

  it('matches the thesis windowing in channel order lung, mediastinum, soft tissue, bone', () => {
    const n = golden.hu.length;
    const out = applyWindows(Float32Array.from(golden.hu));
    let maxDiff = 0;
    for (let c = 0; c < 4; c++)
      for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(out[c * n + i]! - golden.expected[c]![i]!));
    expect(maxDiff).toBeLessThanOrEqual(1e-6);
  });
});
