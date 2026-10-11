// The model grid: 128 x 256 x 256 voxels (D, H, W) spanning the whole field of view, and where
// its voxels sit in world space. Mirrors demo/src/mucodi_demo/cases.py::grid_affine.
import type { Mat4 } from './affine';
import { matmul } from './affine';

export const GRID_SHAPE: [number, number, number] = [128, 256, 256];

/**
 * World affine of the model grid stored as NIfTI in (i, j, k) = (W, H, D) order, so that the
 * stored Fortran-order data equals the model's (D, H, W) C-order array. Model voxel (d, h, w)
 * samples canonical voxel ((h + .5) sx - .5, (w + .5) sy - .5, (d + .5) sz - .5), the
 * align_corners=False mapping of F.interpolate. Mirrors demo cases.grid_affine.
 */
export function gridAffine(canonicalAffine: Mat4, canonicalShape: [number, number, number]): Mat4 {
  const sx = canonicalShape[0] / GRID_SHAPE[1];
  const sy = canonicalShape[1] / GRID_SHAPE[2];
  const sz = canonicalShape[2] / GRID_SHAPE[0];
  const scale: Mat4 = [
    [0, sx, 0, 0.5 * sx - 0.5],
    [sy, 0, 0, 0.5 * sy - 0.5],
    [0, 0, sz, 0.5 * sz - 0.5],
    [0, 0, 0, 1]
  ];
  return matmul(canonicalAffine, scale);
}

/** NIfTI dims of the model grid file: (W, H, D). */
export const GRID_NIFTI_DIMS: [number, number, number] = [
  GRID_SHAPE[2],
  GRID_SHAPE[1],
  GRID_SHAPE[0]
];

/** Voxel size of the model grid in mm as (D, H, W). */
export function gridSpacing(gridAff: Mat4): [number, number, number] {
  const col = (j: number) => Math.hypot(gridAff[0]![j]!, gridAff[1]![j]!, gridAff[2]![j]!);
  return [col(2), col(1), col(0)];
}
