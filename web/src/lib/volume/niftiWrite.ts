// Minimal NIfTI-1 writer for in-memory overlay volumes (float32 or RGBA32) handed to NiiVue.
import type { Mat4 } from './affine';

export type NiftiData =
  { kind: 'float32'; data: Float32Array } | { kind: 'rgba'; data: Uint8Array };

/** Writes a single-file NIfTI-1 (.nii) with sform and qform code 0/1 set from `affine`. */
export function writeNifti(
  dims: [number, number, number],
  affine: Mat4,
  payload: NiftiData,
  scl: { slope: number; inter: number } = { slope: 1, inter: 0 }
): ArrayBuffer {
  const voxOffset = 352;
  const bytesPer = payload.kind === 'float32' ? 4 : 4;
  const n = dims[0] * dims[1] * dims[2];
  const buf = new ArrayBuffer(voxOffset + n * bytesPer);
  const v = new DataView(buf);
  const le = true;
  v.setInt32(0, 348, le);
  v.setInt16(40, 3, le);
  v.setInt16(42, dims[0], le);
  v.setInt16(44, dims[1], le);
  v.setInt16(46, dims[2], le);
  v.setInt16(48, 1, le);
  v.setInt16(50, 1, le);
  v.setInt16(52, 1, le);
  v.setInt16(54, 1, le);
  v.setInt16(70, payload.kind === 'float32' ? 16 : 2304, le); // DT_FLOAT32 or DT_RGBA32
  v.setInt16(72, payload.kind === 'float32' ? 32 : 32, le);
  v.setFloat32(76, 1, le); // qfac
  for (let j = 0; j < 3; j++) {
    const z = Math.hypot(affine[0]![j]!, affine[1]![j]!, affine[2]![j]!);
    v.setFloat32(80 + j * 4, z, le);
  }
  v.setFloat32(108, voxOffset, le);
  v.setFloat32(112, scl.slope, le);
  v.setFloat32(116, scl.inter, le);
  v.setInt16(252, 0, le); // qform_code
  v.setInt16(254, 1, le); // sform_code (scanner)
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 4; c++) v.setFloat32(280 + r * 16 + c * 4, affine[r]![c]!, le);
  const magic = 'n+1\0';
  for (let i = 0; i < 4; i++) v.setUint8(344 + i, magic.charCodeAt(i));
  if (payload.kind === 'float32') new Float32Array(buf, voxOffset, n).set(payload.data);
  else new Uint8Array(buf, voxOffset, n * 4).set(payload.data);
  return buf;
}
