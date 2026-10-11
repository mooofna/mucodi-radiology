// NIfTI-1/2 reading that mirrors nibabel 5.4: best affine, scaling and data layout.
// The thesis evaluation path reads scans with nibabel `get_fdata()` (scaling applied in float64)
// and casts to float32, so this module returns float32 HU computed the same way.
import * as nifti from 'nifti-reader-js';
import type { Mat4 } from './affine';
import { baseAffine, qformAffine, sformAffine } from './affine';

export interface Volume {
  /** Voxel counts along the stored axes (x fastest), as nibabel's array shape. */
  shape: [number, number, number];
  /** float32 values after scl_slope/scl_inter, x fastest (Fortran order, as stored). */
  data: Float32Array;
  /** nibabel get_best_affine(): voxel index -> world mm. */
  affine: Mat4;
  zooms: [number, number, number];
  datatype: string;
}

export class NiftiError extends Error {}

const TYPES: Record<number, { name: string; bytes: number }> = {
  2: { name: 'uint8', bytes: 1 },
  4: { name: 'int16', bytes: 2 },
  8: { name: 'int32', bytes: 4 },
  16: { name: 'float32', bytes: 4 },
  64: { name: 'float64', bytes: 8 },
  256: { name: 'int8', bytes: 1 },
  512: { name: 'uint16', bytes: 2 },
  768: { name: 'uint32', bytes: 4 }
};

export async function gunzipIfNeeded(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  if (head[0] !== 0x1f || head[1] !== 0x8b) return buffer;
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

function reader(view: DataView, code: number, little: boolean): (offset: number) => number {
  switch (code) {
    case 2:
      return (o) => view.getUint8(o);
    case 256:
      return (o) => view.getInt8(o);
    case 4:
      return (o) => view.getInt16(o, little);
    case 512:
      return (o) => view.getUint16(o, little);
    case 8:
      return (o) => view.getInt32(o, little);
    case 768:
      return (o) => view.getUint32(o, little);
    case 16:
      return (o) => view.getFloat32(o, little);
    case 64:
      return (o) => view.getFloat64(o, little);
    default:
      throw new NiftiError(`unsupported NIfTI datatype ${code}`);
  }
}

function typedView(buffer: ArrayBuffer, offset: number, n: number, code: number): ArrayLike<number> | null {
  switch (code) {
    case 2:
      return new Uint8Array(buffer, offset, n);
    case 256:
      return new Int8Array(buffer, offset, n);
    case 4:
      return new Int16Array(buffer, offset, n);
    case 512:
      return new Uint16Array(buffer, offset, n);
    case 8:
      return new Int32Array(buffer, offset, n);
    case 768:
      return new Uint32Array(buffer, offset, n);
    case 16:
      return new Float32Array(buffer, offset, n);
    case 64:
      return new Float64Array(buffer, offset, n);
    default:
      return null;
  }
}

/** srow_x/y/z straight from the header (nifti-reader-js prefers the qform when both are set). */
function readSrows(buffer: ArrayBuffer, little: boolean): number[][] {
  const view = new DataView(buffer);
  const isV2 = view.getInt32(0, little) === 540;
  const rows: number[][] = [];
  for (let r = 0; r < 3; r++) {
    const row: number[] = [];
    for (let c = 0; c < 4; c++) {
      row.push(isV2 ? view.getFloat64(400 + r * 32 + c * 8, little) : view.getFloat32(280 + r * 16 + c * 4, little));
    }
    rows.push(row);
  }
  return rows;
}

/** Parse an (optionally gzipped) NIfTI file into float32 values and the nibabel affine. */
export async function readNifti(input: ArrayBuffer): Promise<Volume> {
  const buffer = await gunzipIfNeeded(input);
  if (!nifti.isNIFTI(buffer)) throw new NiftiError('not a NIfTI-1 or NIfTI-2 file');
  const hdr = nifti.readHeader(buffer);
  const type = TYPES[hdr.datatypeCode];
  if (!type) throw new NiftiError(`unsupported NIfTI datatype ${hdr.datatypeCode} (RGB or complex?)`);

  const ndim = hdr.dims[0] ?? 0;
  const nx = hdr.dims[1] ?? 1;
  const ny = ndim >= 2 ? (hdr.dims[2] ?? 1) : 1;
  const nz = ndim >= 3 ? (hdr.dims[3] ?? 1) : 1;
  for (let d = 4; d <= ndim; d++) {
    if ((hdr.dims[d] ?? 1) > 1) throw new NiftiError('4D or higher volumes are not supported');
  }
  if (ndim < 3 || nz < 2) throw new NiftiError('expected a 3D volume');

  const n = nx * ny * nz;
  const offset = Math.round(hdr.vox_offset);
  if (offset + n * type.bytes > buffer.byteLength) throw new NiftiError('file is truncated');
  const view = new DataView(buffer, offset, n * type.bytes);
  const read = reader(view, hdr.datatypeCode, hdr.littleEndian);
  const fast = hdr.littleEndian && offset % type.bytes === 0 ? typedView(buffer, offset, n, hdr.datatypeCode) : null;

  // nibabel: slope 0 or non-finite means "no scaling".
  let slope = hdr.scl_slope;
  let inter = hdr.scl_inter;
  if (!Number.isFinite(slope) || slope === 0) {
    slope = 1;
    inter = 0;
  }
  if (!Number.isFinite(inter)) inter = 0;
  const data = new Float32Array(n);
  const scaled = slope !== 1 || inter !== 0;
  if (fast) {
    // float64 arithmetic, then rounding to float32 (get_fdata().astype(np.float32)).
    if (scaled) for (let i = 0; i < n; i++) data[i] = fast[i]! * slope + inter;
    else data.set(fast);
  } else {
    for (let i = 0, o = 0; i < n; i++, o += type.bytes) {
      const raw = read(o);
      data[i] = scaled ? raw * slope + inter : raw;
    }
  }

  const pix = hdr.pixDims;
  const zooms: [number, number, number] = [Math.abs(pix[1] ?? 1), Math.abs(pix[2] ?? 1), Math.abs(pix[3] ?? 1)];
  let affine: Mat4;
  if (hdr.sform_code !== 0) {
    affine = sformAffine(readSrows(buffer, hdr.littleEndian));
  } else if (hdr.qform_code !== 0) {
    affine = qformAffine(
      [hdr.quatern_b, hdr.quatern_c, hdr.quatern_d],
      [hdr.qoffset_x, hdr.qoffset_y, hdr.qoffset_z],
      [pix[1] ?? 1, pix[2] ?? 1, pix[3] ?? 1],
      pix[0] ?? 1
    );
  } else {
    affine = baseAffine([nx, ny, nz], [pix[1] ?? 1, pix[2] ?? 1, pix[3] ?? 1]);
  }
  return { shape: [nx, ny, nz], data, affine, zooms, datatype: type.name };
}
