// The student's four HU windows, in the channel order of
// rate-evals/configs/model/mucokd_student3d.yaml. Values from rve.ANATOMICAL_WINDOWS["CT"]
// (YalaLab/rave); formula from rate_eval/models/common.py: clamp((HU - (c - w/2)) / (w + 1e-8), 0, 1)
// in float32, where the 1e-8 vanishes.

export interface Window {
  name: string;
  label: string;
  center: number;
  width: number;
}

export const WINDOWS: readonly Window[] = [
  { name: 'lung', label: 'Lung', center: -600, width: 1500 },
  { name: 'mediastinum', label: 'Mediastinum', center: 50, width: 400 },
  { name: 'soft_tissue', label: 'Soft tissue', center: 50, width: 350 },
  { name: 'bone', label: 'Bone', center: 400, width: 1800 }
];

const f = Math.fround;

/** (D, H, W) HU grid -> (1, 4, D, H, W) model input, written into one Float32Array. */
export function applyWindows(grid: Float32Array, out?: Float32Array): Float32Array {
  const n = grid.length;
  const input = out ?? new Float32Array(n * WINDOWS.length);
  const low = WINDOWS.map((w) => w.center - w.width / 2);
  const width = WINDOWS.map((w) => f(w.width + 1e-8));
  const [l0, l1, l2, l3] = low as [number, number, number, number];
  const [w0, w1, w2, w3] = width as [number, number, number, number];
  const clamp = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  for (let i = 0; i < n; i++) {
    const hu = grid[i]!;
    input[i] = clamp(f(f(hu - l0) / w0));
    input[n + i] = clamp(f(f(hu - l1) / w1));
    input[2 * n + i] = clamp(f(f(hu - l2) / w2));
    input[3 * n + i] = clamp(f(f(hu - l3) / w3));
  }
  return input;
}
