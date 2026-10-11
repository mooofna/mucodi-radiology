import { describe, expect, it } from 'vitest';
import { bodyMask } from './body';

// A phantom at the in-plane scale of the model grid: air at -1000 HU, padding at -2048 HU outside
// the reconstruction circle, an elliptic trunk with two lungs (enclosed air), an arm lying next to
// the trunk, and a table under the patient that touches the back through a thin bridge.
function phantom(D = 6, H = 128, W = 128) {
  const hu = new Float32Array(D * H * W).fill(-1000);
  const at = (d: number, h: number, w: number) => (d * H + h) * W + w;
  const inside = (h: number, w: number, ch: number, cw: number, rh: number, rw: number) =>
    ((h - ch) / rh) ** 2 + ((w - cw) / rw) ** 2 <= 1;
  for (let d = 0; d < D; d++)
    for (let h = 0; h < H; h++)
      for (let w = 0; w < W; w++) {
        let v = -1000;
        if (Math.hypot(h - 63.5, w - 63.5) > 64) v = -2048;
        if (inside(h, w, 64, 64, 35, 50)) v = 40;
        if (inside(h, w, 60, 40, 15, 12) || inside(h, w, 60, 88, 15, 12)) v = -850;
        if (inside(h, w, 40, 10, 9, 9)) v = 40;
        if (h >= 104 && h <= 106 && w >= 10 && w <= 118) v = 300;
        if (h >= 99 && h <= 103 && w >= 63 && w <= 65) v = 100;
        hu[at(d, h, w)] = v;
      }
  return { hu, shape: [D, H, W] as [number, number, number], at };
}

describe('body mask', () => {
  const { hu, shape, at } = phantom();
  const m = bodyMask(hu, shape);

  it('keeps the trunk, its lungs and an arm next to it', () => {
    expect(m[at(3, 64, 64)]).toBe(1); // mediastinum
    expect(m[at(3, 60, 40)]).toBe(1); // lung, enclosed air
    expect(m[at(3, 40, 10)]).toBe(1); // arm
  });

  it('drops the table, the air around the patient and the padding', () => {
    expect(m[at(3, 105, 15)]).toBe(0); // table, far from the bridge
    expect(m[at(3, 105, 112)]).toBe(0);
    expect(m[at(3, 10, 64)]).toBe(0); // air above the patient
    expect(m[at(3, 0, 0)]).toBe(0); // padding corner
  });

  it('keeps a thin margin of air at the skin', () => {
    // Row 64: the trunk spans w = 14 to 114.
    expect(hu[at(1, 64, 13)]).toBe(-1000);
    expect(m[at(1, 64, 13)]).toBe(1);
    expect(m[at(1, 64, 4)]).toBe(0);
  });

  it('returns the shape of its input and an empty mask for a volume of air', () => {
    expect(m.length).toBe(hu.length);
    const air = bodyMask(new Float32Array(2 * 8 * 8).fill(-1000), [2, 8, 8]);
    expect(air.length).toBe(128);
    expect(air.every((v) => v === 0)).toBe(true);
  });
});
