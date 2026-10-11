import { describe, expect, it } from 'vitest';
import { cellVector, cosineMap, eigSym, pca3, toRgba } from './pca';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

describe('eigSym', () => {
  it('satisfies A v = lambda v on a random symmetric matrix', () => {
    const n = 7;
    const r = rng(1);
    const a = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) a[i * n + j] = a[j * n + i] = r() - 0.5;
    const { values, vectors } = eigSym(a, n);
    for (let e = 0; e < n; e++) {
      for (let i = 0; i < n; i++) {
        let av = 0;
        for (let j = 0; j < n; j++) av += a[i * n + j]! * vectors[j * n + e]!;
        expect(av).toBeCloseTo(values[e]! * vectors[i * n + e]!, 9);
      }
    }
  });
});

describe('pca3', () => {
  // Cells along a planted direction plus small noise: PC1 must recover it on both routes.
  function planted(C: number, N: number) {
    const r = rng(3);
    const dir = Array.from({ length: C }, () => r() - 0.5);
    const norm = Math.hypot(...dir);
    const data = new Float32Array(C * N);
    const t = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      t[k] = (r() - 0.5) * 10;
      for (let c = 0; c < C; c++) data[c * N + k] = (t[k]! * dir[c]!) / norm + (r() - 0.5) * 0.01;
    }
    return { data, t };
  }

  for (const [C, N, route] of [
    [6, 400, 'covariance'],
    [300, 60, 'Gram']
  ] as const) {
    it(`recovers a planted component (${route} route)`, () => {
      const { data, t } = planted(C, N);
      const mask = new Uint8Array(N).fill(1);
      const p = pca3(data, C, N, mask);
      expect(p.explained[0]).toBeGreaterThan(0.99);
      // Scores are centred; centre the planted coordinate before comparing.
      const tm = t.reduce((a, b) => a + b, 0) / N;
      let num = 0,
        da = 0,
        db = 0;
      for (let k = 0; k < N; k++) {
        num += p.scores[k * 3]! * (t[k]! - tm);
        da += p.scores[k * 3]! ** 2;
        db += (t[k]! - tm) ** 2;
      }
      expect(Math.abs(num / Math.sqrt(da * db))).toBeGreaterThan(0.999);
      const rgba = toRgba(p, N, mask);
      expect(rgba.length).toBe(N * 4);
      expect(rgba[3]).toBe(255);
    });
  }
});

describe('cosineMap', () => {
  it('is 1 at the reference cell and bounded by 1', () => {
    const C = 5,
      N = 50;
    const r = rng(9);
    const data = Float32Array.from({ length: C * N }, () => r() - 0.5);
    const map = cosineMap(data, C, N, cellVector(data, C, N, 17));
    expect(map[17]).toBeCloseTo(1, 6);
    for (const v of map) expect(Math.abs(v)).toBeLessThanOrEqual(1 + 1e-6);
  });
});
