// Feature-map maths for the network view. A tap is a (C, D, H, W) tensor in C order: channel c of
// spatial cell k is data[c * N + k]. Everything here describes the representation; nothing here
// explains a prediction.

/** Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix (row-major Float64Array). */
export function eigSym(
  a: Float64Array,
  n: number,
  sweeps = 30
): { values: Float64Array; vectors: Float64Array } {
  const m = Float64Array.from(a);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p * n + q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m[p * n + q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (m[q * n + q]! - m[p * n + p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k * n + p]!,
            mkq = m[k * n + q]!;
          m[k * n + p] = c * mkp - s * mkq;
          m[k * n + q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p * n + k]!,
            mqk = m[q * n + k]!;
          m[p * n + k] = c * mpk - s * mqk;
          m[q * n + k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p]!,
            vkq = v[k * n + q]!;
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = m[i * n + i]!;
  return { values, vectors: v };
}

export interface Pca {
  /** N x 3 scores (cell-major). */
  scores: Float32Array;
  /** Share of the masked variance carried by each of the three components. */
  explained: [number, number, number];
}

/**
 * Top-3 principal components of the masked cells. With few channels the C x C covariance is
 * decomposed; with many channels and few cells, the N x N Gram matrix (same components).
 */
export function pca3(data: Float32Array, C: number, N: number, mask: Uint8Array): Pca {
  const idx: number[] = [];
  for (let k = 0; k < N; k++) if (mask[k]) idx.push(k);
  const M = idx.length;
  const mean = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    const base = c * N;
    for (const k of idx) s += data[base + k]!;
    mean[c] = M ? s / M : 0;
  }
  const scores = new Float32Array(N * 3);
  const explained: [number, number, number] = [0, 0, 0];
  if (M < 3) return { scores, explained };

  let total = 0;
  if (C <= M) {
    // Covariance route.
    const cov = new Float64Array(C * C);
    const x = new Float64Array(C);
    for (const k of idx) {
      for (let c = 0; c < C; c++) x[c] = data[c * N + k]! - mean[c]!;
      for (let i = 0; i < C; i++) {
        const xi = x[i]!;
        if (xi === 0) continue;
        const row = i * C;
        for (let j = i; j < C; j++) cov[row + j]! += xi * x[j]!;
      }
    }
    for (let i = 0; i < C; i++) for (let j = 0; j < i; j++) cov[i * C + j] = cov[j * C + i]!;
    const { values, vectors } = eigSym(cov, C);
    const order = Array.from(values.keys()).sort((a, b) => values[b]! - values[a]!);
    for (let i = 0; i < C; i++) total += Math.max(values[i]!, 0);
    for (let r = 0; r < 3 && r < C; r++) {
      const e = order[r]!;
      explained[r] = total > 0 ? Math.max(values[e]!, 0) / total : 0;
      for (let k = 0; k < N; k++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += (data[c * N + k]! - mean[c]!) * vectors[c * C + e]!;
        scores[k * 3 + r] = s;
      }
    }
  } else {
    // Gram route: eigenvectors u of X X^T give components X^T u.
    const G = new Float64Array(M * M);
    for (let a = 0; a < M; a++) {
      for (let b = a; b < M; b++) {
        let s = 0;
        const ka = idx[a]!,
          kb = idx[b]!;
        for (let c = 0; c < C; c++)
          s += (data[c * N + ka]! - mean[c]!) * (data[c * N + kb]! - mean[c]!);
        G[a * M + b] = s;
        G[b * M + a] = s;
      }
    }
    const { values, vectors } = eigSym(G, M);
    const order = Array.from(values.keys()).sort((a, b) => values[b]! - values[a]!);
    for (let i = 0; i < M; i++) total += Math.max(values[i]!, 0);
    for (let r = 0; r < 3; r++) {
      const e = order[r]!;
      explained[r] = total > 0 ? Math.max(values[e]!, 0) / total : 0;
      // Component direction w = X^T u / |X^T u|; scores for every cell (masked or not).
      const w = new Float64Array(C);
      for (let a = 0; a < M; a++) {
        const u = vectors[a * M + e]!;
        const k = idx[a]!;
        for (let c = 0; c < C; c++) w[c]! += (data[c * N + k]! - mean[c]!) * u;
      }
      let norm = 0;
      for (let c = 0; c < C; c++) norm += w[c]! ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let k = 0; k < N; k++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += ((data[c * N + k]! - mean[c]!) * w[c]!) / norm;
        scores[k * 3 + r] = s;
      }
    }
  }
  // Deterministic signs: positive skew over the masked cells.
  for (let r = 0; r < 3; r++) {
    let m3 = 0;
    for (const k of idx) m3 += scores[k * 3 + r]! ** 3;
    if (m3 < 0) for (let k = 0; k < N; k++) scores[k * 3 + r] = -scores[k * 3 + r]!;
  }
  return { scores, explained };
}

function percentile(sorted: Float32Array, p: number): number {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))));
  return sorted[i]!;
}

/** RGBA voxels: each component mapped to 0..255 by its 1st to 99th percentile over masked cells. */
export function toRgba(p: Pca, N: number, mask: Uint8Array, alpha = 255): Uint8Array {
  const out = new Uint8Array(N * 4);
  const ranges: [number, number][] = [];
  for (let r = 0; r < 3; r++) {
    const vals: number[] = [];
    for (let k = 0; k < N; k++) if (mask[k]) vals.push(p.scores[k * 3 + r]!);
    const sorted = Float32Array.from(vals).sort();
    ranges.push([percentile(sorted, 0.01), percentile(sorted, 0.99)]);
  }
  for (let k = 0; k < N; k++) {
    if (!mask[k]) continue;
    for (let r = 0; r < 3; r++) {
      const [lo, hi] = ranges[r]!;
      const t = hi > lo ? (p.scores[k * 3 + r]! - lo) / (hi - lo) : 0.5;
      out[k * 4 + r] = Math.round(255 * Math.min(1, Math.max(0, t)));
    }
    out[k * 4 + 3] = alpha;
  }
  return out;
}

/** Cosine similarity of every cell's feature vector to a reference vector. */
export function cosineMap(
  data: Float32Array,
  C: number,
  N: number,
  ref: Float64Array
): Float32Array {
  let rn = 0;
  for (let c = 0; c < C; c++) rn += ref[c]! ** 2;
  rn = Math.sqrt(rn) || 1;
  const dot = new Float64Array(N);
  const nn = new Float64Array(N);
  for (let c = 0; c < C; c++) {
    const r = ref[c]!;
    const base = c * N;
    for (let k = 0; k < N; k++) {
      const v = data[base + k]!;
      dot[k]! += v * r;
      nn[k]! += v * v;
    }
  }
  const out = new Float32Array(N);
  for (let k = 0; k < N; k++) out[k] = dot[k]! / (Math.sqrt(nn[k]!) * rn + 1e-12);
  return out;
}

export function cellVector(data: Float32Array, C: number, N: number, k: number): Float64Array {
  const v = new Float64Array(C);
  for (let c = 0; c < C; c++) v[c] = data[c * N + k]!;
  return v;
}
