// Projecting the teachers' CPU time onto the visitor's device. The teachers are never run in the
// browser: their time is the thesis's single-thread Xeon time (Table 6.5) scaled by how much
// slower or faster this device ran the students than the thesis Xeon did.

export interface Measurement {
  rung: string;
  liveMs: number;
  thesisMs: number;
}

export interface DeviceFactor {
  factor: number;
  min: number;
  max: number;
  n: number;
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** f = median over measured rungs of live / thesis, with the min-max range. */
export function deviceFactor(ms: Measurement[]): DeviceFactor | null {
  const ratios = ms.filter((m) => m.liveMs > 0 && m.thesisMs > 0).map((m) => m.liveMs / m.thesisMs);
  if (!ratios.length) return null;
  return { factor: median(ratios), min: Math.min(...ratios), max: Math.max(...ratios), n: ratios.length };
}

export function project(thesisMs: number, f: DeviceFactor): { ms: number; lo: number; hi: number } {
  return { ms: thesisMs * f.factor, lo: thesisMs * f.min, hi: thesisMs * f.max };
}

const KEY = 'mucodi-measurements';

/** Single-thread measurements persist per device, so the race and the reading room agree. */
export function loadMeasurements(): Measurement[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Measurement[]) : [];
  } catch {
    return [];
  }
}

export function saveMeasurement(m: Measurement): Measurement[] {
  const all = loadMeasurements().filter((x) => x.rung !== m.rung);
  all.push(m);
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    // Storage unavailable: keep the value for this page only.
  }
  return all;
}

export function formatSeconds(ms: number): string {
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(2)} s`;
  if (s < 100) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${(s / 3600).toFixed(1)} h`;
}
