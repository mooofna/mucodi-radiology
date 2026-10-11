// Logarithmic chart axes with plain decimal tick labels.
//
// Observable Plot's own log scale prints SI prefixes ("400m" for 0.4) and leaves some ticks without
// labels. Plotting log10 of the values on a linear scale keeps the full control over ticks and
// labels; marks pass their values through `log10`.

export const log10 = Math.log10;

/** Ticks m * 10^k inside [min, max] for the given mantissas (1, 2 and 5 by default). */
export function logTicks(min: number, max: number, mantissas: number[] = [1, 2, 5]): number[] {
  const ticks: number[] = [];
  for (let e = Math.floor(log10(min)); e <= Math.ceil(log10(max)); e++)
    for (const m of mantissas) {
      const v = Number((m * 10 ** e).toPrecision(6));
      if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
    }
  return ticks;
}

/** 0.05 -> "0.05", 2 -> "2", 100 -> "100", 1500 -> "1,500". */
export function formatDecimal(v: number): string {
  if (v >= 1000) return Math.round(v).toLocaleString('en-US');
  if (v >= 1) return String(Number(v.toPrecision(4)));
  return String(Number(v.toPrecision(2)));
}

/** Options for a Plot position scale showing [min, max] on a log axis. */
export function logScale(label: string, min: number, max: number, mantissas?: number[]) {
  return {
    type: 'linear' as const,
    domain: [log10(min), log10(max)],
    ticks: logTicks(min, max, mantissas).map(log10),
    tickFormat: (d: number) => formatDecimal(10 ** d),
    label
  };
}
