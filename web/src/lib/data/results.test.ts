import { describe, expect, it } from 'vitest';
import { naturalFrequencies } from './results';

describe('natural frequencies', () => {
  it('splits 1,000 scans by prevalence, sensitivity and specificity', () => {
    const r = naturalFrequencies(0.9, 0.8, 0.2);
    expect(r.tp).toBeCloseTo(180, 9);
    expect(r.fn).toBeCloseTo(20, 9);
    expect(r.tn).toBeCloseTo(640, 9);
    expect(r.fp).toBeCloseTo(160, 9);
    expect(r.ppv).toBeCloseTo(180 / 340, 9);
    expect(r.npv).toBeCloseTo(640 / 660, 9);
  });
});
