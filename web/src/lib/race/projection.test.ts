import { describe, expect, it } from 'vitest';
import { deviceFactor, formatSeconds, project } from './projection';

describe('device factor projection', () => {
  it('uses the median ratio and keeps the range', () => {
    const f = deviceFactor([
      { rung: 'tmin', liveMs: 414, thesisMs: 868.6 },
      { rung: 't2', liveMs: 543, thesisMs: 997.8 },
      { rung: 't6', liveMs: 1650, thesisMs: 2743.5 }
    ])!;
    expect(f.n).toBe(3);
    expect(f.factor).toBeCloseTo(543 / 997.8, 6);
    expect(f.min).toBeCloseTo(414 / 868.6, 6);
    expect(f.max).toBeCloseTo(1650 / 2743.5, 6);
    const p = project(71746.2, f);
    expect(p.ms).toBeCloseTo(71746.2 * f.factor, 3);
    expect(p.lo).toBeLessThan(p.ms);
    expect(p.hi).toBeGreaterThan(p.ms);
  });

  it('with only TMIN measured, the projected ratio equals the thesis ratio by construction', () => {
    const f = deviceFactor([{ rung: 'tmin', liveMs: 400, thesisMs: 868.5587 }])!;
    expect(project(71746.2, f).ms / 400).toBeCloseTo(71746.2 / 868.5587, 9);
  });

  it('ignores empty input and formats durations', () => {
    expect(deviceFactor([])).toBeNull();
    expect(formatSeconds(870)).toBe('0.87 s');
    expect(formatSeconds(71746)).toBe('71.7 s');
    expect(formatSeconds(436449)).toBe('7 min 16 s');
  });
});
