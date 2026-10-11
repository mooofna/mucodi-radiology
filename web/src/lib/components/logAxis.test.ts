import { describe, expect, it } from 'vitest';
import { formatDecimal, logScale, logTicks } from './logAxis';

describe('log axis', () => {
  it('places 1-2-5 ticks inside the domain', () => {
    expect(logTicks(0.3, 60)).toEqual([0.5, 1, 2, 5, 10, 20, 50]);
    expect(logTicks(0.025, 110, [1, 3])).toEqual([0.03, 0.1, 0.3, 1, 3, 10, 30, 100]);
  });

  it('labels every tick as a plain decimal', () => {
    expect([0.05, 0.4, 1, 2.5, 100, 1500].map(formatDecimal)).toEqual([
      '0.05',
      '0.4',
      '1',
      '2.5',
      '100',
      '1,500'
    ]);
    const s = logScale('Seconds', 0.4, 8);
    expect(s.ticks.map(s.tickFormat)).toEqual(['0.5', '1', '2', '5']);
  });
});
