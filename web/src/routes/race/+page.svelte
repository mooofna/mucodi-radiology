<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import Plot from '#lib/components/Plot.svelte';
  import { Lane, median } from '#lib/inference/engine.ts';
  import { prepare } from '#lib/volume/prep.ts';
  import { GRID_SHAPE } from '#lib/volume/grid.ts';
  import { assetUrl, loadCases, loadModels, type ModelsManifest } from '#lib/data/manifest.ts';
  import { byLabel, loadInference, loadLadder, type Inference, type LadderRung } from '#lib/data/evidence.ts';
  import {
    deviceFactor,
    formatSeconds,
    loadMeasurements,
    project,
    saveMeasurement,
    type Measurement
  } from '#lib/race/projection.ts';

  interface LiveLane {
    rung: string;
    label: string;
    lane: Lane | null;
    state: 'idle' | 'loading' | 'warming' | 'running' | 'stopped' | 'failed';
    progress: string;
    completions: number[];
    latencies: number[];
    error?: string;
  }

  let models = $state<ModelsManifest | null>(null);
  let inference = $state<Inference | null>(null);
  let ladder = $state<LadderRung[]>([]);
  let input = $state<Float32Array | null>(null);
  let caseLabel = $state('');
  let loadError = $state('');

  let challenger = $state('t6');
  let lanes = $state<LiveLane[]>([]);
  let racing = $state(false);
  let t0 = 0;
  let elapsed = $state(0);
  let frame = 0;

  let measurements = $state<Measurement[]>([]);
  let benchRunning = $state(false);
  let benchStatus = $state('');
  let includeLarge = $state(false);
  let hiddenDuring = $state(false);

  const p0Ms = $derived(inference ? (byLabel(inference, 'Pillar-0').cpu_median_ms ?? 0) : 0);
  const c1Ms = $derived(inference ? (byLabel(inference, 'Curia-1').cpu_median_ms ?? 0) : 0);
  const thesisMs = (r: string) => inference?.records.find((x) => x.rung === r)?.cpu_median_ms ?? 0;
  const label = (r: string) => ladder.find((l) => l.rung === r)?.label ?? r.toUpperCase();

  /** Isolated single-thread measurements first; during a race, TMIN's first lap is provisional. */
  const factor = $derived.by(() => {
    if (measurements.length) return { f: deviceFactor(measurements)!, provisional: false };
    const tmin = lanes.find((l) => l.rung === 'tmin');
    if (tmin?.latencies.length) {
      return {
        f: deviceFactor([{ rung: 'tmin', liveMs: tmin.latencies[0]!, thesisMs: thesisMs('tmin') }])!,
        provisional: true
      };
    }
    return null;
  });
  const p0 = $derived(factor ? project(p0Ms, factor.f) : null);
  const c1 = $derived(factor ? project(c1Ms, factor.f) : null);
  const axisMs = $derived(p0 ? p0.ms * 1.08 : 60000);
  const p0Done = $derived(!!p0 && elapsed >= p0.ms);

  onMount(() => {
    measurements = loadMeasurements();
    Promise.all([loadModels(), loadCases(), loadInference(), loadLadder()])
      .then(async ([m, c, i, l]) => {
        models = m;
        inference = i;
        ladder = l;
        const hero = c.cases.find((x) => x.hero) ?? c.cases[0]!;
        caseLabel = `${hero.patient}, ${hero.collection}`;
        const buffer = await (await fetch(assetUrl(`cases/${hero.files.grid}`))).arrayBuffer();
        input = (await prepare('grid', buffer, hero.files.grid)).input;
      })
      .catch((e: unknown) => (loadError = e instanceof Error ? e.message : String(e)));
  });

  onDestroy(() => stop());

  function files(rung: string) {
    const asset = models!.rungs.find((m) => m.rung === rung)!;
    const [main, ...rest] = asset.files.core;
    return {
      url: assetUrl(`models/${main!.path}`),
      external: rest.map((f) => ({ path: f.path, url: assetUrl(`models/${f.path}`) }))
    };
  }

  async function startLane(l: LiveLane) {
    const lane = new Lane(1);
    l.lane = lane;
    l.state = 'loading';
    const f = files(l.rung);
    await lane.load(`${l.rung}.core`, f.url, f.external, (done, total) => (l.progress = `${Math.round((100 * done) / (total || 1))} %`));
    await lane.setInput('input', input!, [1, 4, ...GRID_SHAPE]);
    l.state = 'warming';
    await lane.run(`${l.rung}.core`, 'input', { warmup: 0, iters: 1 });
  }

  async function loop(l: LiveLane) {
    while (racing && l.lane) {
      const r = await l.lane.run(`${l.rung}.core`, 'input', { warmup: 0, iters: 1 });
      if (!racing) break;
      l.latencies = [...l.latencies, r.times[0]!];
      l.completions = [...l.completions, performance.now() - t0];
    }
  }

  async function start() {
    if (!input || !models || racing) return;
    stop();
    const make = (rung: string): LiveLane => ({
      rung,
      label: label(rung),
      lane: null,
      state: 'idle',
      progress: '',
      completions: [],
      latencies: []
    });
    lanes = [make('tmin'), make(challenger)];
    elapsed = 0;
    try {
      await Promise.all(lanes.map((l) => startLane(l)));
    } catch (e) {
      for (const l of lanes) {
        l.state = 'failed';
        l.error = e instanceof Error ? e.message : String(e);
        l.lane?.terminate();
      }
      return;
    }
    racing = true;
    hiddenDuring = document.visibilityState === 'hidden';
    t0 = performance.now();
    for (const l of lanes) l.state = 'running';
    const tick = () => {
      elapsed = performance.now() - t0;
      if (document.visibilityState === 'hidden') hiddenDuring = true;
      if (racing) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    lanes.forEach((l) =>
      loop(l).catch((e: unknown) => {
        l.state = 'failed';
        l.error = e instanceof Error ? e.message : String(e);
      })
    );
  }

  function stop() {
    racing = false;
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frame);
    for (const l of lanes) {
      l.lane?.terminate();
      l.lane = null;
      if (l.state === 'running' || l.state === 'loading' || l.state === 'warming') l.state = 'stopped';
    }
  }

  const countAt = (l: LiveLane, ms: number) => l.completions.filter((c) => c <= ms).length;

  async function benchmark() {
    if (!input || !models || benchRunning) return;
    stop();
    benchRunning = true;
    const order = ['tmin', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 'b0', ...(includeLarge ? ['b2', 'b3', 'b4', 'b5', 'b6'] : [])];
    try {
      for (const rung of order) {
        if (!models.rungs.some((m) => m.rung === rung)) continue;
        benchStatus = `${label(rung)}: loading`;
        const lane = new Lane(1);
        try {
          const f = files(rung);
          await lane.load(`${rung}.core`, f.url, f.external);
          await lane.setInput('input', input, [1, 4, ...GRID_SHAPE]);
          benchStatus = `${label(rung)}: one warm-up and five timed passes`;
          const hiddenBefore = document.visibilityState === 'hidden';
          const r = await lane.run(`${rung}.core`, 'input', { warmup: 1, iters: 5 });
          if (hiddenBefore || document.visibilityState === 'hidden') {
            benchStatus = 'Paused: the tab went to the background, which browsers slow down. Bring it to the front and run again.';
            break;
          }
          measurements = saveMeasurement({ rung, liveMs: median(r.times), thesisMs: thesisMs(rung) });
        } finally {
          lane.terminate();
        }
      }
      if (!benchStatus.startsWith('Paused')) benchStatus = 'Done';
    } catch (e) {
      benchStatus = e instanceof Error ? `Stopped: ${e.message}` : 'Stopped';
    } finally {
      benchRunning = false;
    }
  }

  const benchRows = $derived(
    [...measurements]
      .sort((a, b) => a.thesisMs - b.thesisMs)
      .map((m) => ({ ...m, label: label(m.rung), ratio: m.liveMs / m.thesisMs }))
  );

  function chart(P: typeof import('@observablehq/plot'), width: number) {
    if (!benchRows.length || !factor) return null;
    const f = factor.f.factor;
    const values = benchRows.flatMap((r) => [r.thesisMs, r.liveMs]).map((v) => v / 1000);
    const lo = Math.min(...values) / 1.4;
    const hi = Math.max(...values) * 1.4;
    return P.plot({
      width,
      height: Math.min(380, width * 0.7),
      marginLeft: 52,
      marginBottom: 40,
      x: { type: 'log', label: 'Thesis Xeon Silver 4214R, 1 thread (s)', domain: [lo, hi] },
      y: { type: 'log', label: 'This device, 1 thread (s)', domain: [lo, hi] },
      marks: [
        P.line([[lo, lo], [hi, hi]], { stroke: 'currentColor', strokeOpacity: 0.35, strokeDasharray: '3,3' }),
        P.line([[lo, lo * f], [hi, hi * f]], { stroke: 'var(--accent)' }),
        P.dot(benchRows, { x: (d) => d.thesisMs / 1000, y: (d) => d.liveMs / 1000, fill: 'currentColor', r: 3 }),
        P.text(benchRows, { x: (d) => d.thesisMs / 1000, y: (d) => d.liveMs / 1000, text: 'label', dx: 8, textAnchor: 'start' })
      ]
    });
  }

  const tmin = $derived(inference ? byLabel(inference, 'TMIN') : null);
</script>

<svelte:head>
  <title>CPU race, MuCoDi Chest CT</title>
</svelte:head>

<svg width="0" height="0" style="position: absolute" aria-hidden="true">
  <defs>
    <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#4477aa" stroke-width="2" />
    </pattern>
  </defs>
</svg>

<div class="page">
  <header class="intro">
    <p class="label">CPU race</p>
    <h1>One CPU thread. The same scan.</h1>
    <p class="lede">
      On one thread of the thesis benchmark machine the smallest student encoded a chest CT in
      <span class="num">{formatSeconds(thesisMs('tmin'))}</span> and Pillar-0 took
      <span class="num">{formatSeconds(p0Ms)}</span>. Here the students run live on your CPU; the
      teachers are projected from those measurements, never run.
    </p>
  </header>

  {#if loadError}<p class="error">{loadError}</p>{/if}

  <section class="controls">
    <label>
      <span class="label">Challenger</span>
      <select bind:value={challenger} disabled={racing}>
        {#each ladder.filter((l) => l.rung !== 'tmin' && l.params_m < 13) as l (l.rung)}
          <option value={l.rung}>{l.label}, {l.params_m < 1 ? `${(l.params_m * 1000).toFixed(0)} k` : `${l.params_m} M`} parameters</option>
        {/each}
      </select>
    </label>
    <p class="muted small">Scan: {caseLabel || 'loading'}. Each student runs in its own worker on one thread.</p>
    {#if racing}
      <button onclick={stop}>Stop</button>
    {:else}
      <button class="primary" disabled={!input || !models} onclick={start}>Start the race</button>
    {/if}
  </section>

  <section class="board" aria-label="Race">
    <p class="clock num" aria-hidden="true">{(elapsed / 1000).toFixed(1)}<span class="unit"> s</span></p>
    <div class="lanes">
      {#each lanes as l (l.rung)}
        <div class="lane">
          <div class="who">
            <span class="name">{l.label}</span>
            <span class="tag">live, this device</span>
          </div>
          <div class="track">
            {#each l.completions as c, i (i)}
              <span class="tickmark" style={`left: ${Math.min(100, (c / axisMs) * 100)}%`}></span>
            {/each}
          </div>
          <div class="count num">
            {#if l.state === 'loading'}loading {l.progress}{:else if l.state === 'warming'}warming up{:else if l.state === 'failed'}failed{:else}{l.completions.length} scans{/if}
          </div>
        </div>
      {/each}
      {#if lanes.length}
        {#each [{ name: 'Pillar-0', p: p0 }, { name: 'Curia-1', p: c1 }] as t (t.name)}
          <div class="lane projected">
            <div class="who">
              <span class="name">{t.name}</span>
              <span class="tag">projected, not run</span>
            </div>
            <div class="track">
              {#if t.p}
                <svg class="hatch" style={`width: ${Math.min(100, (Math.min(elapsed, t.p.ms) / axisMs) * 100)}%`} aria-hidden="true">
                  <rect width="100%" height="100%" fill="url(#hatch)" />
                </svg>
                {#if t.p.ms <= axisMs}<span class="finish" style={`left: ${(t.p.ms / axisMs) * 100}%`}></span>{/if}
              {/if}
            </div>
            <div class="count num">
              {#if t.p}{elapsed >= t.p.ms ? '1 scan' : `${Math.round((100 * elapsed) / t.p.ms)} %`}{:else}waiting{/if}
            </div>
          </div>
        {/each}
      {/if}
    </div>

    {#if hiddenDuring}
      <p class="warn small">The tab was in the background during this race. Browsers slow background tabs down, so these counts are lower than this device can do.</p>
    {/if}
    {#if p0Done && p0 && lanes.length}
      <p class="verdict" aria-live="polite">
        In the time Pillar-0 would need for one scan on this CPU (about {formatSeconds(p0.ms)}),
        {#each lanes as l, i (l.rung)}{i > 0 ? ' and ' : ''}{l.label} encoded <span class="num">{countAt(l, p0.ms)}</span>{/each}.
      </p>
    {/if}

    {#if factor}
      <p class="muted small">
        Projection: thesis single-thread time times this device's factor
        <span class="num">{factor.f.factor.toFixed(2)}</span>, the ratio of TMIN's time here to its thesis time, so the
        projected Pillar-0 to TMIN ratio is the thesis's {tmin?.cpu_speedup_vs_pillar0?.toFixed(1)} by construction and your
        device sets the time scale.{#if factor.f.n > 1}
          Other students ran at ratios from <span class="num">{factor.f.min.toFixed(2)}</span> to
          <span class="num">{factor.f.max.toFixed(2)}</span>, which would put Pillar-0 between
          {formatSeconds(p0Ms * factor.f.min)} and {formatSeconds(p0Ms * factor.f.max)}.{/if}
        {#if factor.provisional}Provisional: estimated from the first TMIN pass of this race; the benchmark below measures each student alone.{/if}
        Students racing side by side share memory bandwidth, so their live counts are, if anything, conservative.
      </p>
    {/if}
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Benchmark</p>
      <h2>Reproduce Table 6.5 on this device</h2>
    </div>
    <div class="section-body">
      <p>
        Each student runs alone in a fresh worker on one thread: one warm-up, then the median of five forward passes
        on the same scan, as in the thesis. The ratio to the thesis Xeon times is this device's factor.
      </p>
      <div class="benchctl">
        <button class="primary" disabled={!input || benchRunning || racing} onclick={benchmark}>
          {benchRunning ? 'Measuring' : 'Run the benchmark'}
        </button>
        <label class="small"><input type="checkbox" bind:checked={includeLarge} disabled={benchRunning} /> Include B2 to B6 (about 600 MB of models)</label>
        <span class="muted small">{benchStatus}</span>
      </div>
      {#if benchRows.length}
        <Plot make={chart} label="Single-thread encode time per student: this device against the thesis Xeon" />
        <table>
          <thead>
            <tr><th>Student</th><th class="num">Thesis Xeon (s)</th><th class="num">This device (s)</th><th class="num">Ratio</th></tr>
          </thead>
          <tbody>
            {#each benchRows as r (r.rung)}
              <tr><td>{r.label}</td><td class="num">{(r.thesisMs / 1000).toFixed(2)}</td><td class="num">{(r.liveMs / 1000).toFixed(2)}</td><td class="num">{r.ratio.toFixed(2)}</td></tr>
            {/each}
            {#if p0 && c1}
              <tr class="proj"><td>Pillar-0, projected</td><td class="num">{(p0Ms / 1000).toFixed(2)}</td><td class="num">{(p0.ms / 1000).toFixed(1)}</td><td></td></tr>
              <tr class="proj"><td>Curia-1, projected</td><td class="num">{(c1Ms / 1000).toFixed(2)}</td><td class="num">{(c1.ms / 1000).toFixed(1)}</td><td></td></tr>
            {/if}
          </tbody>
        </table>
        <p class="muted small">
          Thesis: Intel Xeon Silver 4214R, PyTorch 2.11, fp32, one pinned thread (Table 6.5). Here: ONNX Runtime Web 1.30
          (WebAssembly with SIMD), fp32, one thread. Measurements stay in this browser.
        </p>
      {/if}
    </div>
  </section>

  {#if tmin}
    <section class="section">
      <div class="section-label"><p class="label">Context</p></div>
      <div class="section-body">
        <p>
          On a GPU the gap closes. On an A100 at batch size 1 the smallest student is
          <span class="num">{tmin.gpu_b1_speedup_vs_pillar0}</span> times faster than Pillar-0, against
          <span class="num">{tmin.cpu_speedup_vs_pillar0?.toFixed(1)}</span> times on one CPU thread (Table 6.5). The thesis
          reads this as the GPU forward pass being bound by memory bandwidth, so there the parameter cut surfaces as
          reduced memory rather than lower latency.
        </p>
      </div>
    </section>
  {/if}
</div>

<style>
  .intro h1 {
    font-size: var(--text-display);
    margin: 0.3rem 0 0.8rem;
  }

  .lede {
    font-size: var(--text-lg);
    max-width: 46rem;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: var(--space-2) var(--space-3);
    border-top: 1px solid var(--rule-strong);
    padding-top: var(--space-2);
  }

  .controls label {
    display: grid;
    gap: 0.2rem;
  }

  .controls p {
    margin: 0;
  }

  .board {
    margin-top: var(--space-3);
  }

  .clock {
    font-size: clamp(3.2rem, 2rem + 5vw, 6.5rem);
    line-height: 1;
    margin: 0 0 var(--space-2);
    letter-spacing: -0.03em;
  }

  .unit {
    font-size: 0.4em;
    color: var(--muted);
  }

  .lanes {
    display: grid;
    border-top: 1px solid var(--rule-strong);
  }

  .lane {
    display: grid;
    grid-template-columns: 11rem minmax(0, 1fr) 7rem;
    align-items: center;
    gap: var(--space-2);
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--rule);
  }

  .who {
    display: grid;
  }

  .name {
    font-weight: 600;
  }

  .tag {
    font-size: var(--text-xs);
    color: var(--muted);
  }

  .projected .name {
    color: var(--c-pillar0);
  }

  .track {
    position: relative;
    height: 26px;
    background: var(--bg-sunk);
  }

  .tickmark {
    position: absolute;
    top: 3px;
    bottom: 3px;
    width: 2px;
    margin-left: -1px;
    background: var(--accent);
  }

  .hatch {
    position: absolute;
    top: 3px;
    left: 0;
    height: 20px;
    border-right: 1px solid var(--c-pillar0);
  }

  .finish {
    position: absolute;
    top: -4px;
    bottom: -4px;
    border-left: 2px solid var(--c-pillar0);
  }

  .count {
    text-align: right;
    font-size: var(--text-sm);
  }

  .verdict {
    font-size: var(--text-lg);
    margin: var(--space-2) 0;
    max-width: 52rem;
    border-left: 3px solid var(--accent);
    padding-left: 0.8rem;
  }

  .benchctl {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }

  .small {
    font-size: var(--text-xs);
  }

  tr.proj td {
    color: var(--muted);
  }

  .error,
  .warn {
    color: var(--warn);
  }

  @media (max-width: 760px) {
    .lane {
      grid-template-columns: 6rem minmax(0, 1fr) 4.5rem;
    }
  }
</style>
