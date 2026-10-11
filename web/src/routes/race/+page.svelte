<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import Plot from '#lib/components/Plot.svelte';
  import { log10, logScale } from '#lib/components/logAxis.ts';
  import { Lane, median } from '#lib/inference/engine.ts';
  import { prepare } from '#lib/volume/prep.ts';
  import { GRID_SHAPE } from '#lib/volume/grid.ts';
  import { assetUrl, loadCases, loadModels, type ModelsManifest } from '#lib/data/manifest.ts';
  import {
    byLabel,
    loadInference,
    loadLadder,
    type Inference,
    type LadderRung
  } from '#lib/data/evidence.ts';
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
  /** Thesis timings are quoted as in Table 6.5, to two decimals. */
  const fmtThesis = (ms: number) => `${(ms / 1000).toFixed(2)} s`;
  const fmtParams = (p: number) => (p < 1 ? `${(p * 1000).toFixed(0)} k` : `${p} M`);

  /** Isolated single-thread measurements first; during a race, TMIN's first pass is provisional. */
  const factor = $derived.by(() => {
    if (measurements.length) return { f: deviceFactor(measurements)!, provisional: false };
    const tmin = lanes.find((l) => l.rung === 'tmin');
    if (tmin?.latencies.length) {
      return {
        f: deviceFactor([
          { rung: 'tmin', liveMs: tmin.latencies[0]!, thesisMs: thesisMs('tmin') }
        ])!,
        provisional: true
      };
    }
    return null;
  });
  const p0 = $derived(factor ? project(p0Ms, factor.f) : null);
  const c1 = $derived(factor ? project(c1Ms, factor.f) : null);
  const axisMs = $derived(p0 ? p0.ms * 1.08 : 60000);
  const p0Done = $derived(!!p0 && elapsed >= p0.ms);

  const blank = (rung: string): LiveLane => ({
    rung,
    label: label(rung),
    lane: null,
    state: 'idle',
    progress: '',
    completions: [],
    latencies: []
  });
  // Before the first race the board already shows its four lanes.
  const shown = $derived<LiveLane[]>(lanes.length ? lanes : [blank('tmin'), blank(challenger)]);

  onMount(() => {
    measurements = loadMeasurements();
    Promise.all([loadModels(), loadCases(), loadInference(), loadLadder()])
      .then(async ([m, c, i, l]) => {
        models = m;
        inference = i;
        ladder = l;
        const hero = c.cases.find((x) => x.hero) ?? c.cases[0]!;
        caseLabel = `${hero.patient} (${hero.collection})`;
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
    await lane.load(
      `${l.rung}.core`,
      f.url,
      f.external,
      (done, total) => (l.progress = `${Math.round((100 * done) / (total || 1))} %`)
    );
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
    lanes = [blank('tmin'), blank(challenger)];
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
      if (l.state === 'running' || l.state === 'loading' || l.state === 'warming')
        l.state = 'stopped';
    }
  }

  const countAt = (l: LiveLane, ms: number) => l.completions.filter((c) => c <= ms).length;

  async function benchmark() {
    if (!input || !models || benchRunning) return;
    stop();
    benchRunning = true;
    const order = [
      'tmin',
      't0',
      't1',
      't2',
      't3',
      't4',
      't5',
      't6',
      'b0',
      ...(includeLarge ? ['b2', 'b3', 'b4', 'b5', 'b6'] : [])
    ];
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
            benchStatus =
              'Paused because the tab went to the background, which browsers slow down. Bring it to the front and run again.';
            break;
          }
          measurements = saveMeasurement({
            rung,
            liveMs: median(r.times),
            thesisMs: thesisMs(rung)
          });
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

  const order = (r: string) => ladder.findIndex((l) => l.rung === r);
  const benchRows = $derived(
    [...measurements]
      .sort((a, b) => order(a.rung) - order(b.rung))
      .map((m) => ({ ...m, label: label(m.rung), ratio: m.liveMs / m.thesisMs }))
  );

  const THESIS = '#9aa7b4';
  const HERE = 'var(--accent)';

  function chart(P: typeof import('@observablehq/plot'), width: number) {
    if (!benchRows.length) return null;
    const rows = [
      ...benchRows.map((r) => ({
        label: r.label,
        thesis: r.thesisMs / 1000,
        here: r.liveMs / 1000
      })),
      ...(p0 && c1
        ? [
            { label: 'Pillar-0', thesis: p0Ms / 1000, here: p0.ms / 1000 },
            { label: 'Curia-1', thesis: c1Ms / 1000, here: c1.ms / 1000 }
          ]
        : [])
    ];
    const values = rows.flatMap((r) => [r.thesis, r.here]);
    const lo = Math.min(...values) / 1.3;
    const hi = Math.max(...values) * 1.3;
    const points = rows.flatMap((r) => [
      { label: r.label, s: r.thesis, where: 'Thesis CPU' },
      {
        label: r.label,
        s: r.here,
        where:
          r.label.startsWith('Pillar') || r.label.startsWith('Curia')
            ? 'This computer, projected'
            : 'This computer'
      }
    ]);
    return P.plot({
      width,
      height: 26 * rows.length + 70,
      marginLeft: 72,
      marginRight: 16,
      marginBottom: 44,
      x: {
        ...logScale('Seconds per scan, one thread', lo, hi),
        grid: true,
        labelAnchor: 'center',
        labelOffset: 36
      },
      y: { domain: rows.map((r) => r.label), label: null },
      color: {
        domain: ['Thesis CPU', 'This computer', 'This computer, projected'],
        range: [THESIS, HERE, HERE]
      },
      symbol: {
        domain: ['Thesis CPU', 'This computer', 'This computer, projected'],
        range: ['circle', 'circle', 'diamond']
      },
      marks: [
        P.ruleY(rows, {
          y: 'label',
          x1: (d) => log10(d.thesis),
          x2: (d) => log10(d.here),
          stroke: 'var(--rule)',
          strokeWidth: 2
        }),
        P.dot(points, {
          x: (d) => log10(d.s),
          y: 'label',
          fill: 'where',
          symbol: 'where',
          r: 4.5
        }),
        P.tip(
          points,
          P.pointer({
            x: (d) => log10(d.s),
            y: 'label',
            title: (d) => `${d.label}, ${d.where.toLowerCase()}: ${formatSeconds(d.s * 1000)}`
          })
        )
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
    <pattern
      id="hatch"
      width="6"
      height="6"
      patternUnits="userSpaceOnUse"
      patternTransform="rotate(45)"
    >
      <line x1="0" y1="0" x2="0" y2="6" stroke="#4477aa" stroke-width="2" />
    </pattern>
  </defs>
</svg>

<div class="page">
  <header class="intro">
    <h1>CPU race</h1>
    <p class="lede">
      Two students encode the same chest CT on this computer, each on one CPU thread, as often as
      they can. The teachers are not run. Their lanes advance at the single-thread times measured in
      the thesis, {fmtThesis(p0Ms)} per scan for Pillar-0 and {fmtThesis(c1Ms)} for Curia-1, scaled to
      the speed of this computer.
    </p>
  </header>

  {#if loadError}<p class="error">{loadError}</p>{/if}

  <section class="race" aria-label="Race">
    <div class="controls">
      <label class="field">
        <span>Second student</span>
        <select bind:value={challenger} disabled={racing}>
          {#each ladder.filter((l) => l.rung !== 'tmin' && l.params_m < 13) as l (l.rung)}
            <option value={l.rung}>{l.label}, {fmtParams(l.params_m)} parameters</option>
          {/each}
        </select>
      </label>
      {#if racing}
        <button onclick={stop}>Stop</button>
      {:else}
        <button class="primary" disabled={!input || !models} onclick={start}>
          {lanes.length ? 'Race again' : 'Start the race'}
        </button>
      {/if}
      <p class="hint">TMIN always races. Scan: {caseLabel || 'loading'}.</p>
    </div>

    <p class="clock num" class:idle={!racing && !lanes.length} aria-hidden="true">
      {(elapsed / 1000).toFixed(1)}<span class="unit"> s</span>
    </p>

    <div class="lanes">
      {#each shown as l (l.rung)}
        <div class="lane">
          <div class="who">
            <span class="name">{l.label}</span>
            <span class="tag">runs on this computer</span>
          </div>
          <div class="track">
            {#each l.completions as c, i (i)}
              <span class="tickmark" style={`left: ${Math.min(100, (c / axisMs) * 100)}%`}></span>
            {/each}
          </div>
          <div class="count num">
            {#if l.state === 'loading'}loading {l.progress}{:else if l.state === 'warming'}warming
              up{:else if l.state === 'failed'}failed{:else if l.state === 'idle'}<span
                class="muted">ready</span
              >{:else}{l.completions.length} scans{/if}
          </div>
        </div>
      {/each}
      {#each [{ name: 'Pillar-0', p: p0 }, { name: 'Curia-1', p: c1 }] as t (t.name)}
        <div class="lane projected">
          <div class="who">
            <span class="name">{t.name}</span>
            <span class="tag"
              >{t.p ? `projected, ${formatSeconds(t.p.ms)} per scan` : 'projected, not run'}</span
            >
          </div>
          <div class="track">
            {#if t.p && lanes.length}
              <svg
                class="hatch"
                style={`width: ${Math.min(100, (Math.min(elapsed, t.p.ms) / axisMs) * 100)}%`}
                aria-hidden="true"
              >
                <rect width="100%" height="100%" fill="url(#hatch)" />
              </svg>
            {/if}
            {#if t.p && t.p.ms <= axisMs}<span
                class="finish"
                style={`left: ${(t.p.ms / axisMs) * 100}%`}
              ></span>{/if}
          </div>
          <div class="count num">
            {#if t.p && lanes.length}{elapsed >= t.p.ms
                ? '1 scan'
                : `${Math.round((100 * elapsed) / t.p.ms)} %`}{:else}<span class="muted"
                >{t.p ? 'ready' : 'waiting'}</span
              >{/if}
          </div>
        </div>
      {/each}
    </div>

    {#if hiddenDuring}
      <p class="warn">
        The tab was in the background during this race. Browsers slow background tabs down, so these
        counts are lower than this computer can do.
      </p>
    {/if}
    {#if p0Done && p0 && lanes.length}
      <p class="verdict" aria-live="polite">
        In the {formatSeconds(p0.ms)} that Pillar-0 would need for one scan on this computer,
        {#each lanes as l, i (l.rung)}{i > 0 ? ' and ' : ''}{l.label} encoded
          <span class="num">{countAt(l, p0.ms)}</span> scans{/each}.
      </p>
    {/if}

    <details class="more">
      <summary>How the teacher lanes are projected</summary>
      <p class="hint">
        The thesis measured every model on one thread of the same Intel Xeon Silver 4214R (Table
        6.5). This computer's speed is the ratio of a student's time here to its thesis time{#if factor},
          currently {factor.f.factor.toFixed(2)}{#if factor.provisional}, from the first TMIN pass
            of this race{/if}{/if}. The teachers' thesis times are multiplied by the same ratio, so
        the gap between Pillar-0 and TMIN stays at the {tmin?.cpu_speedup_vs_pillar0?.toFixed(1)} times
        the thesis measured. The benchmark below measures each student on its own. Students racing side
        by side share memory bandwidth, so their counts here are lower than they would be alone.
      </p>
    </details>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Benchmark</p>
      <h2>Benchmark on this computer</h2>
    </div>
    <div class="section-body">
      <p>
        Each student runs alone on one thread: one warm-up pass, then the median of five timed
        passes on the same scan, as in the thesis. The results set the speed ratio used for the
        projections.
      </p>
      <div class="benchctl">
        <button class="primary" disabled={!input || benchRunning || racing} onclick={benchmark}>
          {benchRunning ? 'Measuring' : 'Run the benchmark'}
        </button>
        <label class="hint check"
          ><input type="checkbox" bind:checked={includeLarge} disabled={benchRunning} /> Include B2 to
          B6 (about 460 MB more to download)</label
        >
      </div>
      {#if benchStatus}<p class="hint" aria-live="polite">{benchStatus}</p>{/if}
      {#if benchRows.length}
        <ul class="legend" aria-label="Legend">
          <li><span class="dot thesis"></span>Thesis CPU, measured</li>
          <li><span class="dot here"></span>This computer, measured</li>
          <li><span class="diamond"></span>This computer, projected</li>
        </ul>
        <Plot
          make={chart}
          label="Seconds per scan on one thread: the thesis CPU and this computer"
        />
        <table>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" class="num">Thesis CPU (s)</th>
              <th scope="col" class="num">This computer (s)</th>
              <th scope="col" class="num">Ratio</th>
            </tr>
          </thead>
          <tbody>
            {#each benchRows as r (r.rung)}
              <tr>
                <td>{r.label}</td>
                <td class="num">{(r.thesisMs / 1000).toFixed(2)}</td>
                <td class="num">{(r.liveMs / 1000).toFixed(2)}</td>
                <td class="num">{r.ratio.toFixed(2)}</td>
              </tr>
            {/each}
            {#if p0 && c1}
              <tr class="proj">
                <td>Pillar-0, projected</td>
                <td class="num">{(p0Ms / 1000).toFixed(2)}</td>
                <td class="num">{(p0.ms / 1000).toFixed(1)}</td>
                <td></td>
              </tr>
              <tr class="proj">
                <td>Curia-1, projected</td>
                <td class="num">{(c1Ms / 1000).toFixed(2)}</td>
                <td class="num">{(c1.ms / 1000).toFixed(1)}</td>
                <td></td>
              </tr>
            {/if}
          </tbody>
        </table>
        <p class="hint">
          Thesis: Intel Xeon Silver 4214R, PyTorch 2.11, 32-bit floats, one pinned thread (Table
          6.5). Here: ONNX Runtime Web 1.30 (WebAssembly with SIMD), 32-bit floats, one thread. The
          measurements stay in this browser.
        </p>
      {/if}
    </div>
  </section>

  {#if tmin}
    <section class="section">
      <div class="section-label">
        <p class="label">GPU</p>
        <h2>On a GPU</h2>
      </div>
      <div class="section-body">
        <p>
          On one NVIDIA A100 at batch size 1, TMIN is {tmin.gpu_b1_speedup_vs_pillar0} times faster than
          Pillar-0, against {tmin.cpu_speedup_vs_pillar0?.toFixed(1)} times on one CPU thread (Table 6.5).
          At this input size the GPU forward pass is limited by memory bandwidth, so on a GPU the smaller
          model mainly needs less memory. See the
          <a href={resolve('/results')}>results</a> for all models.
        </p>
      </div>
    </section>
  {/if}
</div>

<style>
  .intro h1 {
    font-size: var(--text-display);
    margin: 0 0 0.8rem;
  }

  .lede {
    font-size: var(--text-lg);
    max-width: 48rem;
  }

  .race {
    border-top: 1px solid var(--rule-strong);
    margin-top: var(--space-4);
    padding-top: var(--space-3);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: var(--space-2) var(--space-3);
  }

  .controls .hint {
    margin: 0;
    padding-bottom: 0.4rem;
  }

  .clock {
    font-size: clamp(3rem, 2rem + 4vw, 5.5rem);
    font-weight: 500;
    line-height: 1;
    margin: var(--space-3) 0 var(--space-2);
    letter-spacing: -0.03em;
  }

  .clock.idle {
    color: var(--muted);
  }

  .unit {
    font-size: 0.4em;
    font-weight: 400;
    color: var(--muted);
  }

  .lanes {
    display: grid;
    border-top: 1px solid var(--rule);
  }

  .lane {
    display: grid;
    grid-template-columns: 13rem minmax(0, 1fr) 6rem;
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
    color: var(--c-pillar0-text);
  }

  .track {
    position: relative;
    height: 24px;
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
    height: 18px;
  }

  .finish {
    position: absolute;
    top: -3px;
    bottom: -3px;
    border-left: 2px solid var(--c-pillar0);
  }

  .count {
    text-align: right;
    font-size: var(--text-sm);
  }

  .verdict {
    font-size: var(--text-lg);
    margin: var(--space-3) 0 var(--space-2);
    max-width: 52rem;
    border-left: 3px solid var(--accent);
    padding-left: 0.9rem;
  }

  .race details {
    margin-top: var(--space-2);
  }

  .race details p {
    max-width: 48rem;
  }

  .benchctl {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-1);
  }

  .check {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  tr.proj td {
    color: var(--muted);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem var(--space-3);
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    font-size: var(--text-sm);
    color: var(--muted);
  }

  .legend li {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
  }

  .dot.thesis {
    background: #9aa7b4;
  }

  .dot.here {
    background: var(--accent);
  }

  .diamond {
    width: 8px;
    height: 8px;
    background: var(--accent);
    transform: rotate(45deg);
  }

  .error,
  .warn {
    color: var(--warn);
  }

  @media (max-width: 760px) {
    .lane {
      grid-template-columns: 6.5rem minmax(0, 1fr) 4.5rem;
    }

    .tag {
      display: none;
    }
  }
</style>
