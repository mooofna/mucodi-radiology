<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import Plot from '#lib/components/Plot.svelte';
  import { log10, logScale } from '#lib/components/logAxis.ts';
  import { getJson } from '#lib/data/manifest.ts';
  import {
    loadInference,
    loadLadder,
    type Inference,
    type LadderRung
  } from '#lib/data/evidence.ts';
  import {
    STUDENTS,
    TEACHER_COLORS,
    loadCohorts,
    loadCurves,
    loadDeploy,
    loadEndpoints,
    loadTradeoff,
    naturalFrequencies,
    type CohortCurves,
    type Cohorts,
    type DeployRow,
    type Endpoint,
    type Tradeoff
  } from '#lib/data/results.ts';
  import type * as PlotNS from '@observablehq/plot';

  type Probe = 'mlp' | 'linear';
  const MLP = '#2E7D32';
  const LINEAR = '#CC4444';
  const PROBE_LABEL: Record<Probe, string> = { mlp: 'MLP probe', linear: 'linear probe' };
  const PARAMS = { min: 0.025, max: 110, mantissas: [1, 3] };
  const paramAxis = () => ({
    ...logScale('Encoder parameters (millions)', PARAMS.min, PARAMS.max, PARAMS.mantissas),
    labelAnchor: 'center' as const,
    labelOffset: 34
  });

  const SECTIONS = [
    { id: 'size', label: 'Model size' },
    { id: 'cohorts', label: 'Cohorts' },
    { id: 'endpoints', label: 'Endpoints' },
    { id: 'deployment', label: 'Deployment' },
    { id: 'speed', label: 'Speed' },
    { id: 'verification', label: 'Verification' }
  ];

  let tradeoff = $state<Tradeoff | null>(null);
  let cohorts = $state<Cohorts | null>(null);
  let endpoints = $state<Endpoint[]>([]);
  let deploy = $state<DeployRow[]>([]);
  let inference = $state<Inference | null>(null);
  let ladder = $state<LadderRung[]>([]);
  let curvesReport = $state<{ checked: number; mismatches: string[]; missing: string[] } | null>(
    null
  );
  let loadError = $state('');

  let heatProbe = $state<Probe>('mlp');
  let heatDelta = $state(false);
  let epCohort = $state('lidc');
  let epName = $state('malig');
  let epProbe = $state<Probe>('mlp');
  let epModels = $state<string[]>(['TMIN', 'T2', 'B6', 'Pillar-0', 'Curia-1']);
  let curves = $state<CohortCurves | null>(null);
  let opName = $state('youden');
  let prevalenceOverride = $state<number | null>(null);
  let deployProbe = $state<Probe>('mlp');

  const params = $derived.by(() => {
    const p: Record<string, number> = {};
    for (const l of ladder) p[l.label] = l.params_m;
    for (const r of inference?.records ?? [])
      if (r.family === 'teacher')
        p[r.label === 'CT-CLIP (VocabFine)' ? 'CT-CLIP-VF' : r.label] = r.params_m;
    return p;
  });

  onMount(() => {
    Promise.all([
      loadTradeoff(),
      loadCohorts(),
      loadEndpoints(),
      loadDeploy(),
      loadInference(),
      loadLadder()
    ])
      .then(([t, c, e, d, i, l]) => {
        tradeoff = t;
        cohorts = c;
        endpoints = e.endpoints;
        deploy = d.rows;
        inference = i;
        ladder = l;
      })
      .catch((err: unknown) => (loadError = err instanceof Error ? err.message : String(err)));
    getJson<{ checked: number; mismatches: string[]; missing: string[] }>(
      'evidence/curves_report.json'
    )
      .then((r) => (curvesReport = r))
      .catch(() => (curvesReport = null));
  });

  $effect(() => {
    const cohort = epCohort;
    curves = null;
    loadCurves(cohort)
      .then((c) => {
        if (cohort === epCohort) curves = c;
      })
      .catch(() => (curves = null));
  });

  // ---- Model size ------------------------------------------------------------------------------
  function tradeoffChart(P: typeof PlotNS, width: number) {
    if (!tradeoff) return null;
    const rows = (probe: Probe) =>
      tradeoff!.probes[probe]
        .filter((r) => STUDENTS.includes(r.label))
        .map((r) => ({ ...r, x: log10(params[r.label] ?? NaN), probe }));
    const teacher = (probe: Probe, label: string) =>
      tradeoff!.probes[probe].find((r) => r.label === label)!;
    const floor = teacher('mlp', 'Untrained (B6)').avg_ext;
    const series = [...rows('mlp'), ...rows('linear')];
    const right = log10(PARAMS.max);
    return P.plot({
      width,
      height: Math.min(460, Math.round(width * 0.62)),
      marginLeft: 44,
      marginRight: 150,
      marginBottom: 44,
      x: paramAxis(),
      y: {
        label: 'Average external AUROC (%)',
        domain: [48, 82],
        grid: true,
        ticks: [50, 55, 60, 65, 70, 75, 80]
      },
      marks: [
        P.ruleY([floor], { stroke: '#8a939c', strokeDasharray: '6,3,1,3' }),
        P.text([floor], {
          x: right,
          y: (d) => d,
          text: () => 'untrained B6',
          textAnchor: 'start',
          dx: 6,
          fill: '#7a848e'
        }),
        ...(['mlp', 'linear'] as const).flatMap((probe) => {
          const p0 = teacher(probe, 'Pillar-0');
          const color = probe === 'mlp' ? MLP : LINEAR;
          return [
            P.ruleY([p0.avg_ext], { stroke: color, strokeDasharray: '2,3', strokeOpacity: 0.8 }),
            P.text([p0.avg_ext], {
              x: right,
              y: (d) => d,
              text: () => `Pillar-0, ${probe === 'mlp' ? 'MLP' : 'linear'}`,
              textAnchor: 'start',
              dx: 6,
              fill: color
            }),
            P.ruleX(rows(probe), {
              x: 'x',
              y1: (d) => d.avg_ext - d.spread,
              y2: (d) => d.avg_ext + d.spread,
              stroke: color,
              strokeOpacity: 0.22,
              dx: probe === 'mlp' ? -2 : 2
            }),
            P.line(rows(probe), {
              x: 'x',
              y: 'avg_ext',
              stroke: color,
              strokeWidth: 1.75,
              strokeDasharray: probe === 'linear' ? '5,3' : undefined
            }),
            P.dot(rows(probe), {
              x: 'x',
              y: 'avg_ext',
              fill: color,
              r: 3.2,
              symbol: probe === 'mlp' ? 'circle' : 'square'
            })
          ];
        }),
        P.text(
          rows('mlp').filter((r) => ['TMIN', 'T2', 'B6'].includes(r.label)),
          { x: 'x', y: 'avg_ext', text: 'label', dy: -11, fill: 'currentColor' }
        ),
        P.tip(
          series,
          P.pointer({
            x: 'x',
            y: 'avg_ext',
            title: (d) =>
              `${d.label}, ${PROBE_LABEL[d.probe as Probe]}\n${d.avg_ext.toFixed(1)} % average external AUROC\nspread between cohorts ${d.spread.toFixed(1)}\n${params[d.label]} M parameters`
          })
        )
      ]
    });
  }

  // ---- Cohorts ---------------------------------------------------------------------------------
  const ROWS = [
    ...STUDENTS,
    'Pillar-0',
    'Curia-1',
    'Curia-2',
    'CT-CLIP',
    'CT-CLIP-VF',
    'TANGERINE',
    'Untrained (B6)'
  ];
  const COHORT_LABELS: Record<string, string> = {
    lidc_chest_ct: 'LIDC',
    stoic2021: 'STOIC',
    rspect: 'RSPECT',
    mosmed: 'MosMed',
    coca: 'COCA',
    coca_gated: 'COCA gated',
    osic: 'OSIC',
    dlcs24: 'DLCS24',
    lndb: 'LNDb',
    midrc_ricord: 'MIDRC',
    radchestct: 'RAD-ChestCT',
    ctrate: 'CT-RATE'
  };

  function heatChart(P: typeof PlotNS, width: number) {
    if (!cohorts) return null;
    const cells = cohorts.cells.filter((c) => c.probe === heatProbe);
    const p0 = new Map(cells.filter((c) => c.model === 'Pillar-0').map((c) => [c.cohort, c.auroc]));
    const data = cells.map((c) => ({
      ...c,
      value: heatDelta ? c.auroc - p0.get(c.cohort)! : c.auroc,
      cohortLabel: COHORT_LABELS[c.cohort] ?? c.cohort
    }));
    return P.plot({
      width,
      height: 22 * ROWS.length + 96,
      marginLeft: 100,
      marginTop: 40,
      marginBottom: 8,
      padding: 0.05,
      x: {
        domain: Object.values(COHORT_LABELS),
        label: null,
        axis: 'top',
        tickSize: 0,
        tickFormat: (d: string) =>
          d.replace('RAD-ChestCT', 'RAD-\nChestCT').replace('COCA gated', 'COCA\ngated')
      },
      y: { domain: ROWS, label: null, tickSize: 0 },
      color: heatDelta
        ? {
            type: 'diverging',
            scheme: 'rdbu',
            domain: [-15, 15],
            legend: true,
            label: 'Difference to Pillar-0 (AUROC points)'
          }
        : {
            type: 'linear',
            scheme: 'blues',
            domain: [48, 92],
            legend: true,
            label: 'Macro AUROC (%)'
          },
      marks: [
        P.cell(data, { x: 'cohortLabel', y: 'model', fill: 'value' }),
        P.text(data, {
          x: 'cohortLabel',
          y: 'model',
          text: (d) =>
            heatDelta ? `${d.value >= 0 ? '+' : ''}${d.value.toFixed(1)}` : d.value.toFixed(1),
          fill: (d) =>
            heatDelta
              ? Math.abs(d.value) > 9
                ? 'white'
                : 'black'
              : d.value > 76
                ? 'white'
                : 'black',
          fontSize: 10.5
        }),
        P.tip(
          data,
          P.pointer({
            x: 'cohortLabel',
            y: 'model',
            title: (d) =>
              `${d.model}, ${d.cohortLabel}, ${PROBE_LABEL[heatProbe]}\nAUROC ${d.auroc.toFixed(1)} % (95 % interval ${d.ci[0].toFixed(1)} to ${d.ci[1].toFixed(1)})`
          })
        )
      ]
    });
  }

  // ---- Endpoints -------------------------------------------------------------------------------
  const epCohorts = $derived(
    [...new Map(endpoints.map((e) => [e.cohort, e.title ?? e.cohort])).entries()].map(
      ([id, title]) => ({ id, title })
    )
  );
  const epList = $derived(
    endpoints
      .filter((e) => e.cohort === epCohort)
      .sort((a, b) => Number(a.order ?? 0) - Number(b.order ?? 0))
  );
  const ep = $derived(
    endpoints.find((e) => e.cohort === epCohort && e.endpoint === epName) ?? epList[0] ?? null
  );
  const epCurves = $derived(ep && curves ? (curves.endpoints[ep.endpoint] ?? null) : null);
  const prevalence = $derived(
    prevalenceOverride ??
      (epCurves
        ? epCurves.n_pos / (epCurves.n_pos + epCurves.n_neg)
        : ep?.prevalence
          ? Number(ep.prevalence) / 100
          : 0.1)
  );

  $effect(() => {
    if (ep && ep.endpoint !== epName) epName = ep.endpoint;
  });

  // Students on a teal ramp from small to large, teachers in the thesis palette. All read on both
  // the light and the dark background.
  const MODEL_COLORS: Record<string, string> = {
    TMIN: '#8cc5cf',
    T2: '#3ea0b0',
    T6: '#1b7a8a',
    B0: '#5f6f7f',
    B3: '#8a96a3',
    B6: '#c06a2b',
    ...TEACHER_COLORS
  };
  const CURVE_MODELS = ['TMIN', 'T2', 'T6', 'B0', 'B3', 'B6', 'Pillar-0', 'Curia-1'];
  const shade = (label: string) => MODEL_COLORS[label] ?? '#8a939c';

  function ladderChart(P: typeof PlotNS, width: number) {
    if (!ep) return null;
    const pts = Object.entries(ep.auroc)
      .filter(([m, v]) => v[epProbe] && params[m])
      .map(([m, v]) => ({
        model: m,
        auroc: v[epProbe]![0],
        half: v[epProbe]![1],
        x: log10(params[m]!),
        teacher: !STUDENTS.includes(m)
      }));
    const students = pts.filter((p) => !p.teacher).sort((a, b) => a.x - b.x);
    const teachers = pts.filter((p) => p.teacher && ['Pillar-0', 'Curia-1'].includes(p.model));
    const color = epProbe === 'mlp' ? MLP : LINEAR;
    const lows = pts.map((p) => p.auroc - p.half);
    const highs = pts.map((p) => p.auroc + p.half);
    const lo = Math.max(0, Math.floor(Math.min(45, ...lows) / 5) * 5);
    const hi = Math.min(100, Math.ceil(Math.max(...highs) / 5) * 5);
    return P.plot({
      width,
      height: 280,
      marginLeft: 44,
      marginRight: 70,
      marginBottom: 44,
      x: paramAxis(),
      y: { label: 'AUROC (%)', domain: [lo, hi], grid: true },
      marks: [
        P.ruleY([50], { stroke: '#8a939c', strokeDasharray: '4,3' }),
        P.ruleX(students, {
          x: 'x',
          y1: (d) => d.auroc - d.half,
          y2: (d) => d.auroc + d.half,
          stroke: color,
          strokeOpacity: 0.35
        }),
        P.line(students, { x: 'x', y: 'auroc', stroke: color, strokeWidth: 1.6 }),
        P.dot(students, { x: 'x', y: 'auroc', fill: color, r: 3 }),
        P.ruleX(teachers, {
          x: 'x',
          y1: (d) => d.auroc - d.half,
          y2: (d) => d.auroc + d.half,
          stroke: (d) => TEACHER_COLORS[d.model]
        }),
        P.dot(teachers, {
          x: 'x',
          y: 'auroc',
          fill: (d) => TEACHER_COLORS[d.model],
          r: 4.5,
          symbol: 'diamond'
        }),
        P.text(teachers, {
          x: 'x',
          y: 'auroc',
          text: 'model',
          dx: 8,
          textAnchor: 'start',
          fill: (d) => TEACHER_COLORS[d.model]
        }),
        P.tip(
          pts,
          P.pointer({
            x: 'x',
            y: 'auroc',
            title: (d) =>
              `${d.model}: AUROC ${d.auroc.toFixed(1)} % (plus or minus ${d.half.toFixed(1)})`
          })
        )
      ]
    });
  }

  function curveChart(kind: 'roc' | 'pr') {
    return (P: typeof PlotNS, width: number) => {
      if (!epCurves) return null;
      const grid = Array.from({ length: 201 }, (_, i) => i / 200);
      const lines = epModels.flatMap((m) => {
        const c = epCurves!.models[m]?.[epProbe];
        if (!c) return [];
        const ys = kind === 'roc' ? c.roc_tpr : c.pr_precision;
        return grid.map((x, i) => ({ model: m, x, y: ys[i]!, auroc: c.auroc }));
      });
      const base =
        kind === 'roc'
          ? [
              { x: 0, y: 0 },
              { x: 1, y: 1 }
            ]
          : [
              { x: 0, y: prevalence },
              { x: 1, y: prevalence }
            ];
      return P.plot({
        width,
        height: Math.min(340, Math.round(width * 0.9)),
        marginLeft: 44,
        marginBottom: 44,
        x: {
          label: kind === 'roc' ? 'False positive rate (1 − specificity)' : 'Sensitivity (recall)',
          labelAnchor: 'center',
          labelOffset: 34,
          domain: [0, 1]
        },
        y: {
          label: kind === 'roc' ? 'Sensitivity' : 'Positive predictive value',
          domain: [0, 1],
          grid: true
        },
        color: { domain: epModels, range: epModels.map(shade) },
        marks: [
          P.line(base, { x: 'x', y: 'y', stroke: '#8a939c', strokeDasharray: '3,3' }),
          P.line(lines, {
            x: 'x',
            y: 'y',
            stroke: 'model',
            strokeWidth: 1.75,
            curve: kind === 'roc' ? 'step-after' : 'linear'
          }),
          P.tip(
            lines,
            P.pointer({
              x: 'x',
              y: 'y',
              stroke: 'model',
              title: (d) => `${d.model}, AUROC ${(d.auroc * 100).toFixed(1)} %`
            })
          )
        ]
      });
    };
  }

  const opRows = $derived(
    epCurves
      ? epModels
          .map((m) => {
            const c = epCurves!.models[m]?.[epProbe];
            const op = c?.ops.find((o) => o.name === opName);
            if (!c || !op) return null;
            return {
              model: m,
              auroc: c.auroc * 100,
              ...op,
              ...naturalFrequencies(op.sens, op.spec, prevalence)
            };
          })
          .filter((x): x is NonNullable<typeof x> => x !== null)
      : []
  );

  // ---- Deployment and speed --------------------------------------------------------------------
  function deployChart(P: typeof PlotNS, width: number) {
    if (!deploy.length) return null;
    const order = [
      ...STUDENTS,
      'Pillar-0',
      'Curia-1',
      'Curia-2',
      'CT-CLIP',
      'CT-CLIP-VF',
      'TANGERINE'
    ];
    const rows = deploy.filter((d) => order.includes(d.model));
    const data = rows.flatMap((d) => [
      { model: d.model, side: 'CT-RATE (training hospital)', auroc: d[deployProbe].internal },
      { model: d.model, side: 'RAD-ChestCT (other hospital)', auroc: d[deployProbe].external }
    ]);
    return P.plot({
      width,
      height: 24 * rows.length + 70,
      marginLeft: 92,
      marginBottom: 44,
      x: {
        label: 'Macro AUROC (%)',
        domain: [54, 84],
        grid: true,
        labelAnchor: 'center',
        labelOffset: 34
      },
      y: { domain: order.filter((m) => rows.some((r) => r.model === m)), label: null },
      color: {
        domain: ['CT-RATE (training hospital)', 'RAD-ChestCT (other hospital)'],
        range: ['#9aa7b4', '#0e6e85'],
        legend: true
      },
      marks: [
        P.ruleY(rows, {
          y: 'model',
          x1: (d) => d[deployProbe].external,
          x2: (d) => d[deployProbe].internal,
          stroke: '#9aa7b4',
          strokeOpacity: 0.5
        }),
        P.dot(data, { x: 'auroc', y: 'model', fill: 'side', r: 4.5 }),
        P.tip(
          data,
          P.pointer({
            x: 'auroc',
            y: 'model',
            title: (d) => `${d.model}, ${d.side}: ${d.auroc.toFixed(1)} %`
          })
        )
      ]
    });
  }

  function speedChart(P: typeof PlotNS, width: number) {
    if (!inference) return null;
    const rows = inference.records
      .filter((r) => r.rung && r.cpu_speedup_vs_pillar0 && r.gpu_b1_speedup_vs_pillar0)
      .sort((a, b) => a.params_m - b.params_m);
    const data = rows.flatMap((r) => [
      { model: r.label, kind: 'One CPU thread', speedup: r.cpu_speedup_vs_pillar0! },
      { model: r.label, kind: 'NVIDIA A100, batch size 1', speedup: r.gpu_b1_speedup_vs_pillar0! }
    ]);
    return P.plot({
      width,
      height: 24 * rows.length + 70,
      marginLeft: 48,
      marginBottom: 44,
      x: {
        ...logScale('Times faster than Pillar-0', 0.8, 120),
        grid: true,
        labelAnchor: 'center',
        labelOffset: 34
      },
      y: { domain: rows.map((r) => r.label), label: null },
      color: {
        domain: ['One CPU thread', 'NVIDIA A100, batch size 1'],
        range: ['#0e6e85', '#9aa7b4'],
        legend: true
      },
      marks: [
        P.ruleX([0], { stroke: '#4477aa', strokeDasharray: '2,3' }),
        P.ruleY(rows, {
          y: 'label',
          x1: (d) => log10(d.gpu_b1_speedup_vs_pillar0!),
          x2: (d) => log10(d.cpu_speedup_vs_pillar0!),
          stroke: '#9aa7b4',
          strokeOpacity: 0.5
        }),
        P.dot(data, { x: (d) => log10(d.speedup), y: 'model', fill: 'kind', r: 4.5 }),
        P.tip(
          data,
          P.pointer({
            x: (d) => log10(d.speedup),
            y: 'model',
            title: (d) => `${d.model}, ${d.kind}: ${d.speedup.toFixed(1)} times faster`
          })
        )
      ]
    });
  }

  const pct = (v: number) => `${v.toFixed(1)}\u00a0%`;
  const row = (probe: Probe, label: string) =>
    tradeoff?.probes[probe].find((r) => r.label === label);
  const tmin = $derived(inference?.records.find((r) => r.label === 'TMIN'));
  const dep = (label: string) => deploy.find((d) => d.model === label);
</script>

<svelte:head>
  <title>Results, MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <header class="intro">
    <h1>Results</h1>
    <p class="lede">
      The results of the thesis, recomputed from its result files and checked against its tables.
      Each frozen encoder is read out by a linear and an MLP probe trained on each cohort's labels,
      with five-fold cross-validation and patient-level bootstrap intervals.
    </p>
    <nav class="toc" aria-label="On this page">
      {#each SECTIONS as s (s.id)}<a href={`#${s.id}`}>{s.label}</a>{/each}
    </nav>
  </header>
  {#if loadError}<p class="error">{loadError}</p>{/if}

  <section class="section" id="size">
    <div class="section-label">
      <p class="label">Model size</p>
      <h2>External AUROC and model size</h2>
    </div>
    <div class="section-body">
      {#if tradeoff}
        <p>
          How much of the teachers' performance the students keep depends on the probe. With an MLP
          probe, the average external AUROC levels off early: T2, with 0.27 M parameters, reaches
          {pct(row('mlp', 'T2')!.avg_ext)} and B6 {pct(row('mlp', 'B6')!.avg_ext)}, against
          {pct(row('mlp', 'Pillar-0')!.avg_ext)} for <span class="nb">Pillar-0</span>. With a linear
          probe it rises with model size, from {pct(row('linear', 'TMIN')!.avg_ext)} for TMIN to
          {pct(row('linear', 'B6')!.avg_ext)} for B6, against
          {pct(row('linear', 'Pillar-0')!.avg_ext)} for <span class="nb">Pillar-0</span>.
        </p>
        <ul class="legend" aria-label="Legend">
          <li><span class="key solid" style={`--c: ${MLP}`}></span>MLP probe</li>
          <li><span class="key dashed" style={`--c: ${LINEAR}`}></span>Linear probe</li>
          <li><span class="key bar"></span>Spread between cohorts</li>
        </ul>
        <Plot
          make={tradeoffChart}
          label="Average external AUROC against encoder parameters for the MLP and linear probes"
        />
        <p class="hint">
          Average over the ten external cohorts (COCA's two acquisitions count as one). The vertical
          bars show {tradeoff.spread_definition}, a spread between cohorts rather than a confidence
          interval. Thesis Tables 6.2 and 6.3.
        </p>
      {/if}
    </div>
  </section>

  <section class="section" id="cohorts">
    <div class="section-label">
      <p class="label">Cohorts</p>
      <h2>AUROC by cohort</h2>
    </div>
    <div class="section-body">
      <p>
        Macro AUROC over the endpoints of each cohort, for the fourteen students and the six
        teachers the thesis compared.
      </p>
      <div class="controls">
        <div class="seg" role="group" aria-label="Probe">
          <button aria-pressed={heatProbe === 'mlp'} onclick={() => (heatProbe = 'mlp')}
            >MLP probe</button
          >
          <button aria-pressed={heatProbe === 'linear'} onclick={() => (heatProbe = 'linear')}
            >Linear probe</button
          >
        </div>
        <label class="check"
          ><input type="checkbox" bind:checked={heatDelta} /> Show the difference to Pillar-0</label
        >
      </div>
      <div class="scroll">
        <Plot make={heatChart} label="Macro AUROC per cohort and model" />
      </div>
      <p class="hint">
        Hover a cell for its 95&nbsp;% patient-level bootstrap interval. <span class="nb"
          >CT-RATE</span
        >
        is the students' training corpus and is not part of any external average. Thesis Tables 6.2 and
        6.3.
      </p>
    </div>
  </section>

  <section class="section" id="endpoints">
    <div class="section-label">
      <p class="label">Endpoints</p>
      <h2>Single endpoints</h2>
    </div>
    <div class="section-body">
      <div class="controls">
        <label class="field"
          ><span>Cohort</span>
          <select bind:value={epCohort}>
            {#each epCohorts as c (c.id)}<option value={c.id}>{c.title}</option>{/each}
          </select>
        </label>
        <label class="field"
          ><span>Endpoint</span>
          <select bind:value={epName}>
            {#each epList as e (e.endpoint)}<option value={e.endpoint}
                >{e.name ?? e.endpoint}</option
              >{/each}
          </select>
        </label>
        <div class="field">
          <span>Probe</span>
          <div class="seg" role="group" aria-label="Probe">
            <button aria-pressed={epProbe === 'mlp'} onclick={() => (epProbe = 'mlp')}>MLP</button>
            <button aria-pressed={epProbe === 'linear'} onclick={() => (epProbe = 'linear')}
              >Linear</button
            >
          </div>
        </div>
      </div>
      {#if ep}
        <dl class="epmeta">
          {#if ep.definition}<dt>Definition</dt>
            <dd>{ep.definition}</dd>{/if}
          {#if ep.threshold}<dt>Positive if</dt>
            <dd>{ep.threshold}</dd>{/if}
          {#if ep.n}
            <dt>Scans</dt>
            <dd>
              {ep.n}{#if ep.prevalence}, of which {ep.prevalence}&nbsp;% positive{/if}
            </dd>
          {/if}
        </dl>
        <Plot
          make={ladderChart}
          label="Endpoint AUROC across the student ladder with the teachers"
        />
        {#if epCurves}
          <div class="pickers" role="group" aria-label="Models in the curves">
            {#each CURVE_MODELS as m (m)}
              <label class="check">
                <input
                  type="checkbox"
                  checked={epModels.includes(m)}
                  onchange={(e) =>
                    (epModels = (e.currentTarget as HTMLInputElement).checked
                      ? [...epModels, m]
                      : epModels.filter((x) => x !== m))}
                />
                <span class="swatch" style={`background: ${shade(m)}`}></span>{m}
              </label>
            {/each}
          </div>
          <div class="two">
            <Plot make={curveChart('roc')} label="ROC curves" />
            <Plot make={curveChart('pr')} label="Precision against recall" />
          </div>
          <div class="controls">
            <label class="field"
              ><span>Operating point</span>
              <select bind:value={opName}>
                <option value="youden">Maximum Youden index</option>
                <option value="sens90">Sensitivity at least 90 %</option>
                <option value="spec90">Specificity at least 90 %</option>
              </select>
            </label>
            <label class="field slider"
              ><span>Prevalence {(prevalence * 100).toFixed(1)} %</span>
              <input
                type="range"
                min="0.005"
                max="0.6"
                step="0.005"
                value={prevalence}
                oninput={(e) =>
                  (prevalenceOverride = Number((e.currentTarget as HTMLInputElement).value))}
              />
            </label>
            {#if prevalenceOverride !== null}<button onclick={() => (prevalenceOverride = null)}
                >Use the cohort's prevalence</button
              >{/if}
          </div>
          <div class="scroll">
            <table>
              <thead>
                <tr>
                  <th rowspan="2" scope="col">Model</th>
                  <th rowspan="2" scope="col" class="num">AUROC</th>
                  <th rowspan="2" scope="col" class="num">Sensitivity</th>
                  <th rowspan="2" scope="col" class="num">Specificity</th>
                  <th rowspan="2" scope="col" class="num">PPV</th>
                  <th rowspan="2" scope="col" class="num">NPV</th>
                  <th colspan="3" scope="colgroup" class="num"
                    >Per 1,000 scans at this prevalence</th
                  >
                </tr>
                <tr>
                  <th scope="col" class="num">True pos.</th>
                  <th scope="col" class="num">False neg.</th>
                  <th scope="col" class="num">False pos.</th>
                </tr>
              </thead>
              <tbody>
                {#each opRows as r (r.model)}
                  <tr>
                    <td>{r.model}</td>
                    <td class="num">{r.auroc.toFixed(1)}</td>
                    <td class="num">{(r.sens * 100).toFixed(0)} %</td>
                    <td class="num">{(r.spec * 100).toFixed(0)} %</td>
                    <td class="num"
                      >{Number.isFinite(r.ppv) ? `${(r.ppv * 100).toFixed(0)} %` : ''}</td
                    >
                    <td class="num"
                      >{Number.isFinite(r.npv) ? `${(r.npv * 100).toFixed(0)} %` : ''}</td
                    >
                    <td class="num">{Math.round(r.tp)}</td>
                    <td class="num">{Math.round(r.fn)}</td>
                    <td class="num">{Math.round(r.fp)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="hint">
            The curves are computed from the out-of-fold predictions of the cross-validated probes;
            no individual case is published. Operating points are chosen on the same predictions, so
            they are optimistic, and probe scores are not calibrated probabilities. These are
            evaluation results of research probes, not a diagnostic test.
          </p>
        {:else}
          <p class="hint">Loading the curves for this cohort.</p>
        {/if}
      {/if}
    </div>
  </section>

  <section class="section" id="deployment">
    <div class="section-label">
      <p class="label">Deployment</p>
      <h2>
        Trained on <span class="nb">CT-RATE</span>, tested on <span class="nb">RAD-ChestCT</span>
      </h2>
    </div>
    <div class="section-body">
      {#if deploy.length}
        <p>
          Probes trained on <span class="nb">CT-RATE</span>, the students' training corpus, are
          applied without change to <span class="nb">RAD-ChestCT</span>, a cohort from another
          hospital, and scored by the macro AUROC over 16 abnormality classes. With the MLP probe,
          TMIN reaches {pct(dep('TMIN')?.mlp.external ?? 0)} on <span class="nb">RAD-ChestCT</span>,
          against {pct(dep('Curia-1')?.mlp.external ?? 0)} for <span class="nb">Curia-1</span> and
          {pct(dep('Pillar-0')?.mlp.external ?? 0)} for <span class="nb">Pillar-0</span>.
        </p>
      {/if}
      <div class="seg" role="group" aria-label="Probe">
        <button aria-pressed={deployProbe === 'mlp'} onclick={() => (deployProbe = 'mlp')}
          >MLP probe</button
        >
        <button aria-pressed={deployProbe === 'linear'} onclick={() => (deployProbe = 'linear')}
          >Linear probe</button
        >
      </div>
      <Plot make={deployChart} label="Macro AUROC on CT-RATE and on RAD-ChestCT per model" />
      <p class="hint">Thesis Table 6.4.</p>
    </div>
  </section>

  <section class="section" id="speed">
    <div class="section-label">
      <p class="label">Speed</p>
      <h2>Inference speed</h2>
    </div>
    <div class="section-body">
      {#if tmin}
        <p>
          On one CPU thread, TMIN encodes a scan {tmin.cpu_speedup_vs_pillar0?.toFixed(1)} times faster
          than <span class="nb">Pillar-0</span>. On an NVIDIA A100 at batch size 1, the same
          comparison gives
          {tmin.gpu_b1_speedup_vs_pillar0} times: the GPU forward pass is limited by memory bandwidth.
        </p>
      {/if}
      <Plot
        make={speedChart}
        label="Speed-up over Pillar-0 on one CPU thread and on an A100 at batch size 1"
      />
      <p class="hint">
        Thesis Table 6.5: Intel Xeon Silver 4214R with one pinned thread and 32-bit floats; NVIDIA
        A100 80GB PCIe with bfloat16. The <a href={resolve('/race')}>CPU race</a> measures the students
        on this computer.
      </p>
    </div>
  </section>

  <section class="section" id="verification">
    <div class="section-label">
      <p class="label">Verification</p>
      <h2>How the numbers were checked</h2>
    </div>
    <div class="section-body">
      <ul class="ledger">
        <li>
          Teachers: all six model wrappers reproduce their published forward pass exactly, and 18 of
          21 published reference results are reproduced within overlapping confidence intervals. The
          three exceptions are fine-tuning results of TANGERINE (thesis Section 6.1).
        </li>
        <li>
          This page: the external averages, their spreads and every cohort AUROC are recomputed from
          the raw result files and match thesis Tables 6.2 and 6.3 to one decimal. The build stops
          on any mismatch.
        </li>
        {#if curvesReport}
          <li>
            Endpoint curves: {curvesReport.checked.toLocaleString('en-US')} combinations of endpoint,
            model and probe are recomputed from the saved out-of-fold predictions and match the thesis
            endpoint tables to one decimal{#if curvesReport.missing.length}; {curvesReport.missing
                .length} combinations have no saved predictions{/if}.
          </li>
        {/if}
        <li>
          Browser models: each ONNX export matches its PyTorch checkpoint with a cosine similarity
          of at least 0.999998. Every encode of a bundled scan in the reading room is compared with
          the embedding of the thesis pipeline in PyTorch; the test suite requires at least 0.9999.
        </li>
      </ul>
    </div>
  </section>
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

  .toc {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem var(--space-3);
    font-size: var(--text-sm);
  }

  .toc a {
    color: var(--muted);
    text-decoration: none;
  }

  .toc a:hover {
    color: var(--accent);
  }

  .section {
    scroll-margin-top: calc(var(--bar-height) + 1rem);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: var(--space-2) var(--space-3);
    margin: 0.6rem 0 var(--space-2);
  }

  .check {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: var(--text-sm);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem var(--space-3);
    list-style: none;
    margin: 0 0 0.4rem;
    padding: 0;
    font-size: var(--text-sm);
    color: var(--muted);
  }

  .legend li {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .key {
    display: inline-block;
    width: 1.6rem;
    height: 0;
    border-top: 2px solid var(--c);
  }

  .key.dashed {
    border-top-style: dashed;
  }

  .key.bar {
    width: 0;
    height: 0.9rem;
    border-top: 0;
    border-left: 2px solid #9aa7b4;
  }

  .scroll {
    overflow-x: auto;
  }

  thead th {
    vertical-align: bottom;
  }

  .epmeta {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 0.2rem 1rem;
    margin: 0 0 var(--space-2);
    max-width: var(--measure);
    font-size: var(--text-sm);
  }

  .epmeta dt {
    color: var(--muted);
  }

  .epmeta dd {
    margin: 0;
  }

  .swatch {
    display: inline-block;
    width: 0.75rem;
    height: 0.75rem;
  }

  .pickers {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.1rem;
    margin: var(--space-2) 0 0.6rem;
  }

  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }

  .slider input {
    width: 11rem;
  }

  .ledger {
    margin: 0;
    padding-left: 1.1rem;
    display: grid;
    gap: 0.6rem;
    max-width: var(--measure);
  }

  .error {
    color: var(--warn);
  }

  @media (max-width: 760px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
</style>
