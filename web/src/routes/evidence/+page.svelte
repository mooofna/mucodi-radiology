<script lang="ts">
  import { onMount } from 'svelte';
  import Plot from '#lib/components/Plot.svelte';
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
  const PROBE_LABEL: Record<Probe, string> = { mlp: 'MLP probe', linear: 'Linear probe' };
  // Log axis over encoder parameters with decade ticks in plain decimals: Plot's SI default prints 0.1
  // as "100m", and its log axis leaves non-decade ticks such as 0.3 unlabelled.
  const PARAM_TICKS = [0.1, 1, 10, 100];
  const paramAxis = (label: string) => ({
    type: 'log' as const,
    label,
    domain: [0.025, 110],
    ticks: PARAM_TICKS,
    tickFormat: String
  });

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

  // ---- Trade-off curve -------------------------------------------------------------------------
  function tradeoffChart(P: typeof PlotNS, width: number) {
    if (!tradeoff) return null;
    const rows = (probe: Probe) =>
      tradeoff!.probes[probe]
        .filter((r) => STUDENTS.includes(r.label))
        .map((r) => ({ ...r, params: params[r.label] ?? NaN, probe }));
    const teacher = (probe: Probe, label: string) =>
      tradeoff!.probes[probe].find((r) => r.label === label)!;
    const floor = teacher('mlp', 'Untrained (B6)').avg_ext;
    const series = [...rows('mlp'), ...rows('linear')];
    return P.plot({
      width,
      height: Math.min(440, width * 0.62),
      marginLeft: 48,
      marginRight: 170,
      x: paramAxis('Encoder parameters (millions)'),
      y: { label: 'Average external AUROC (%)', domain: [50, 76], grid: true },
      marks: [
        P.ruleY([floor], { stroke: '#888', strokeDasharray: '6,3,1,3' }),
        P.text([floor], {
          x: 110,
          y: (d) => d,
          text: () => 'untrained floor',
          textAnchor: 'start',
          dx: 4,
          fill: '#888'
        }),
        ...(['mlp', 'linear'] as const).flatMap((probe) => {
          const p0 = teacher(probe, 'Pillar-0');
          const color = probe === 'mlp' ? MLP : LINEAR;
          return [
            P.ruleY([p0.avg_ext], { stroke: color, strokeDasharray: '2,3', strokeOpacity: 0.8 }),
            P.text([p0.avg_ext], {
              x: 110,
              y: (d) => d,
              text: () =>
                `Pillar-0, ${probe === 'mlp' ? 'MLP' : 'linear'} ${p0.avg_ext.toFixed(1)}`,
              textAnchor: 'start',
              dx: 4,
              fill: color
            }),
            P.ruleX(rows(probe), {
              x: 'params',
              y1: (d) => d.avg_ext - d.spread,
              y2: (d) => d.avg_ext + d.spread,
              stroke: color,
              strokeOpacity: 0.25
            }),
            P.line(rows(probe), {
              x: 'params',
              y: 'avg_ext',
              stroke: color,
              strokeWidth: 1.6,
              strokeDasharray: probe === 'linear' ? '5,3' : undefined
            }),
            P.dot(rows(probe), {
              x: 'params',
              y: 'avg_ext',
              fill: color,
              r: 3.2,
              symbol: probe === 'mlp' ? 'circle' : 'square'
            })
          ];
        }),
        P.text(
          rows('mlp').filter((r) => ['TMIN', 'T2', 'B3', 'B6'].includes(r.label)),
          { x: 'params', y: 'avg_ext', text: 'label', dy: -10, fill: 'currentColor' }
        ),
        P.tip(
          series,
          P.pointer({
            x: 'params',
            y: 'avg_ext',
            title: (d) =>
              `${d.label}, ${PROBE_LABEL[d.probe as Probe]}\n${d.avg_ext.toFixed(1)} +/- ${d.spread.toFixed(1)} over 10 external units\n${d.params} M parameters`
          })
        )
      ]
    });
  }

  // ---- Heatmap ---------------------------------------------------------------------------------
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
    ctrate: 'CT-RATE (in corpus)'
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
      height: 22 * ROWS.length + 70,
      marginLeft: 104,
      marginBottom: 52,
      padding: 0.04,
      x: { domain: Object.values(COHORT_LABELS), label: null, tickRotate: -30 },
      y: { domain: ROWS, label: null },
      color: heatDelta
        ? { type: 'diverging', scheme: 'rdbu', domain: [-15, 15] }
        : { type: 'linear', scheme: 'blues', domain: [48, 92] },
      marks: [
        P.cell(data, { x: 'cohortLabel', y: 'model', fill: 'value' }),
        P.text(data, {
          x: 'cohortLabel',
          y: 'model',
          text: (d) =>
            heatDelta ? `${d.value >= 0 ? '+' : ''}${d.value.toFixed(1)}` : d.value.toFixed(1),
          fill: (d) => (!heatDelta && d.value > 76 ? 'white' : 'black'),
          fontSize: 10
        }),
        P.tip(
          data,
          P.pointer({
            x: 'cohortLabel',
            y: 'model',
            title: (d) =>
              `${d.model} on ${d.cohortLabel}, ${PROBE_LABEL[heatProbe]}\nAUROC ${d.auroc.toFixed(1)} % (95 % CI ${d.ci[0].toFixed(1)} to ${d.ci[1].toFixed(1)})`
          })
        )
      ]
    });
  }

  // ---- Endpoint explorer -----------------------------------------------------------------------
  const epCohorts = $derived([...new Set(endpoints.map((e) => e.cohort))]);
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

  const shade = (label: string) => {
    if (TEACHER_COLORS[label]) return TEACHER_COLORS[label]!;
    const i = STUDENTS.indexOf(label);
    const t = i / (STUDENTS.length - 1);
    const v = Math.round(190 - 170 * t);
    return `rgb(${v - 20}, ${v - 4}, ${v + 20})`;
  };

  function ladderChart(P: typeof PlotNS, width: number) {
    if (!ep) return null;
    const pts = Object.entries(ep.auroc)
      .filter(([m, v]) => v[epProbe] && params[m])
      .map(([m, v]) => ({
        model: m,
        auroc: v[epProbe]![0],
        half: v[epProbe]![1],
        params: params[m]!,
        teacher: !STUDENTS.includes(m)
      }));
    const students = pts.filter((p) => !p.teacher).sort((a, b) => a.params - b.params);
    const teachers = pts.filter((p) => p.teacher && ['Pillar-0', 'Curia-1'].includes(p.model));
    const color = epProbe === 'mlp' ? MLP : LINEAR;
    return P.plot({
      width,
      height: 300,
      marginLeft: 44,
      x: paramAxis('Encoder parameters (millions)'),
      y: { label: 'AUROC (%)', domain: [40, 100], grid: true },
      marks: [
        P.ruleY([50], { stroke: '#888', strokeDasharray: '4,3' }),
        P.ruleX(students, {
          x: 'params',
          y1: (d) => d.auroc - d.half,
          y2: (d) => d.auroc + d.half,
          stroke: color,
          strokeOpacity: 0.35
        }),
        P.line(students, { x: 'params', y: 'auroc', stroke: color }),
        P.dot(students, { x: 'params', y: 'auroc', fill: color, r: 3 }),
        P.ruleX(teachers, {
          x: 'params',
          y1: (d) => d.auroc - d.half,
          y2: (d) => d.auroc + d.half,
          stroke: (d) => TEACHER_COLORS[d.model]
        }),
        P.dot(teachers, {
          x: 'params',
          y: 'auroc',
          fill: (d) => TEACHER_COLORS[d.model],
          r: 4,
          symbol: 'diamond'
        }),
        P.text(teachers, {
          x: 'params',
          y: 'auroc',
          text: 'model',
          dx: -8,
          textAnchor: 'end',
          fill: (d) => TEACHER_COLORS[d.model]
        }),
        P.tip(
          pts,
          P.pointer({
            x: 'params',
            y: 'auroc',
            title: (d) => `${d.model}: ${d.auroc.toFixed(1)} +/- ${d.half.toFixed(1)}`
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
          label:
            kind === 'roc' ? 'False positive rate (1 \u2212 specificity)' : 'Sensitivity (recall)',
          labelAnchor: 'center',
          labelOffset: 36,
          domain: [0, 1]
        },
        y: {
          label: kind === 'roc' ? 'Sensitivity' : 'Positive predictive value',
          domain: [0, 1],
          grid: true
        },
        color: { domain: epModels, range: epModels.map(shade) },
        marks: [
          P.line(base, { x: 'x', y: 'y', stroke: '#888', strokeDasharray: '3,3' }),
          P.line(lines, {
            x: 'x',
            y: 'y',
            stroke: 'model',
            strokeWidth: 1.6,
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

  // ---- Cold deploy and speed ---------------------------------------------------------------------
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
      { model: d.model, side: 'CT-RATE (internal)', auroc: d[deployProbe].internal },
      { model: d.model, side: 'RAD-ChestCT (external)', auroc: d[deployProbe].external }
    ]);
    return P.plot({
      width,
      height: 22 * rows.length + 50,
      marginLeft: 92,
      x: { label: 'Macro AUROC (%)', domain: [54, 84], grid: true },
      y: { domain: order.filter((m) => rows.some((r) => r.model === m)), label: null },
      color: {
        domain: ['CT-RATE (internal)', 'RAD-ChestCT (external)'],
        range: ['#9AA7B4', '#0E6E85'],
        legend: true
      },
      marks: [
        P.ruleY(rows, {
          y: 'model',
          x1: (d) => d[deployProbe].external,
          x2: (d) => d[deployProbe].internal,
          stroke: '#ccc'
        }),
        P.dot(data, { x: 'auroc', y: 'model', fill: 'side', r: 4 }),
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
      { model: r.label, kind: 'A100, batch 1', speedup: r.gpu_b1_speedup_vs_pillar0! }
    ]);
    return P.plot({
      width,
      height: 22 * rows.length + 60,
      marginLeft: 48,
      x: { type: 'log', label: 'Speed-up over Pillar-0 (times)', domain: [0.8, 120], grid: true },
      y: { domain: rows.map((r) => r.label), label: null },
      color: {
        domain: ['One CPU thread', 'A100, batch 1'],
        range: ['#0E6E85', '#9AA7B4'],
        legend: true
      },
      marks: [
        P.ruleX([1], { stroke: '#4477AA', strokeDasharray: '2,3' }),
        P.ruleY(rows, {
          y: 'label',
          x1: (d) => d.gpu_b1_speedup_vs_pillar0!,
          x2: (d) => d.cpu_speedup_vs_pillar0!,
          stroke: '#ccc'
        }),
        P.dot(data, { x: 'speedup', y: 'model', fill: 'kind', r: 4 }),
        P.tip(
          data,
          P.pointer({
            x: 'speedup',
            y: 'model',
            title: (d) => `${d.model}, ${d.kind}: ${d.speedup.toFixed(1)} times`
          })
        )
      ]
    });
  }

  const pct = (v: number) => `${v.toFixed(1)} %`;
  const row = (probe: Probe, label: string) =>
    tradeoff?.probes[probe].find((r) => r.label === label);
  const tmin = $derived(inference?.records.find((r) => r.label === 'TMIN'));
  const dep = (label: string) => deploy.find((d) => d.model === label);
</script>

<svelte:head>
  <title>Evidence, MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <header class="intro">
    <p class="label">Evidence</p>
    <h1>What the students keep, and where.</h1>
    <p class="lede">
      Every number on this page is rebuilt from the thesis result files and checked against the
      thesis tables. Frozen encoders are read out by probes trained on each cohort's labels; AUROC
      is pooled over five-fold cross-validation with patient-clustered bootstrap intervals.
    </p>
  </header>
  {#if loadError}<p class="error">{loadError}</p>{/if}

  <section class="section">
    <div class="section-label">
      <p class="label">Trade-off</p>
      <h2>Parameters against external performance</h2>
    </div>
    <div class="section-body">
      {#if tradeoff}
        <p>
          The trade-off depends on the probe. Under the MLP probe the external average saturates
          early: T2 reaches
          <span class="num">{pct(row('mlp', 'T2')!.avg_ext)}</span> with 0.27 M parameters and B6
          <span class="num">{pct(row('mlp', 'B6')!.avg_ext)}</span>, against Pillar-0's
          <span class="num">{pct(row('mlp', 'Pillar-0')!.avg_ext)}</span>. Under the linear probe it
          climbs with capacity, from
          <span class="num">{pct(row('linear', 'TMIN')!.avg_ext)}</span> at TMIN to
          <span class="num">{pct(row('linear', 'B6')!.avg_ext)}</span> at B6 (Pillar-0
          <span class="num">{pct(row('linear', 'Pillar-0')!.avg_ext)}</span>). In the thesis's
          words, the divergence between the two read-outs, not either curve alone, is the finding.
        </p>
        <Plot
          make={tradeoffChart}
          label="Average external AUROC against encoder parameters for the MLP and linear probes"
        />
        <p class="muted small">
          Average over the ten external cohort units (COCA's two acquisitions count as one). Bars
          show
          {tradeoff.spread_definition}: a spread between cohorts, not a confidence interval.
          Sources: thesis Tables 6.2 and 6.3, recomputed from
          <span class="mono">cross_cohort_cells.csv</span>.
        </p>
      {/if}
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Cohorts</p>
      <h2>Twelve cohorts, every model</h2>
    </div>
    <div class="section-body">
      <div class="controls">
        <div class="seg" role="group" aria-label="Probe">
          <button class:on={heatProbe === 'mlp'} onclick={() => (heatProbe = 'mlp')}
            >MLP probe</button
          >
          <button class:on={heatProbe === 'linear'} onclick={() => (heatProbe = 'linear')}
            >Linear probe</button
          >
        </div>
        <label class="small"
          ><input type="checkbox" bind:checked={heatDelta} /> Difference to Pillar-0</label
        >
      </div>
      <Plot make={heatChart} label="Macro AUROC per cohort and model" />
      <p class="muted small">
        Macro AUROC over each cohort's endpoints (percent). Hover for the 95 % patient-cluster
        bootstrap interval. CT-RATE is the students' training corpus and is excluded from every
        average. Source: <span class="mono">cross_cohort_cells.csv</span>.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Endpoints</p>
      <h2>One endpoint at a time</h2>
    </div>
    <div class="section-body">
      <div class="controls">
        <label class="small"
          >Cohort
          <select bind:value={epCohort}>
            {#each epCohorts as c (c)}<option value={c}>{c}</option>{/each}
          </select>
        </label>
        <label class="small"
          >Endpoint
          <select bind:value={epName}>
            {#each epList as e (e.endpoint)}<option value={e.endpoint}
                >{e.name ?? e.endpoint}</option
              >{/each}
          </select>
        </label>
        <div class="seg" role="group" aria-label="Probe">
          <button class:on={epProbe === 'mlp'} onclick={() => (epProbe = 'mlp')}>MLP</button>
          <button class:on={epProbe === 'linear'} onclick={() => (epProbe = 'linear')}
            >Linear</button
          >
        </div>
      </div>
      {#if ep}
        <dl class="epmeta small">
          <dt>Endpoint</dt>
          <dd>{ep.title ?? ep.cohort}, {ep.name ?? ep.endpoint}</dd>
          {#if ep.definition}<dt>Definition</dt>
            <dd>{ep.definition}</dd>{/if}
          {#if ep.threshold}<dt>Positive if</dt>
            <dd>{ep.threshold}</dd>{/if}
          {#if ep.n}
            <dt>Scans</dt>
            <dd>
              <span class="num">{ep.n}</span>{#if ep.prevalence}, prevalence <span class="num"
                  >{ep.prevalence} %</span
                >{/if}
            </dd>
          {/if}
        </dl>
        <Plot
          make={ladderChart}
          label="Endpoint AUROC across the student ladder with the teachers"
        />
        {#if epCurves}
          <div class="pickers" role="group" aria-label="Models">
            {#each ['TMIN', 'T2', 'T6', 'B0', 'B3', 'B6', 'Pillar-0', 'Curia-1'] as m (m)}
              <label class="small">
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
            <label class="small"
              >Operating point
              <select bind:value={opName}>
                <option value="youden">Maximum Youden index</option>
                <option value="sens90">Sensitivity at least 90 %</option>
                <option value="spec90">Specificity at least 90 %</option>
              </select>
            </label>
            <label class="small slider"
              >Prevalence {(prevalence * 100).toFixed(1)} %
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
            {#if prevalenceOverride !== null}<button
                class="small"
                onclick={() => (prevalenceOverride = null)}>Cohort prevalence</button
              >{/if}
          </div>
          <table>
            <thead>
              <tr>
                <th rowspan="2" scope="col">Model</th>
                <th rowspan="2" scope="col" class="num">AUROC</th>
                <th rowspan="2" scope="col" class="num">Sens.</th>
                <th rowspan="2" scope="col" class="num">Spec.</th>
                <th rowspan="2" scope="col" class="num">PPV</th>
                <th rowspan="2" scope="col" class="num">NPV</th>
                <th colspan="3" scope="colgroup" class="num">Per 1,000 scans at this prevalence</th>
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
                  <td>{r.model}</td><td class="num">{r.auroc.toFixed(1)}</td><td class="num"
                    >{(r.sens * 100).toFixed(0)} %</td
                  ><td class="num">{(r.spec * 100).toFixed(0)} %</td>
                  <td class="num"
                    >{Number.isFinite(r.ppv) ? `${(r.ppv * 100).toFixed(0)} %` : ''}</td
                  ><td class="num"
                    >{Number.isFinite(r.npv) ? `${(r.npv * 100).toFixed(0)} %` : ''}</td
                  >
                  <td class="num">{Math.round(r.tp)}</td><td class="num">{Math.round(r.fn)}</td><td
                    class="num">{Math.round(r.fp)}</td
                  >
                </tr>
              {/each}
            </tbody>
          </table>
          <p class="muted small">
            Curves aggregate the out-of-fold predictions of the cross-validated probes on fixed
            grids; no individual case is published. Operating points are chosen on the same
            predictions, so they are optimistic, and probe scores are not calibrated probabilities.
            These are evaluation results of research probes, not a diagnostic test.
          </p>
        {:else}
          <p class="muted small">Loading curves for this cohort.</p>
        {/if}
      {/if}
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Cold deploy</p>
      <h2>Trained on one hospital, applied to another</h2>
    </div>
    <div class="section-body">
      {#if deploy.length}
        <p>
          Heads trained on CT-RATE applied unchanged to RAD-ChestCT: the external macro AUROC over
          16 abnormality classes. With the MLP head, TMIN reaches <span class="num"
            >{dep('TMIN')?.mlp.external.toFixed(1)}</span
          >
          against Curia-1's
          <span class="num">{dep('Curia-1')?.mlp.external.toFixed(1)}</span> and Pillar-0's
          <span class="num">{dep('Pillar-0')?.mlp.external.toFixed(1)}</span>.
        </p>
      {/if}
      <div class="seg" role="group" aria-label="Probe">
        <button class:on={deployProbe === 'mlp'} onclick={() => (deployProbe = 'mlp')}
          >MLP head</button
        >
        <button class:on={deployProbe === 'linear'} onclick={() => (deployProbe = 'linear')}
          >Linear head</button
        >
      </div>
      <Plot make={deployChart} label="Cold deploy: internal and external macro AUROC per model" />
      <p class="muted small">
        Source: thesis Table 6.4, cross-checked against <span class="mono">deploy_cells.csv</span>.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Inference</p>
      <h2>Where the parameter cut pays off</h2>
    </div>
    <div class="section-body">
      {#if tmin}
        <p>
          On one CPU thread the smallest student is <span class="num"
            >{tmin.cpu_speedup_vs_pillar0?.toFixed(1)}</span
          >
          times faster than Pillar-0; on an A100 at batch size 1 the same cut buys
          <span class="num">{tmin.gpu_b1_speedup_vs_pillar0}</span> times.
        </p>
      {/if}
      <Plot
        make={speedChart}
        label="Speed-up over Pillar-0 on one CPU thread and on an A100 at batch size 1"
      />
      <p class="muted small">
        Source: thesis Table 6.5 (Intel Xeon Silver 4214R, one pinned thread, fp32; NVIDIA A100 80GB
        PCIe, bf16). Live numbers on your own CPU are on the <a href="race">race page</a>.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Verification</p>
      <h2>How these numbers were checked</h2>
    </div>
    <div class="section-body">
      <ul class="ledger">
        <li>
          Thesis teachers: all six wrappers reproduce their upstream forward pass exactly, and 18 of
          21 published anchor results reproduce within confidence-interval overlap; the three
          failures are TANGERINE fine-tuning cells (thesis Section 6.1).
        </li>
        <li>
          This page: the external averages, their spreads and every per-cohort AUROC above were
          recomputed from the raw result files and match thesis Tables 6.2 and 6.3 at one decimal;
          the build stops on any mismatch.
        </li>
        {#if curvesReport}
          <li>
            Endpoint curves: <span class="num">{curvesReport.checked.toLocaleString('en-US')}</span>
            endpoint, model and probe combinations were recomputed from the saved out-of-fold predictions
            and match the thesis endpoint tables at one decimal{#if curvesReport.missing.length};
              {curvesReport.missing.length} combinations have no saved predictions{/if}.
          </li>
        {/if}
        <li>
          The browser models: each ONNX export matches its PyTorch checkpoint (cosine at least
          0.999998). Every encode of a bundled scan in the reading room is compared with the
          embedding the thesis pipeline computes in PyTorch, and the test suite requires a cosine of
          at least 0.9999.
        </li>
      </ul>
    </div>
  </section>
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
    align-items: center;
    gap: var(--space-2);
    margin: 0.4rem 0 0.8rem;
  }

  .controls label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .seg {
    display: flex;
  }

  .seg button {
    font-size: var(--text-xs);
    margin-left: -1px;
  }

  .seg button.on {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
  }

  thead th {
    vertical-align: bottom;
  }

  .epmeta {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 0.15rem 1rem;
    margin: 0 0 var(--space-2);
    max-width: var(--measure);
  }

  .epmeta dt {
    color: var(--muted);
  }

  .epmeta dd {
    margin: 0;
  }

  .swatch {
    display: inline-block;
    width: 0.7rem;
    height: 0.7rem;
    margin-right: 0.35rem;
    vertical-align: -0.05rem;
  }

  .pickers {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1rem;
    margin: 0.6rem 0;
  }

  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }

  .slider input {
    width: 10rem;
    accent-color: var(--accent);
  }

  .small {
    font-size: var(--text-xs);
  }

  .ledger {
    margin: 0;
    padding-left: 1.1rem;
    display: grid;
    gap: 0.5rem;
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
