<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import Viewer, { type Layer, type Layout } from '#lib/viewer/Viewer.svelte';
  import Ladder, { type Rung } from '#lib/reading/Ladder.svelte';
  import Pipeline, { type Step } from '#lib/reading/Pipeline.svelte';
  import EmbeddingStrip from '#lib/reading/EmbeddingStrip.svelte';
  import { Lane, cosine } from '#lib/inference/engine.ts';
  import { prepare, type PrepResult } from '#lib/volume/prep.ts';
  import { WINDOWS } from '#lib/volume/window.ts';
  import { GRID_NIFTI_DIMS, GRID_SHAPE, gridSpacing } from '#lib/volume/grid.ts';
  import { writeNifti } from '#lib/volume/niftiWrite.ts';
  import {
    assetUrl,
    loadCases,
    loadGoldens,
    loadModels,
    type CaseAsset,
    type CasesManifest,
    type Goldens,
    type ModelsManifest
  } from '#lib/data/manifest.ts';
  import { byLabel, loadInference, loadLadder, type Inference, type LadderRung } from '#lib/data/evidence.ts';
  import {
    deviceFactor,
    formatSeconds,
    loadMeasurements,
    project,
    saveMeasurement,
    type Measurement
  } from '#lib/race/projection.ts';

  type Source =
    | { kind: 'case'; case: CaseAsset; full: boolean }
    | { kind: 'own'; name: string; buffer: ArrayBuffer; detail: string };

  let models = $state<ModelsManifest | null>(null);
  let cases = $state<CasesManifest | null>(null);
  let goldens = $state<Goldens | null>(null);
  let inference = $state<Inference | null>(null);
  let ladder = $state<LadderRung[]>([]);
  let loadError = $state('');

  let source = $state<Source | null>(null);
  let background = $state<Layer | null>(null);
  let prep = $state<PrepResult | null>(null);
  let prepSteps = $state<Step[]>([]);
  let prepError = $state('');

  let rung = $state('tmin');
  let threadsMode = $state<'one' | 'all'>('one');
  let isolated = $state(false);
  let cores = $state(4);

  let runSteps = $state<Step[]>([]);
  let running = $state(false);
  let forwardMs = $state<number | null>(null);
  let embedding = $state<Float32Array | null>(null);
  let goldenCos = $state<number | null>(null);
  let runError = $state('');
  let measurements = $state<Measurement[]>([]);
  let hiddenRun = $state(false);

  let windowIndex = $state(0);
  let presetIndex = $state(0);
  const PRESETS = [
    { label: 'Lungs and airways', cmap: 'ct_airways' },
    { label: 'Soft tissue', cmap: 'ct_soft_tissue' },
    { label: 'Bone', cmap: 'ct_bones' },
    { label: 'Contrast', cmap: 'ct_w_contrast' }
  ];
  let layout = $state<Layout>('grid');
  let viewMode = $state<'ct' | 'model'>('ct');
  let dropActive = $state(false);

  let lane: Lane | null = null;
  const loaded = new Set<string>();
  let inputKey = '';

  const win = $derived(WINDOWS[windowIndex]!);
  const factor = $derived(deviceFactor(measurements));
  const p0 = $derived(inference ? byLabel(inference, 'Pillar-0') : null);
  const c1 = $derived(inference ? byLabel(inference, 'Curia-1') : null);
  const thesisMs = (r: string) => inference?.records.find((x) => x.rung === r)?.cpu_median_ms ?? null;

  const rungs = $derived<Rung[]>(
    ladder.map((l) => {
      const asset = models?.rungs.find((m) => m.rung === l.rung);
      const bytes = asset ? asset.files.core.reduce((s, f) => s + f.bytes, 0) : null;
      const big = l.params_m > 13;
      const lowMemory = typeof navigator !== 'undefined' && (navigator as { deviceMemory?: number }).deviceMemory !== undefined && ((navigator as { deviceMemory?: number }).deviceMemory ?? 8) < 8;
      return {
        rung: l.rung,
        label: l.label,
        params_m: l.params_m,
        cpu_ms: thesisMs(l.rung),
        bytes,
        available: !!asset && !(big && lowMemory),
        reason: big && lowMemory ? 'Needs more memory than this device reports' : undefined
      };
    })
  );

  const modelLayer = $derived.by<Layer | null>(() => {
    if (!prep || viewMode !== 'model') return null;
    // The model-grid HU array is (D, H, W) C order, i.e. a NIfTI with dims (W, H, D).
    return {
      key: `model-${source?.kind === 'case' ? source.case.id : 'own'}-${prep.gridHu.length}`,
      name: 'model-grid.nii',
      buffer: writeNifti(GRID_NIFTI_DIMS, prep.gridAffine, { kind: 'float32', data: prep.gridHu })
    };
  });

  onMount(() => {
    isolated = self.crossOriginIsolated;
    cores = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8));
    measurements = loadMeasurements();
    try {
      if (localStorage.getItem('mucodi-threads') === 'all' && isolated) threadsMode = 'all';
    } catch {
      // ignore
    }
    Promise.all([loadModels(), loadCases(), loadGoldens(), loadInference(), loadLadder()])
      .then(([m, c, g, i, l]) => {
        models = m;
        cases = c;
        goldens = g;
        inference = i;
        ladder = l;
        const first = c.cases.find((x) => x.hero) ?? c.cases[0];
        if (first) selectCase(first, false);
      })
      .catch((e: unknown) => (loadError = e instanceof Error ? e.message : String(e)));
  });

  onDestroy(() => lane?.terminate());  // lane only exists in the browser

  function resetRun() {
    forwardMs = null;
    embedding = null;
    goldenCos = null;
    runError = '';
    runSteps = [];
  }

  async function fetchBuffer(url: string): Promise<ArrayBuffer> {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
    return r.arrayBuffer();
  }

  async function selectCase(c: CaseAsset, full: boolean) {
    source = { kind: 'case', case: c, full };
    background = { key: `view-${c.id}`, name: c.files.view, url: assetUrl(`cases/${c.files.view}`) };
    await runPrep(async () => {
      const file = full && c.files.full ? c.files.full : c.files.grid;
      return { kind: full && c.files.full ? 'native' : 'grid', buffer: await fetchBuffer(assetUrl(`cases/${file}`)), name: file };
    });
  }

  async function runPrep(get: () => Promise<{ kind: 'grid' | 'native'; buffer: ArrayBuffer; name: string }>) {
    prep = null;
    prepError = '';
    resetRun();
    inputKey = '';
    const t0 = performance.now();
    prepSteps = [{ name: 'Download', ms: null, state: 'running' }];
    try {
      const { kind, buffer, name } = await get();
      prepSteps = [{ name: 'Download', ms: performance.now() - t0, state: 'done' }];
      const result = await prepare(kind, buffer, name);
      prepSteps = [...prepSteps, ...result.timings.map((t) => ({ name: t.name, ms: t.ms, state: 'done' as const }))];
      prep = result;
    } catch (e) {
      prepError = e instanceof Error ? e.message : String(e);
    }
  }

  async function ensureLane(): Promise<Lane> {
    const threads = threadsMode === 'all' && isolated ? cores : 1;
    if (lane && lane.threads === threads) return lane;
    lane?.terminate();
    loaded.clear();
    inputKey = '';
    lane = new Lane(threads);
    await lane.ready();
    return lane;
  }

  async function encode() {
    if (!prep || !models || running) return;
    const asset = models.rungs.find((m) => m.rung === rung);
    if (!asset) return;
    running = true;
    resetRun();
    const steps: Step[] = [
      { name: 'Load model', ms: null, state: 'running' },
      { name: 'Copy input', ms: null, state: 'pending' },
      { name: 'Warm-up', ms: null, state: 'pending' },
      { name: 'Forward pass', ms: null, state: 'pending' }
    ];
    runSteps = steps;
    const set = (i: number, patch: Partial<Step>) => {
      runSteps = runSteps.map((s, j) => (j === i ? { ...s, ...patch } : s));
    };
    try {
      const l = await ensureLane();
      const key = `${rung}.core`;
      let t = performance.now();
      if (!loaded.has(key)) {
        const [main, ...rest] = asset.files.core;
        await l.load(
          key,
          assetUrl(`models/${main!.path}`),
          rest.map((f) => ({ path: f.path, url: assetUrl(`models/${f.path}`) })),
          (done, total) => set(0, { note: `${Math.round((100 * done) / (total || 1))} %` })
        );
        loaded.add(key);
      }
      set(0, { ms: performance.now() - t, state: 'done' });
      set(1, { state: 'running' });
      t = performance.now();
      const caseKey = source?.kind === 'case' ? `${source.case.id}:${source.full}` : `own:${source?.name}`;
      if (inputKey !== caseKey) {
        await l.setInput('input', prep.input, [1, 4, ...GRID_SHAPE]);
        inputKey = caseKey;
      }
      set(1, { ms: performance.now() - t, state: 'done' });
      set(2, { state: 'running' });
      let wasHidden = document.visibilityState === 'hidden';
      const watch = () => (wasHidden ||= document.visibilityState === 'hidden');
      document.addEventListener('visibilitychange', watch);
      const result = await l.run(key, 'input', { warmup: 0, iters: 2, outputs: ['embedding'] }).finally(() =>
        document.removeEventListener('visibilitychange', watch)
      );
      hiddenRun = wasHidden;
      set(2, { ms: result.times[0]!, state: 'done' });
      set(3, { ms: result.times[1]!, state: 'done' });
      forwardMs = result.times[1]!;
      embedding = result.outputs.embedding!.data;
      if (source?.kind === 'case' && goldens) {
        const g = goldens.cases[source.case.id]?.[rung];
        if (g) goldenCos = cosine(embedding, g.embedding);
      }
      const th = thesisMs(rung);
      // Background tabs are throttled by the browser; such timings are shown but not kept.
      if (l.threads === 1 && th && !hiddenRun) measurements = saveMeasurement({ rung, liveMs: forwardMs, thesisMs: th });
    } catch (e) {
      runError = e instanceof Error ? e.message : String(e);
      runSteps = runSteps.map((s) => (s.state === 'running' ? { ...s, state: 'pending' } : s));
      lane?.terminate();
      lane = null;
    } finally {
      running = false;
    }
  }

  function setThreads(mode: 'one' | 'all') {
    if (mode === threadsMode) return;
    try {
      localStorage.setItem('mucodi-threads', mode);
    } catch {
      // ignore
    }
    if (mode === 'all' && !self.crossOriginIsolated) {
      // Multi-threaded WASM needs cross-origin isolation; the service worker provides it after
      // one reload (see app.html).
      location.reload();
      return;
    }
    threadsMode = mode;
    lane?.terminate();
    lane = null;
  }

  // Bring your own scan: NIfTI files are read directly; DICOM folders go through dcm2niix (WASM).
  async function collectFiles(items: DataTransferItemList): Promise<File[]> {
    const out: File[] = [];
    const walk = async (entry: FileSystemEntry, path: string): Promise<void> => {
      if (entry.isFile) {
        const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
        Object.defineProperty(file, '_webkitRelativePath', { value: path + file.name });
        out.push(file);
      } else if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        for (;;) {
          const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
          if (!batch.length) break;
          for (const e of batch) await walk(e, `${path}${entry.name}/`);
        }
      }
    };
    const entries = Array.from(items)
      .map((i) => i.webkitGetAsEntry())
      .filter((e): e is FileSystemEntry => !!e);
    for (const e of entries) await walk(e, '');
    return out;
  }

  async function openOwn(files: File[]) {
    if (!files.length) return;
    const nifti = files.find((f) => /\.nii(\.gz)?$/i.test(f.name));
    await runPrep(async () => {
      let file: File;
      let detail: string;
      if (nifti) {
        file = nifti;
        detail = 'NIfTI';
      } else {
        prepSteps = [{ name: 'DICOM to NIfTI', ms: null, state: 'running' }];
        const { Dcm2niix } = await import('@niivue/dcm2niix');
        const d = new Dcm2niix();
        await d.init();
        const out = (await d.input(files).run()) as File[];
        const volumes = out.filter((f) => /\.nii(\.gz)?$/i.test(f.name)).sort((a, b) => b.size - a.size);
        if (!volumes.length) throw new Error('no CT series could be converted from these files');
        file = volumes[0]!;
        detail = `DICOM, ${volumes.length} series found, largest used`;
      }
      const buffer = await file.arrayBuffer();
      source = { kind: 'own', name: file.name, buffer, detail };
      background = { key: `own-${file.name}-${file.size}`, name: file.name, buffer: buffer.slice(0) };
      return { kind: 'native', buffer: buffer.slice(0), name: file.name };
    });
  }

  async function onDrop(event: DragEvent) {
    event.preventDefault();
    dropActive = false;
    if (!event.dataTransfer) return;
    await openOwn(await collectFiles(event.dataTransfer.items));
  }

  function heroToggle(full: boolean) {
    if (source?.kind === 'case') selectCase(source.case, full);
  }

  function onPick(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    openOwn(Array.from(input.files ?? []));
    input.value = '';
  }

  const gridMm = $derived(prep ? gridSpacing(prep.gridAffine) : null);
  const projected = $derived(
    factor && p0?.cpu_median_ms && c1?.cpu_median_ms
      ? { p0: project(p0.cpu_median_ms, factor), c1: project(c1.cpu_median_ms, factor) }
      : null
  );
</script>

<svelte:head>
  <title>Reading room, MuCoDi Chest CT</title>
</svelte:head>

<div class="room" ondragover={(e) => { e.preventDefault(); dropActive = true; }} ondragleave={() => (dropActive = false)} ondrop={onDrop} role="presentation">
  <aside class="worklist" aria-label="Worklist">
    <h2 class="label">Worklist</h2>
    {#if loadError}<p class="error">{loadError}</p>{/if}
    <ul>
      {#each cases?.cases ?? [] as c (c.id)}
        <li>
          <button class="case" class:active={source?.kind === 'case' && source.case.id === c.id} onclick={() => selectCase(c, false)}>
            <span class="cid mono">{c.patient}</span>
            <span class="ctitle">{c.title}</span>
            <span class="cmeta mono">{c.native_shape_xyz.join(' x ')}, {c.spacing_mm[2]} mm slices</span>
          </button>
        </li>
      {/each}
    </ul>
    <div class="own" class:drop={dropActive}>
      <h2 class="label">Your own scan</h2>
      <p class="muted small">Drop a NIfTI file or a DICOM folder here. It is processed in this tab and never uploaded.</p>
      <label class="pick">
        <input type="file" accept=".nii,.gz,.dcm,application/dicom" multiple onchange={onPick} />
        <span>Choose files</span>
      </label>
      <label class="pick">
        <input type="file" webkitdirectory onchange={onPick} />
        <span>Choose a DICOM folder</span>
      </label>
    </div>
    {#if prep}
      <div class="check">
        <h2 class="label">Input check</h2>
        <dl>
          {#if source?.kind === 'case'}
            <dt>Source</dt><dd>{source.case.collection}, {source.case.licence}</dd>
            <dt>Native</dt><dd class="mono">{source.case.native_shape_xyz.join(' x ')}</dd>
            <dt>Spacing</dt><dd class="mono">{source.case.spacing_mm.map((v) => v.toFixed(2)).join(' x ')} mm</dd>
            <dt>Extent</dt><dd class="mono">{source.case.extent_mm.map((v) => Math.round(v)).join(' x ')} mm</dd>
          {:else if source?.kind === 'own' && prep.native}
            <dt>Source</dt><dd>{source.detail}</dd>
            <dt>Native</dt><dd class="mono">{prep.native.shape.join(' x ')} {prep.native.datatype}</dd>
            <dt>Spacing</dt><dd class="mono">{prep.native.zooms.map((v) => v.toFixed(2)).join(' x ')} mm</dd>
            <dt>Extent</dt><dd class="mono">{prep.stats.extentMm?.map((v) => Math.round(v)).join(' x ')} mm</dd>
            <dt>HU range</dt><dd class="mono">{Math.round(prep.stats.minHu)} to {Math.round(prep.stats.maxHu)}</dd>
          {/if}
          <dt>Model grid</dt><dd class="mono">{GRID_SHAPE.join(' x ')} (D x H x W)</dd>
          {#if gridMm}<dt>Grid voxel</dt><dd class="mono">{gridMm.map((v) => v.toFixed(2)).join(' x ')} mm</dd>{/if}
        </dl>
        {#if source?.kind === 'own' && prep.stats.extentMm && prep.stats.extentMm[2] > 450}
          <p class="warn small">The scan covers more than the thorax. The students were trained on chest CT, so this embedding is outside their training distribution.</p>
        {/if}
        {#if source?.kind === 'case' && source.case.hero}
          <label class="toggle small">
            <input type="checkbox" checked={source.full} onchange={(e) => heroToggle((e.currentTarget as HTMLInputElement).checked)} />
            Run the full preprocessing in this tab (34 MB native scan)
          </label>
        {/if}
      </div>
    {/if}
  </aside>

  <section class="viewer-col" aria-label="Viewer">
    <div class="toolbar">
      {#if layout === 'render'}
        <div class="group" role="group" aria-label="3D preset">
          {#each PRESETS as p, i (p.cmap)}
            <button class:on={presetIndex === i} onclick={() => (presetIndex = i)}>{p.label}</button>
          {/each}
        </div>
      {:else}
        <div class="group" role="group" aria-label="Window">
          {#each WINDOWS as w, i (w.name)}
            <button class:on={windowIndex === i} onclick={() => (windowIndex = i)}>{w.label}</button>
          {/each}
        </div>
      {/if}
      <div class="group" role="group" aria-label="Layout">
        <button class:on={layout === 'grid'} onclick={() => (layout = 'grid')}>2 x 2</button>
        <button class:on={layout === 'axial'} onclick={() => (layout = 'axial')}>Axial</button>
        <button class:on={layout === 'coronal'} onclick={() => (layout = 'coronal')}>Coronal</button>
        <button class:on={layout === 'render'} onclick={() => (layout = 'render')}>3D</button>
      </div>
      <div class="group" role="group" aria-label="Image">
        <button class:on={viewMode === 'ct'} onclick={() => (viewMode = 'ct')}>Scan</button>
        <button class:on={viewMode === 'model'} disabled={!prep} onclick={() => (viewMode = 'model')}>Model input</button>
      </div>
    </div>
    <div class="canvas">
      <Viewer
        background={viewMode === 'model' ? modelLayer : background}
        window={{ center: win.center, width: win.width }}
        transfer={layout === 'render' ? PRESETS[presetIndex]!.cmap : null}
        {layout}
        label="Chest CT, multiplanar and 3D"
      />
    </div>
    <div class="status mono">
      <span>{#if source?.kind === 'case'}{source.case.patient}, {source.case.collection}{:else if source?.kind === 'own'}{source.name}, local file{/if}</span>
      <span>{viewMode === 'model' ? 'Model input grid 128 x 256 x 256' : 'Native scan'}</span>
      <span>{layout === 'render' ? `3D ${PRESETS[presetIndex]!.label}` : `${win.label} W ${win.width} L ${win.center}`}</span>
      <span class="nd">Not for diagnosis</span>
    </div>
  </section>

  <aside class="analysis" aria-label="Analysis">
    <section>
      <h2 class="label">Student encoder</h2>
      {#if rungs.length}<Ladder {rungs} selected={rung} onSelect={(r) => { rung = r; resetRun(); }} />{/if}
    </section>
    <section>
      <div class="threads" role="group" aria-label="CPU threads">
        <button class:on={threadsMode === 'one'} onclick={() => setThreads('one')}>1 thread (as in the thesis)</button>
        <button class:on={threadsMode === 'all'} onclick={() => setThreads('all')}>{cores} threads</button>
      </div>
      <button class="primary encode" disabled={!prep || running} onclick={encode}>
        {running ? 'Encoding' : 'Encode on this CPU'}
      </button>
      {#if prepError}<p class="error">{prepError}</p>{/if}
      {#if runError}<p class="error">{runError}</p>{/if}
    </section>
    <section class="clock-block" aria-live="polite">
      <p class="label">Forward pass on this device</p>
      <p class="clock num">{forwardMs !== null ? (forwardMs / 1000).toFixed(2) : '0.00'}<span class="unit"> s</span></p>
      {#if hiddenRun}<p class="warn small">This tab was in the background, which browsers slow down. Keep it in front for a comparable time.</p>{/if}
      {#if forwardMs !== null && projected && p0 && c1}
        <table class="compare">
          <tbody>
            <tr><td>{rungs.find((r) => r.rung === rung)?.label}, this device</td><td class="num">{formatSeconds(forwardMs)}</td></tr>
            <tr class="proj"><td>Pillar-0, projected</td><td class="num">{formatSeconds(projected.p0.ms)}</td></tr>
            <tr class="proj"><td>Curia-1, projected</td><td class="num">{formatSeconds(projected.c1.ms)}</td></tr>
          </tbody>
        </table>
        <p class="muted small">
          Teachers are not run here. Their single-thread time from thesis Table 6.5 (Pillar-0 {formatSeconds(p0.cpu_median_ms ?? 0)}, Curia-1 {formatSeconds(c1.cpu_median_ms ?? 0)}) is scaled by this device's speed on {factor?.n ?? 0} measured student{factor?.n === 1 ? '' : 's'}. <a href="race">Race them</a>.
        </p>
      {/if}
    </section>
    <section>
      <h2 class="label">Pipeline</h2>
      <Pipeline steps={[...prepSteps, ...runSteps]} />
    </section>
    {#if embedding}
      <section>
        <h2 class="label">Embedding, {embedding.length} values</h2>
        <EmbeddingStrip values={embedding} />
        {#if goldenCos !== null}
          <p class="small">
            Matches the thesis pipeline in PyTorch: cosine <span class="num">{goldenCos.toFixed(6)}</span>
          </p>
        {/if}
        <p class="muted small">
          This vector is what the model outputs. It is not a finding. In the thesis, probes trained on labelled cohorts read it out; see the evidence.
        </p>
      </section>
    {/if}
  </aside>
</div>

<style>
  .room {
    display: grid;
    grid-template-columns: 17rem minmax(0, 1fr) 23rem;
    height: calc(100dvh - 44px - 1.6rem);
    min-height: 560px;
  }

  aside,
  section.viewer-col {
    min-height: 0;
  }

  .worklist,
  .analysis {
    overflow-y: auto;
    padding: var(--space-2);
    font-size: var(--text-sm);
  }

  .worklist {
    border-right: 1px solid var(--rule);
  }

  .analysis {
    border-left: 1px solid var(--rule);
    display: grid;
    align-content: start;
    gap: var(--space-3);
  }

  ul {
    list-style: none;
    margin: 0.4rem 0 var(--space-3);
    padding: 0;
    border-top: 1px solid var(--rule-strong);
  }

  .case {
    display: grid;
    width: 100%;
    text-align: left;
    border: 0;
    border-bottom: 1px solid var(--rule);
    padding: 0.5rem 0.4rem;
    gap: 0.1rem;
  }

  .case.active {
    background: var(--bg-sunk);
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .cid {
    font-size: var(--text-xs);
  }

  .cmeta {
    font-size: 0.7rem;
    color: var(--muted);
  }

  .own {
    border-top: 1px solid var(--rule-strong);
    padding-top: 0.4rem;
    margin-bottom: var(--space-3);
  }

  .own.drop {
    outline: 2px dashed var(--accent);
    outline-offset: 4px;
  }

  .pick {
    display: block;
    margin-top: 0.4rem;
  }

  .pick input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
  }

  .pick span {
    display: inline-block;
    border: 1px solid var(--rule-strong);
    padding: 0.25rem 0.6rem;
    cursor: pointer;
    font-size: var(--text-sm);
  }

  .pick input:focus-visible + span {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .check {
    border-top: 1px solid var(--rule-strong);
    padding-top: 0.4rem;
  }

  dl {
    display: grid;
    grid-template-columns: 5.5rem 1fr;
    gap: 0.15rem 0.5rem;
    margin: 0.4rem 0;
    font-size: var(--text-xs);
  }

  dt {
    color: var(--muted);
  }

  dd {
    margin: 0;
  }

  .small {
    font-size: var(--text-xs);
  }

  .warn {
    color: var(--warn);
  }

  .error {
    color: var(--warn);
    font-size: var(--text-sm);
  }

  .toggle {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
    margin-top: 0.4rem;
  }

  .viewer-col {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    background: var(--viewer-bg);
  }

  .status {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.5rem;
    padding: 0.3rem 0.6rem;
    border-top: 1px solid var(--viewer-rule);
    font-size: 0.7rem;
    color: var(--viewer-dim);
  }

  .status .nd {
    margin-left: auto;
    color: var(--viewer-ink);
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--viewer-rule);
  }

  .group {
    display: flex;
  }

  .toolbar button {
    border-color: var(--viewer-rule);
    color: var(--viewer-dim);
    font-size: var(--text-xs);
    padding: 0.2rem 0.55rem;
    margin-left: -1px;
  }

  .toolbar button.on {
    color: var(--viewer-ink);
    border-color: var(--viewer-accent);
    position: relative;
    z-index: 1;
  }

  .canvas {
    position: relative;
    min-height: 0;
  }


  .threads {
    display: flex;
    margin-bottom: 0.5rem;
  }

  .threads button {
    flex: 1;
    font-size: var(--text-xs);
    margin-left: -1px;
  }

  .threads button.on {
    border-color: var(--accent);
    color: var(--accent);
    position: relative;
  }

  .encode {
    width: 100%;
    padding: 0.55rem;
  }

  .clock-block p {
    margin: 0;
  }

  .clock {
    font-size: var(--text-clock);
    line-height: 1.05;
    letter-spacing: -0.02em;
  }

  .unit {
    font-size: 0.45em;
    color: var(--muted);
  }

  .compare {
    margin: 0.5rem 0;
  }

  .compare tr.proj td {
    color: var(--muted);
  }

  @media (max-width: 1100px) {
    .room {
      grid-template-columns: minmax(0, 1fr);
      height: auto;
    }

    .viewer-col {
      order: -1;
      height: 70dvh;
    }

    .worklist,
    .analysis {
      border: 0;
      border-top: 1px solid var(--rule);
    }
  }
</style>
