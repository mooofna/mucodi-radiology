<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import Viewer, { type DisplayMask, type Layer, type Layout } from '#lib/viewer/Viewer.svelte';
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
  import {
    byLabel,
    loadInference,
    loadLadder,
    type Inference,
    type LadderRung
  } from '#lib/data/evidence.ts';
  import { call as netCall, cellAt, tapAffine, type TapGeom } from '#lib/network/network.ts';
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

  let tab = $state<'encode' | 'network'>('encode');
  let rung = $state('tmin');
  let threadsMode = $state<'one' | 'all'>('one');
  let isolated = $state(false);
  let cores = $state(4);
  let autoEncode = false;

  let runSteps = $state<Step[]>([]);
  let running = $state(false);
  let forwardMs = $state<number | null>(null);
  let ranLabel = $state('');
  let ranThreads = $state(1);
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
  const LAYOUTS: { id: Layout; label: string }[] = [
    { id: 'grid', label: '2 × 2' },
    { id: 'axial', label: 'Axial' },
    { id: 'coronal', label: 'Coronal' },
    { id: 'sagittal', label: 'Sagittal' },
    { id: 'render', label: '3D' }
  ];
  let layout = $state<Layout>('grid');
  let showModelInput = $state(false);
  let dropActive = $state(false);

  let lane: Lane | null = null;
  const loaded = new Set<string>();
  let inputKey = '';

  // Network view: stage feature maps of the taps graphs (TMIN, T2, T6, B0).
  const TAPS_RUNGS = ['tmin', 't2', 't6', 'b0'];
  const STAGE_LABELS = [
    'Stem',
    'Stage 1',
    'Stage 2',
    'Stage 3',
    'Stage 4',
    'Stage 5',
    'Stage 6',
    'Stage 7',
    'Head'
  ];
  let netRung = $state('t2');
  let netKey = $state('');
  let netGeometry = $state<TapGeom[]>([]);
  let stageIndex = $state(2);
  let netMode = $state<'pca' | 'similarity'>('pca');
  let netOverlay = $state<Layer | null>(null);
  let netExplained = $state<[number, number, number] | null>(null);
  let netCell = $state<[number, number, number] | null>(null);
  let netBest = $state<{ cell: [number, number, number]; value: number } | null>(null);
  let netBusy = $state('');
  let netError = $state('');
  let netOpacity = $state(0.55);
  let simMin = $state(0.6);
  let netMm: [number, number, number] | null = null;
  let blocks = $state(false);

  const win = $derived(WINDOWS[windowIndex]!);
  const stage = $derived(netGeometry[stageIndex] ?? null);
  // Feature maps are drawn in the slices and the 2 x 2 view; the full 3D view shows anatomy only.
  const overlays = $derived<Layer[]>(
    netOverlay && !showModelInput && layout !== 'render'
      ? [{ ...netOverlay, opacity: netOpacity }]
      : []
  );
  const factor = $derived(deviceFactor(measurements));
  const p0 = $derived(inference ? byLabel(inference, 'Pillar-0') : null);
  const c1 = $derived(inference ? byLabel(inference, 'Curia-1') : null);
  const thesisMs = (r: string) =>
    inference?.records.find((x) => x.rung === r)?.cpu_median_ms ?? null;
  const labelOf = (r: string) => ladder.find((l) => l.rung === r)?.label ?? r.toUpperCase();

  const fmtParams = (p: number) =>
    p < 1 ? `${(p * 1000).toFixed(p < 0.1 ? 1 : 0)} k` : `${p.toFixed(p < 10 ? 2 : 1)} M`;
  const fmtBytes = (b: number) =>
    b < 1e6 ? `${(b / 1e3).toFixed(0)} kB` : `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;
  /** Thesis timings are quoted as in Table 6.5, to two decimals. */
  const fmtThesis = (ms: number) => `${(ms / 1000).toFixed(2)} s`;
  const fmtPct = (v: number) => `${Math.round(v * 100)} %`;
  const signed = (v: number) => (v < 0 ? `−${-v}` : String(v));
  const times = (v: number[]) => v.join(' × ');

  const options = $derived(
    ladder.map((l) => {
      const asset = models?.rungs.find((m) => m.rung === l.rung);
      const bytes = asset ? asset.files.core.reduce((s, f) => s + f.bytes, 0) : null;
      const lowMemory =
        typeof navigator !== 'undefined' &&
        ((navigator as { deviceMemory?: number }).deviceMemory ?? 8) < 8;
      return {
        rung: l.rung,
        label: l.label,
        params: fmtParams(l.params_m),
        bytes,
        available: !!asset && !(l.params_m > 13 && lowMemory)
      };
    })
  );
  const selected = $derived(options.find((o) => o.rung === rung));

  const modelLayer = $derived.by<Layer | null>(() => {
    if (!prep || !showModelInput) return null;
    // The model-grid HU array is (D, H, W) C order, i.e. a NIfTI with dims (W, H, D).
    return {
      key: `model-${source?.kind === 'case' ? source.case.id : 'own'}-${prep.gridHu.length}`,
      name: 'model-grid.nii',
      buffer: writeNifti(GRID_NIFTI_DIMS, prep.gridAffine, { kind: 'float32', data: prep.gridHu })
    };
  });
  const displayMask = $derived<DisplayMask | null>(
    prep && !showModelInput
      ? {
          key: `${caseKey()}:${prep.body.length}`,
          mask: prep.body,
          affine: prep.gridAffine,
          shape: GRID_SHAPE
        }
      : null
  );

  onMount(() => {
    isolated = self.crossOriginIsolated;
    cores = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8));
    measurements = loadMeasurements();
    autoEncode = new URLSearchParams(location.search).get('encode') === '1';
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

  onDestroy(() => lane?.terminate()); // lane only exists in the browser

  function caseKey() {
    return source?.kind === 'case' ? `${source.case.id}:${source.full}` : `own:${source?.name}`;
  }

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
    showModelInput = false;
    background = {
      key: `view-${c.id}`,
      name: c.files.view,
      url: assetUrl(`cases/${c.files.view}`)
    };
    await runPrep(async () => {
      const file = full && c.files.full ? c.files.full : c.files.grid;
      return {
        kind: full && c.files.full ? 'native' : 'grid',
        buffer: await fetchBuffer(assetUrl(`cases/${file}`)),
        name: file
      };
    });
  }

  function resetNetwork() {
    netKey = '';
    netOverlay = null;
    netExplained = null;
    netCell = null;
    netBest = null;
    netError = '';
  }

  async function runPrep(
    get: () => Promise<{ kind: 'grid' | 'native'; buffer: ArrayBuffer; name: string }>
  ) {
    prep = null;
    resetNetwork();
    prepError = '';
    resetRun();
    inputKey = '';
    const t0 = performance.now();
    prepSteps = [{ name: 'Download', ms: null, state: 'running' }];
    try {
      const { kind, buffer, name } = await get();
      prepSteps = [{ name: 'Download', ms: performance.now() - t0, state: 'done' }];
      const result = await prepare(kind, buffer, name);
      prepSteps = [
        ...prepSteps,
        ...result.timings.map((t) => ({ name: t.name, ms: t.ms, state: 'done' as const }))
      ];
      prep = result;
      if (autoEncode) {
        autoEncode = false;
        encode();
      }
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
      if (inputKey !== caseKey()) {
        await l.setInput('input', prep.input, [1, 4, ...GRID_SHAPE]);
        inputKey = caseKey();
      }
      set(1, { ms: performance.now() - t, state: 'done' });
      set(2, { state: 'running' });
      let wasHidden = document.visibilityState === 'hidden';
      const watch = () => (wasHidden ||= document.visibilityState === 'hidden');
      document.addEventListener('visibilitychange', watch);
      const result = await l
        .run(key, 'input', { warmup: 0, iters: 2, outputs: ['embedding'] })
        .finally(() => document.removeEventListener('visibilitychange', watch));
      hiddenRun = wasHidden;
      set(2, { ms: result.times[0]!, state: 'done' });
      set(3, { ms: result.times[1]!, state: 'done' });
      forwardMs = result.times[1]!;
      ranLabel = labelOf(rung);
      ranThreads = l.threads;
      embedding = result.outputs.embedding!.data;
      if (source?.kind === 'case' && goldens) {
        const g = goldens.cases[source.case.id]?.[rung];
        if (g) goldenCos = cosine(embedding, g.embedding);
      }
      const th = thesisMs(rung);
      // Background tabs are throttled by the browser; such timings are shown but not kept.
      if (l.threads === 1 && th && !hiddenRun)
        measurements = saveMeasurement({ rung, liveMs: forwardMs, thesisMs: th });
    } catch (e) {
      runError = e instanceof Error ? e.message : String(e);
      runSteps = runSteps.map((s) => (s.state === 'running' ? { ...s, state: 'pending' } : s));
      lane?.terminate();
      lane = null;
    } finally {
      running = false;
    }
  }

  async function computeNetwork() {
    if (!prep || !models || netBusy) return;
    const asset = models.rungs.find((m) => m.rung === netRung);
    if (!asset?.files.taps || !asset.taps) return;
    netError = '';
    netBusy = 'Loading the model';
    try {
      const l = await ensureLane();
      const key = `${netRung}.taps`;
      if (!loaded.has(key)) {
        await l.load(key, assetUrl(`models/${asset.files.taps[0]!.path}`));
        loaded.add(key);
      }
      if (inputKey !== caseKey()) {
        await l.setInput('input', prep.input, [1, 4, ...GRID_SHAPE]);
        inputKey = caseKey();
      }
      netBusy = 'Running the network';
      const names = asset.taps.map((g) => g.name);
      const result = await l.run(key, 'input', { warmup: 0, iters: 1, outputs: names });
      netBusy = 'Computing the maps';
      const taps = names.map((n) => ({ name: n, data: result.outputs[n]!.data }));
      const k = `${caseKey()}:${netRung}`;
      const body = prep.body.slice();
      await netCall(
        {
          type: 'set',
          key: k,
          taps,
          geometry: $state.snapshot(asset.taps) as TapGeom[],
          gridShape: GRID_SHAPE,
          body
        },
        [...taps.map((x) => x.data.buffer), body.buffer]
      );
      netGeometry = $state.snapshot(asset.taps) as TapGeom[];
      netKey = k;
      netCell = null;
      netBest = null;
      await showStage();
      netBusy = '';
    } catch (e) {
      netError = e instanceof Error ? e.message : String(e);
      netBusy = '';
    }
  }

  async function showStage() {
    if (!netKey || !stage || !prep) return;
    const affine = tapAffine(prep.gridAffine, stage);
    if (netMode === 'pca') {
      const r = await netCall({ type: 'pca', key: netKey, tap: stage.name });
      if (r.type !== 'pca') return;
      netExplained = r.explained;
      netOverlay = {
        key: `pca-${netKey}-${stage.name}`,
        name: `pca-${stage.name}.nii`,
        buffer: writeNifti(r.dims, affine, { kind: 'rgba', data: r.rgba })
      };
    } else if (netCell) {
      const cell: [number, number, number] = [netCell[0], netCell[1], netCell[2]];
      const r = await netCall({ type: 'similarity', key: netKey, tap: stage.name, cell });
      if (r.type !== 'similarity') return;
      netBest = { cell: r.best, value: r.bestValue };
      netOverlay = {
        key: `sim-${netKey}-${stage.name}-${netCell.join('.')}-${simMin}`,
        name: `similarity-${stage.name}.nii`,
        buffer: writeNifti(r.dims, affine, { kind: 'float32', data: r.map }),
        colormap: 'warm',
        calMin: simMin,
        calMax: 1
      };
    } else {
      netOverlay = null;
    }
  }

  function onLocation(mm: [number, number, number]) {
    if (!netKey || !stage || !prep || netMode !== 'similarity' || tab !== 'network') return;
    netMm = mm;
    const cell = cellAt(prep.gridAffine, stage, mm);
    if (netCell && cell.every((v, i) => v === netCell![i])) return;
    netCell = cell;
    showStage();
  }

  function setStage(i: number) {
    stageIndex = i;
    // Keep pointing at the same place in the scan when switching stages.
    if (netMode === 'similarity' && netMm && prep && netGeometry[i])
      netCell = cellAt(prep.gridAffine, netGeometry[i]!, netMm);
    showStage();
  }

  function setNetMode(mode: 'pca' | 'similarity') {
    netMode = mode;
    showStage();
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
        const file = await new Promise<File>((res, rej) =>
          (entry as FileSystemFileEntry).file(res, rej)
        );
        Object.defineProperty(file, '_webkitRelativePath', { value: path + file.name });
        out.push(file);
      } else if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        for (;;) {
          const batch = await new Promise<FileSystemEntry[]>((res, rej) =>
            reader.readEntries(res, rej)
          );
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
        detail = 'NIfTI file';
      } else {
        prepSteps = [{ name: 'DICOM to NIfTI', ms: null, state: 'running' }];
        const { Dcm2niix } = await import('@niivue/dcm2niix');
        const d = new Dcm2niix();
        await d.init();
        const out = (await d.input(files).run()) as File[];
        const volumes = out
          .filter((f) => /\.nii(\.gz)?$/i.test(f.name))
          .sort((a, b) => b.size - a.size);
        if (!volumes.length) throw new Error('No CT series could be converted from these files.');
        file = volumes[0]!;
        detail =
          volumes.length > 1 ? `DICOM folder, largest of ${volumes.length} series` : 'DICOM folder';
      }
      const buffer = await file.arrayBuffer();
      source = { kind: 'own', name: file.name, buffer, detail };
      showModelInput = false;
      background = {
        key: `own-${file.name}-${file.size}`,
        name: file.name,
        buffer: buffer.slice(0)
      };
      return { kind: 'native', buffer: buffer.slice(0), name: file.name };
    });
  }

  async function onDrop(event: DragEvent) {
    event.preventDefault();
    dropActive = false;
    if (!event.dataTransfer) return;
    await openOwn(await collectFiles(event.dataTransfer.items));
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
  const sourceLabel = $derived(
    source?.kind === 'case'
      ? `${source.case.collection}, ${source.case.patient}`
      : source?.kind === 'own'
        ? source.name
        : ''
  );
</script>

<svelte:head>
  <title>Reading room, MuCoDi Chest CT</title>
</svelte:head>

<div
  class="room"
  ondragover={(e) => {
    e.preventDefault();
    dropActive = true;
  }}
  ondragleave={() => (dropActive = false)}
  ondrop={onDrop}
  role="presentation"
>
  <h1 class="visually-hidden">Reading room</h1>

  <aside class="scans" aria-label="Scans">
    <h2 class="label">Scans</h2>
    {#if loadError}<p class="error">{loadError}</p>{/if}
    <ul>
      {#each cases?.cases ?? [] as c (c.id)}
        <li>
          <button
            class="case"
            aria-current={source?.kind === 'case' && source.case.id === c.id ? 'true' : undefined}
            onclick={() => selectCase(c, false)}
          >
            <span class="ctitle">{c.collection}</span>
            <span class="cdesc"
              >{c.title.replace(/^Chest CT, /, '').replace(/^./, (x) => x.toUpperCase())}</span
            >
            <span class="cmeta num">{c.native_shape_xyz[2]} slices of {c.spacing_mm[2]} mm</span>
          </button>
        </li>
      {/each}
      {#if source?.kind === 'own'}
        <li>
          <button class="case" aria-current="true">
            <span class="ctitle">Your scan</span>
            <span class="cdesc">{source.detail}</span>
            <span class="cmeta">{source.name}</span>
          </button>
        </li>
      {/if}
    </ul>

    <div class="own" class:drop={dropActive}>
      <h2 class="label">Your own scan</h2>
      <p class="hint">
        A NIfTI file or a DICOM folder, dropped here or chosen below. It is read in this browser tab
        and never uploaded.
      </p>
      <div class="picks">
        <label class="pick">
          <input type="file" accept=".nii,.gz,.dcm,application/dicom" multiple onchange={onPick} />
          <span>Choose a file</span>
        </label>
        <label class="pick">
          <input type="file" webkitdirectory onchange={onPick} />
          <span>Choose a DICOM folder</span>
        </label>
      </div>
    </div>

    {#if prep}
      <details class="more scan-details">
        <summary>Scan details</summary>
        <dl class="facts">
          {#if source?.kind === 'case'}
            <dt>Patient</dt>
            <dd class="mono id">{source.case.patient}</dd>
            <dt>Licence</dt>
            <dd>{source.case.licence}</dd>
            <dt>Matrix</dt>
            <dd class="num">{times(source.case.native_shape_xyz)}</dd>
            <dt>Voxel size</dt>
            <dd class="num">{times(source.case.spacing_mm.map((v) => Number(v.toFixed(2))))} mm</dd>
            <dt>Field of view</dt>
            <dd class="num">{times(source.case.extent_mm.map((v) => Math.round(v)))} mm</dd>
          {:else if source?.kind === 'own' && prep.native}
            <dt>Matrix</dt>
            <dd class="num">{times(prep.native.shape)}</dd>
            <dt>Voxel size</dt>
            <dd class="num">{times(prep.native.zooms.map((v) => Number(v.toFixed(2))))} mm</dd>
            <dt>Field of view</dt>
            <dd class="num">
              {prep.stats.extentMm ? times(prep.stats.extentMm.map((v) => Math.round(v))) : ''} mm
            </dd>
            <dt>Values</dt>
            <dd class="num">
              {signed(Math.round(prep.stats.minHu))} to {signed(Math.round(prep.stats.maxHu))} HU
            </dd>
          {/if}
          <dt>Model grid</dt>
          <dd class="num">{times([...GRID_SHAPE])} voxels</dd>
          {#if gridMm}
            <dt>Grid voxel</dt>
            <dd class="num">{times(gridMm.map((v) => Number(v.toFixed(2))))} mm</dd>
          {/if}
        </dl>
        {#if source?.kind === 'case' && source.case.hero}
          <label class="toggle">
            <input
              type="checkbox"
              checked={source.full}
              onchange={(e) => {
                if (source?.kind === 'case')
                  selectCase(source.case, (e.currentTarget as HTMLInputElement).checked);
              }}
            />
            <span
              >Preprocess the original scan in this tab (34 MB) instead of the prepared model grid</span
            >
          </label>
        {/if}
      </details>
      {#if source?.kind === 'own' && prep.stats.extentMm && prep.stats.extentMm[2] > 450}
        <p class="warn">
          This scan covers more than the chest. The students were trained on chest CT, so its
          embedding lies outside their training data.
        </p>
      {/if}
    {/if}
  </aside>

  <section class="viewer-col" aria-label="Viewer">
    <div class="toolbar">
      {#if layout === 'render'}
        <div class="vseg" role="group" aria-label="3D preset">
          {#each PRESETS as p, i (p.cmap)}
            <button aria-pressed={presetIndex === i} onclick={() => (presetIndex = i)}
              >{p.label}</button
            >
          {/each}
        </div>
      {:else}
        <div class="vseg" role="group" aria-label="Window">
          {#each WINDOWS as w, i (w.name)}
            <button aria-pressed={windowIndex === i} onclick={() => (windowIndex = i)}
              >{w.label}</button
            >
          {/each}
        </div>
      {/if}
      <div class="vseg" role="group" aria-label="Layout">
        {#each LAYOUTS as l (l.id)}
          <button aria-pressed={layout === l.id} onclick={() => (layout = l.id)}>{l.label}</button>
        {/each}
      </div>
    </div>
    <div class="canvas">
      <Viewer
        background={showModelInput ? modelLayer : background}
        {overlays}
        window={{ center: win.center, width: win.width }}
        transfer={layout === 'render' ? PRESETS[presetIndex]!.cmap : null}
        nearest={!!netOverlay && blocks}
        {layout}
        {displayMask}
        {onLocation}
        label="Chest CT, multiplanar and 3D"
      />
    </div>
    <div class="status">
      <span>{sourceLabel}</span>
      {#if showModelInput}<span>Model input, 128 × 256 × 256</span>{/if}
      <span
        >{layout === 'render'
          ? `3D, ${PRESETS[presetIndex]!.label.toLowerCase()}`
          : `${win.label} window, W ${win.width}, L ${signed(win.center)}`}</span
      >
      <span class="nd">Research use only, not for diagnosis</span>
    </div>
  </section>

  <aside class="panel" aria-label="Analysis">
    <div class="tabs" role="tablist" aria-label="Analysis">
      <button
        role="tab"
        id="tab-encode"
        aria-selected={tab === 'encode'}
        aria-controls="panel-encode"
        onclick={() => (tab = 'encode')}>Encode</button
      >
      <button
        role="tab"
        id="tab-network"
        aria-selected={tab === 'network'}
        aria-controls="panel-network"
        onclick={() => (tab = 'network')}>Network view</button
      >
    </div>

    {#if tab === 'encode'}
      <div class="tabpanel" role="tabpanel" id="panel-encode" aria-labelledby="tab-encode">
        <label class="field">
          <span>Student model</span>
          <select bind:value={rung} onchange={resetRun} disabled={running}>
            {#each options as o (o.rung)}
              <option value={o.rung} disabled={!o.available}
                >{o.label}, {o.params} parameters{o.available ? '' : ' (needs more memory)'}</option
              >
            {/each}
          </select>
        </label>
        {#if selected}
          <p class="hint">
            {#if thesisMs(rung)}{fmtThesis(thesisMs(rung)!)} per scan on one thread of the thesis CPU.{/if}
            {#if selected.bytes}Model file {fmtBytes(selected.bytes)}.{/if}
          </p>
        {/if}

        <div class="field">
          <span>CPU threads</span>
          <div class="seg" role="group" aria-label="CPU threads">
            <button aria-pressed={threadsMode === 'one'} onclick={() => setThreads('one')}
              >1 thread</button
            >
            <button aria-pressed={threadsMode === 'all'} onclick={() => setThreads('all')}
              >{cores} threads</button
            >
          </div>
          <span class="hint">One thread matches the thesis measurements.</span>
        </div>

        <button class="primary block" disabled={!prep || running} onclick={encode}>
          {running ? 'Encoding' : 'Encode this scan'}
        </button>
        {#if prepError}<p class="error">{prepError}</p>{/if}
        {#if runError}<p class="error">{runError}</p>{/if}

        <div class="result" aria-live="polite">
          {#if forwardMs !== null}
            <p class="big num">{(forwardMs / 1000).toFixed(2)}<span class="unit"> s</span></p>
            <p class="cap">
              {ranLabel} on this computer, {ranThreads === 1
                ? 'one thread'
                : `${ranThreads} threads`}
            </p>
            {#if projected && ranThreads === 1}
              <dl class="compare">
                <dt>Pillar-0, projected</dt>
                <dd class="num">{formatSeconds(projected.p0.ms)}</dd>
                <dt>Curia-1, projected</dt>
                <dd class="num">{formatSeconds(projected.c1.ms)}</dd>
              </dl>
              <p class="hint">
                The teachers are not run here. Their single-thread times from the thesis are scaled
                to this computer's speed. <a href={resolve('/race')}>Compare them in the CPU race</a
                >.
              </p>
            {/if}
            {#if hiddenRun}
              <p class="warn">
                The tab was in the background during this run, which browsers slow down. Keep it in
                front for a comparable time.
              </p>
            {/if}
            {#if goldenCos !== null}
              <p class="hint">
                The embedding matches the thesis pipeline in PyTorch (cosine similarity
                <span class="num">{goldenCos.toFixed(6)}</span>).
              </p>
            {/if}
          {:else if p0?.cpu_median_ms && thesisMs(rung)}
            <p class="hint">
              On one thread of the thesis CPU, {labelOf(rung)} needed {fmtThesis(thesisMs(rung)!)} per
              scan and Pillar-0 {fmtThesis(p0.cpu_median_ms)}. The time on this computer will appear
              here.
            </p>
          {/if}
        </div>

        <details class="more">
          <summary>Details</summary>
          <h3>Steps</h3>
          <Pipeline steps={[...prepSteps, ...runSteps]} />
          {#if embedding}
            <h3>Embedding, {embedding.length} values</h3>
            <EmbeddingStrip values={embedding} />
            <p class="hint">
              The embedding is the model's output, a vector that describes the scan. It is not a
              finding; in the thesis, probes trained on labelled cohorts read it out.
            </p>
          {/if}
          <label class="toggle">
            <input type="checkbox" bind:checked={showModelInput} disabled={!prep} />
            <span>Show the model's input in the viewer: the scan resized to 128 × 256 × 256</span>
          </label>
        </details>
      </div>
    {:else}
      <div class="tabpanel" role="tabpanel" id="panel-network" aria-labelledby="tab-network">
        <p class="hint">
          The feature maps of each stage of a student, drawn on the stage's own grid. They show how
          the network represents this scan; they do not explain a prediction.
        </p>
        <div class="field">
          <span>Student model</span>
          <div class="seg" role="group" aria-label="Student model">
            {#each TAPS_RUNGS as r (r)}
              <button
                aria-pressed={netRung === r}
                disabled={!!netBusy}
                onclick={() => {
                  netRung = r;
                  resetNetwork();
                }}>{labelOf(r)}</button
              >
            {/each}
          </div>
        </div>

        {#if !netKey}
          <button class="primary block" disabled={!prep || !!netBusy} onclick={computeNetwork}>
            {netBusy || 'Compute the feature maps'}
          </button>
          {#if netError}<p class="error">{netError}</p>{/if}
        {:else if stage}
          <label class="field">
            <span>{STAGE_LABELS[stageIndex]}</span>
            <input
              type="range"
              min="0"
              max={netGeometry.length - 1}
              step="1"
              value={stageIndex}
              oninput={(e) => setStage(Number((e.currentTarget as HTMLInputElement).value))}
              aria-valuetext={STAGE_LABELS[stageIndex]}
            />
            <span class="hint"
              >{times(stage.shape.slice(1))} cells with {stage.shape[0]} features each, one cell per
              {stage.stride[0]} voxels along each axis.</span
            >
          </label>

          <div class="field">
            <span>Colour by</span>
            <div class="seg" role="group" aria-label="Colour by">
              <button aria-pressed={netMode === 'pca'} onclick={() => setNetMode('pca')}
                >Main components</button
              >
              <button
                aria-pressed={netMode === 'similarity'}
                onclick={() => setNetMode('similarity')}>Similarity to a point</button
              >
            </div>
          </div>
          {#if netMode === 'pca' && netExplained}
            <p class="hint">
              Red, green and blue are the first three principal components of the features of all
              cells inside the body. Together they explain
              {fmtPct(netExplained[0] + netExplained[1] + netExplained[2])} of the variance. Similar colours
              mean similar features.
            </p>
          {:else if netMode === 'similarity'}
            {#if netCell && netBest}
              <p class="hint">
                Cosine similarity of every cell's features to the cell under the crosshair, shown
                from
                {simMin.toFixed(2)} to 1. Click elsewhere in the scan to move it.
              </p>
              <label class="field">
                <span>Show from {simMin.toFixed(2)}</span>
                <input
                  type="range"
                  min="0"
                  max="0.95"
                  step="0.05"
                  bind:value={simMin}
                  onchange={showStage}
                />
              </label>
            {:else}
              <p class="hint">Click a point in the scan to compare every cell with it.</p>
            {/if}
          {/if}
          <label class="field">
            <span>Opacity</span>
            <input type="range" min="0.1" max="1" step="0.05" bind:value={netOpacity} />
          </label>
          {#if layout === 'render'}
            <p class="hint">Feature maps are shown in the slices and in the 2 × 2 view.</p>
          {/if}
          <details class="more">
            <summary>Details</summary>
            <p class="hint">
              The local receptive field of a cell at this stage spans {stage.rf[0]} voxels along each
              axis. Each stage also receives squeeze-and-excitation gates pooled over the whole scan,
              so strictly every cell depends on the whole volume.
            </p>
            <label class="toggle">
              <input type="checkbox" bind:checked={blocks} />
              <span>Draw the cells as blocks, without smoothing between them</span>
            </label>
          </details>
        {/if}
      </div>
    {/if}
  </aside>
</div>

<style>
  .room {
    display: grid;
    grid-template-columns: 15rem minmax(0, 1fr) 21rem;
    height: 100%;
    min-height: 560px;
  }

  aside,
  section.viewer-col {
    min-height: 0;
  }

  .scans,
  .panel {
    overflow-y: auto;
    font-size: var(--text-sm);
  }

  .scans {
    border-right: 1px solid var(--rule);
    padding: var(--space-2);
  }

  .panel {
    border-left: 1px solid var(--rule);
    padding: 0 var(--space-2) var(--space-3);
  }

  .label {
    margin: 0 0 0.5rem;
  }

  ul {
    list-style: none;
    margin: 0 0 var(--space-3);
    padding: 0;
    border-top: 1px solid var(--rule);
  }

  .case {
    display: grid;
    width: 100%;
    text-align: left;
    border: 0;
    border-bottom: 1px solid var(--rule);
    padding: 0.55rem 0.6rem;
    gap: 0.05rem;
  }

  .case:hover {
    background: var(--bg-sunk);
  }

  .case[aria-current='true'] {
    background: var(--bg-sunk);
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .ctitle {
    font-weight: 500;
  }

  .cdesc {
    color: var(--muted);
  }

  .cmeta {
    font-size: var(--text-xs);
    color: var(--muted);
  }

  .facts .id {
    font-size: 0.68rem;
    white-space: normal;
    overflow-wrap: anywhere;
  }

  .own {
    margin-bottom: var(--space-2);
  }

  .own.drop {
    outline: 2px dashed var(--accent);
    outline-offset: 6px;
  }

  .own .hint {
    margin: 0 0 0.6rem;
  }

  .picks {
    display: grid;
    gap: 0.4rem;
  }

  .pick input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
  }

  .pick span {
    display: block;
    border: 1px solid var(--rule-strong);
    padding: 0.35rem 0.7rem;
    cursor: pointer;
    text-align: center;
  }

  .pick span:hover {
    background: var(--bg-sunk);
  }

  .pick input:focus-visible + span {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .scan-details {
    border-top: 1px solid var(--rule);
    padding-top: 0.3rem;
  }

  .facts {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 0.8rem;
    margin: 0.3rem 0 0.6rem;
    font-size: var(--text-xs);
  }

  .facts dt {
    color: var(--muted);
    white-space: nowrap;
  }

  .facts dd {
    margin: 0;
    white-space: nowrap;
  }

  .toggle {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    font-size: var(--text-xs);
    margin: 0.4rem 0;
    color: var(--muted);
  }

  .toggle input {
    margin-top: 0.15rem;
  }

  .warn {
    color: var(--warn);
    font-size: var(--text-xs);
  }

  .error {
    color: var(--warn);
  }

  .viewer-col {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    background: var(--viewer-bg);
    color: var(--viewer-ink);
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.5rem var(--space-2);
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid var(--viewer-rule);
  }

  .vseg {
    display: inline-flex;
    flex-wrap: wrap;
  }

  .vseg button {
    border: 0;
    color: var(--viewer-dim);
    font-size: var(--text-xs);
    padding: 0.3rem 0.6rem;
  }

  .vseg button:hover {
    color: var(--viewer-ink);
  }

  .vseg button[aria-pressed='true'] {
    color: var(--viewer-ink);
    box-shadow: inset 0 -2px 0 var(--viewer-accent);
  }

  .canvas {
    position: relative;
    min-height: 0;
  }

  .status {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1.5rem;
    padding: 0.35rem 0.7rem;
    border-top: 1px solid var(--viewer-rule);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--viewer-dim);
  }

  .status .nd {
    margin-left: auto;
  }

  .tabs {
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 2;
    margin-bottom: var(--space-2);
  }

  .tabpanel {
    display: grid;
    gap: var(--space-2);
    align-content: start;
  }

  .tabpanel > p,
  .field .hint {
    margin: 0;
  }

  .field select {
    width: 100%;
  }

  .seg {
    width: 100%;
  }

  .seg button {
    flex: 1;
  }

  button.block {
    width: 100%;
    padding: 0.6rem;
  }

  .result {
    border-top: 1px solid var(--rule);
    padding-top: var(--space-2);
  }

  .result p {
    margin: 0 0 0.5rem;
  }

  .big {
    font-size: var(--text-clock);
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem;
  }

  .unit {
    font-size: 0.4em;
    font-weight: 400;
    color: var(--muted);
  }

  .cap {
    color: var(--muted);
  }

  .compare {
    display: grid;
    grid-template-columns: 1fr auto;
    margin: 0.6rem 0;
    border-top: 1px solid var(--rule);
  }

  .compare dt,
  .compare dd {
    margin: 0;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--rule);
  }

  .compare dd {
    text-align: right;
  }

  h3 {
    font-size: var(--text-xs);
    font-weight: 500;
    color: var(--muted);
    margin: 0.6rem 0 0.3rem;
  }

  input[type='range'] {
    width: 100%;
  }

  @media (max-width: 1100px) {
    .room {
      grid-template-columns: minmax(0, 1fr);
      height: auto;
    }

    .viewer-col {
      order: -1;
      height: 72dvh;
    }

    .scans,
    .panel {
      border: 0;
      border-top: 1px solid var(--rule);
      overflow: visible;
    }

    .panel {
      padding-top: var(--space-1);
    }
  }
</style>
