<script lang="ts" module>
  export interface Layer {
    key: string;
    name: string;
    /** Either a URL to fetch or an in-memory NIfTI file. */
    url?: string;
    buffer?: ArrayBuffer;
    colormap?: string;
    calMin?: number;
    calMax?: number;
    opacity?: number;
  }

  export type Layout = 'grid' | 'axial' | 'coronal' | 'sagittal' | 'render';
</script>

<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { Niivue, NVImage } from '@niivue/niivue';

  interface Props {
    background: Layer | null;
    overlays?: Layer[];
    window?: { center: number; width: number };
    /** A NiiVue CT transfer function for 3D (its own HU calibration replaces the window). */
    transfer?: string | null;
    layout?: Layout;
    nearest?: boolean;
    label?: string;
    onLocation?: (mm: [number, number, number], vox: [number, number, number], values: number[]) => void;
  }

  let {
    background,
    overlays = [],
    window: win = { center: -600, width: 1500 },
    transfer = null,
    layout = 'grid',
    nearest = false,
    label = 'CT viewer',
    onLocation
  }: Props = $props();

  let canvas: HTMLCanvasElement;
  let nv: Niivue | null = null;
  let mod: typeof import('@niivue/niivue') | null = null;
  let loadedBackground = '';
  let loadedOverlays: string[] = [];
  let status = $state('Loading viewer');
  let busy = 0;

  async function toImage(layer: Layer): Promise<NVImage> {
    const { NVImage } = mod!;
    const name = layer.name.endsWith('.nii') || layer.name.endsWith('.nii.gz') ? layer.name : `${layer.name}.nii`;
    const image = await NVImage.loadFromUrl({
      url: layer.url ?? name,
      name,
      buffer: layer.buffer,
      colormap: layer.colormap ?? 'gray',
      opacity: layer.opacity ?? 1
    });
    if (layer.calMin !== undefined) image.cal_min = layer.calMin;
    if (layer.calMax !== undefined) image.cal_max = layer.calMax;
    return image;
  }

  async function sync() {
    if (!nv || !mod) return;
    const token = ++busy;
    const bgKey = background?.key ?? '';
    const ovKeys = overlays.map((o) => o.key);
    try {
      if (bgKey !== loadedBackground) {
        status = background ? 'Loading scan' : '';
        for (const v of [...nv.volumes]) nv.removeVolume(v);
        loadedOverlays = [];
        if (background) {
          const img = await toImage(background);
          if (token !== busy) return;
          nv.addVolume(img);
        }
        loadedBackground = bgKey;
      }
      if (ovKeys.join('|') !== loadedOverlays.join('|') && nv.volumes.length > 0) {
        for (const v of nv.volumes.slice(1)) nv.removeVolume(v);
        for (const o of overlays) {
          const img = await toImage(o);
          if (token !== busy) return;
          nv.addVolume(img);
        }
        loadedOverlays = ovKeys;
      }
      applyWindow();
      status = '';
    } catch (err) {
      status = err instanceof Error ? err.message : String(err);
    }
  }

  function applyWindow() {
    const base = nv?.volumes[0];
    if (!nv || !base) return;
    if (transfer) {
      nv.setColormap(base.id, transfer);
      const cm = nv.colormapFromKey(transfer) as { min?: number; max?: number };
      base.cal_min = cm.min ?? base.cal_min;
      base.cal_max = cm.max ?? base.cal_max;
    } else {
      if (base.colormap !== 'gray') nv.setColormap(base.id, 'gray');
      base.cal_min = win.center - win.width / 2;
      base.cal_max = win.center + win.width / 2;
    }
    nv.updateGLVolume();
  }

  function applyLayout() {
    if (!nv || !mod) return;
    const { SLICE_TYPE } = mod;
    const map: Record<Layout, number> = {
      grid: SLICE_TYPE.MULTIPLANAR,
      axial: SLICE_TYPE.AXIAL,
      coronal: SLICE_TYPE.CORONAL,
      sagittal: SLICE_TYPE.SAGITTAL,
      render: SLICE_TYPE.RENDER
    };
    nv.setSliceType(map[layout]);
  }

  onMount(() => {
    let disposed = false;
    import('@niivue/niivue').then(async (m) => {
      if (disposed) return;
      mod = m;
      const { Niivue, MULTIPLANAR_TYPE, SHOW_RENDER, SLICE_TYPE } = m;
      nv = new Niivue({
        backColor: [0, 0, 0, 1],
        crosshairColor: [0.39, 0.76, 0.83, 0.85],
        crosshairWidth: 1,
        fontColor: [0.84, 0.87, 0.9, 1],
        textHeight: 0.035,
        isColorbar: false,
        isOrientCube: false,
        show3Dcrosshair: true,
        isRadiologicalConvention: true,
        sagittalNoseLeft: true,
        isOrientationTextVisible: true,
        multiplanarLayout: MULTIPLANAR_TYPE.GRID,
        multiplanarShowRender: SHOW_RENDER.ALWAYS,
        multiplanarPadPixels: 4,
        sliceType: SLICE_TYPE.MULTIPLANAR,
        dragAndDropEnabled: false,
        isNearestInterpolation: nearest,
        loadingText: '',
        logLevel: 'error'
      });
      await nv.attachToCanvas(canvas);
      const report = (data: unknown) => {
        const d = data as { mm?: ArrayLike<number>; vox?: ArrayLike<number>; values?: { value: number }[] };
        if (!onLocation || !d?.mm) return;
        const vox = d.vox ?? [0, 0, 0];
        onLocation([d.mm[0]!, d.mm[1]!, d.mm[2]!], [vox[0]!, vox[1]!, vox[2]!], (d.values ?? []).map((v) => v.value));
      };
      nv.addEventListener('locationChange', (event) => report((event as CustomEvent).detail));
      if (import.meta.env.DEV) (window as unknown as { __nv?: Niivue }).__nv = nv;
      applyLayout();
      await sync();
    });
    return () => {
      disposed = true;
    };
  });

  onDestroy(() => {
    nv?.cleanup();
    nv = null;
  });

  $effect(() => {
    void background?.key;
    void overlays.map((o) => o.key).join('|');
    sync();
  });

  $effect(() => {
    void win.center;
    void win.width;
    void transfer;
    applyWindow();
  });

  $effect(() => {
    void layout;
    applyLayout();
  });

  $effect(() => {
    nv?.setInterpolation(nearest);
  });
</script>

<div class="viewer" role="img" aria-label={label}>
  <canvas bind:this={canvas}></canvas>
  {#if status}<p class="status">{status}</p>{/if}
</div>

<style>
  .viewer {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 320px;
    background: var(--viewer-bg);
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
    outline: none;
  }

  .status {
    position: absolute;
    inset: auto 0 0 0;
    margin: 0;
    max-width: none;
    padding: 0.4rem 0.8rem;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--viewer-dim);
  }
</style>
