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

  /** Body mask on the model grid (see volume/body.ts) used to show the scan without its surroundings. */
  export interface DisplayMask {
    key: string;
    mask: Uint8Array;
    /** World affine of the grid stored as (W, H, D). */
    affine: Mat4;
    shape: [number, number, number];
  }

  import type { Mat4 } from '#lib/volume/affine.ts';
</script>

<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { Niivue, NVImage } from '@niivue/niivue';
  import { invert, matmul } from '#lib/volume/affine.ts';

  type ColorMap = Parameters<Niivue['addColormap']>[1];

  interface Props {
    background: Layer | null;
    overlays?: Layer[];
    window?: { center: number; width: number };
    /** A NiiVue CT transfer function for 3D (its own HU calibration replaces the window). */
    transfer?: string | null;
    layout?: Layout;
    nearest?: boolean;
    /** Shows air (-1024 HU) outside the body mask and below -1024 HU. Display only. */
    displayMask?: DisplayMask | null;
    label?: string;
    onLocation?: (
      mm: [number, number, number],
      vox: [number, number, number],
      values: number[]
    ) => void;
  }

  let {
    background,
    overlays = [],
    window: win = { center: -600, width: 1500 },
    transfer = null,
    layout = 'grid',
    nearest = false,
    displayMask = null,
    label = 'CT viewer',
    onLocation
  }: Props = $props();

  let canvas: HTMLCanvasElement;
  let nv: Niivue | null = null;
  let mod: typeof import('@niivue/niivue') | null = null;
  let loadedBackground = '';
  let loadedOverlays: string[] = [];
  let maskedKey = '';
  let status = $state('Loading viewer');
  let busy = 0;

  async function toImage(layer: Layer): Promise<NVImage> {
    const { NVImage } = mod!;
    const name =
      layer.name.endsWith('.nii') || layer.name.endsWith('.nii.gz')
        ? layer.name
        : `${layer.name}.nii`;
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
        maskedKey = '';
      }
      applyDisplayMask();
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
      followCrosshair(); // loading volumes resets the clip plane
      status = '';
    } catch (err) {
      status = err instanceof Error ? err.message : String(err);
    }
  }

  // Slices use only the colour of a lookup table; the volume render also uses its alpha. This ramp
  // has the colour of NiiVue's 'gray' but keeps air and lung (below -500 HU) transparent in 3D, so
  // the render tile of the 2 x 2 layout shows the body rather than a field of view full of fog.
  const RENDER_CUTOFF_HU = -500;

  function windowColormap(lo: number, hi: number): { key: string; cmap: ColorMap } {
    const c = Math.round(Math.min(254, Math.max(0, ((RENDER_CUTOFF_HU - lo) / (hi - lo)) * 255)));
    const ramp = c > 0 ? [0, c, 255] : [0, 255];
    const alpha = c > 0 ? [0, 0, 128] : [0, 128];
    return { key: `mucodi-gray-${c}`, cmap: { R: ramp, G: ramp, B: ramp, A: alpha, I: ramp } };
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
      const lo = win.center - win.width / 2;
      const hi = win.center + win.width / 2;
      const { key, cmap } = windowColormap(lo, hi);
      if (base.colormap !== key) {
        nv.addColormap(key, cmap);
        nv.setColormap(base.id, key);
      }
      base.cal_min = lo;
      base.cal_max = hi;
    }
    nv.updateGLVolume();
  }

  // Scanners pad the corners outside the reconstruction circle with values far below air, which
  // shows as a black disc; the table and cables sit next to the patient. For display, everything
  // outside the body mask and everything below -1024 HU is set to air. The data the model reads is
  // unaffected: preprocessing runs on its own copy in a worker.
  function applyDisplayMask() {
    const base = nv?.volumes[0];
    if (!nv || !base || !displayMask) return;
    const key = `${loadedBackground}|${displayMask.key}`;
    if (maskedKey === key) return;
    const hdr = base.hdr as unknown as {
      dims: number[];
      affine: number[][];
      scl_slope: number;
      scl_inter: number;
    };
    const img = base.img as unknown as ArrayLike<number> & { [i: number]: number };
    if (!img || !hdr?.affine) return;
    const [nx, ny, nz] = [hdr.dims[1]!, hdr.dims[2]!, Math.max(1, hdr.dims[3]!)];
    const slope = hdr.scl_slope || 1;
    const inter = hdr.scl_inter || 0;
    const air = Math.max(
      (-1024 - inter) / slope,
      img instanceof Uint8Array || img instanceof Uint16Array ? 0 : -Infinity
    );
    const toGrid = matmul(invert(displayMask.affine), hdr.affine as Mat4);
    const [D, H, W] = displayMask.shape;
    const m = displayMask.mask;
    const col = (c: number) => [toGrid[0]![c]!, toGrid[1]![c]!, toGrid[2]![c]!];
    const [c0, c1, c2, c3] = [col(0), col(1), col(2), col(3)];
    let index = 0;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++) {
        // Grid coordinates (w, h, d) of voxel (0, j, k); each step in i adds column 0.
        let w = c3[0]! + j * c1[0]! + k * c2[0]!;
        let h = c3[1]! + j * c1[1]! + k * c2[1]!;
        let d = c3[2]! + j * c1[2]! + k * c2[2]!;
        for (let i = 0; i < nx; i++, index++) {
          const wi = Math.round(w);
          const hi = Math.round(h);
          const di = Math.round(d);
          const inside =
            wi >= 0 &&
            wi < W &&
            hi >= 0 &&
            hi < H &&
            di >= 0 &&
            di < D &&
            m[(di * H + hi) * W + wi] === 1;
          if (!inside || img[index]! < air) img[index] = air;
          w += c0[0]!;
          h += c0[1]!;
          d += c0[2]!;
        }
      }
    maskedKey = key;
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
    followCrosshair();
  }

  // In the 2 x 2 layout the render tile is cut along the coronal plane through the crosshair: it
  // shows the coronal tile's slice with the anatomy behind it in depth. The full 3D view is uncut.
  function followCrosshair() {
    if (!nv) return;
    if (layout === 'grid') nv.setClipPlane([nv.scene.crosshairPos[1]! - 0.5, 180, 0]);
    else nv.setClipPlane([2, 0, 0]);
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
        multiplanarEqualSize: true,
        isClipAllVolumes: true, // the 3D cut also opens the network-view overlays
        sliceType: SLICE_TYPE.MULTIPLANAR,
        dragAndDropEnabled: false,
        isNearestInterpolation: nearest,
        loadingText: '',
        logLevel: 'error'
      });
      await nv.attachToCanvas(canvas);
      const report = (data: unknown) => {
        const d = data as {
          mm?: ArrayLike<number>;
          vox?: ArrayLike<number>;
          values?: { value: number }[];
        };
        if (!onLocation || !d?.mm) return;
        const vox = d.vox ?? [0, 0, 0];
        onLocation(
          [d.mm[0]!, d.mm[1]!, d.mm[2]!],
          [vox[0]!, vox[1]!, vox[2]!],
          (d.values ?? []).map((v) => v.value)
        );
      };
      nv.addEventListener('locationChange', (event) => {
        report((event as CustomEvent).detail);
        followCrosshair();
      });
      nv.setClipPlaneColor([0, 0, 0, 0]);
      nv.setRenderAzimuthElevation(150, 15);
      await nv.setVolumeRenderIllumination(0.4);
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

  $effect(() => {
    void displayMask?.key;
    applyDisplayMask();
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
