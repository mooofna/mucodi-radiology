// Types and loaders for the demo assets built by `demo/` (models.json, cases.json, goldens.json).
import { siteRoot } from '#lib/paths.ts';

export interface FileRef {
  path: string;
  bytes: number;
  sha256: string;
}

export interface TapGeometry {
  name: string;
  shape: [number, number, number, number];
  stride: [number, number, number];
  offset: [number, number, number];
  rf: [number, number, number];
}

export interface RungAsset {
  rung: string;
  params: number;
  feat_dim: number;
  files: { core: FileRef[]; taps?: FileRef[] };
  taps: TapGeometry[] | null;
  parity: { cosine: number; max_abs: number };
}

export interface ModelsManifest {
  input: { name: string; shape: number[]; dtype: string };
  rungs: RungAsset[];
}

export interface CaseAsset {
  id: string;
  collection: string;
  patient: string;
  licence: string;
  title: string;
  note: string;
  hero: boolean;
  files: { view: string; grid: string; full?: string };
  native_shape_xyz: [number, number, number];
  spacing_mm: [number, number, number];
  extent_mm: [number, number, number];
  manufacturer?: string | null;
  convolution_kernel?: string | null;
  slice_thickness_mm?: number | null;
}

export interface CasesManifest {
  lidc_attribution: string;
  cases: CaseAsset[];
}

export interface Goldens {
  source: string;
  cases: Record<string, Record<string, { embedding: number[]; grid_cosine: number }>>;
}

export const assetUrl = (path: string) => `${siteRoot()}/assets/${path}`;

const cache = new Map<string, Promise<unknown>>();

export function getJson<T>(path: string): Promise<T> {
  let p = cache.get(path);
  if (!p) {
    p = fetch(assetUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`could not load ${path} (${r.status})`);
      return r.json();
    });
    cache.set(path, p);
  }
  return p as Promise<T>;
}

export const loadModels = () => getJson<ModelsManifest>('models/models.json');
export const loadCases = () => getJson<CasesManifest>('cases/cases.json');
export const loadGoldens = () => getJson<Goldens>('cases/goldens.json');
