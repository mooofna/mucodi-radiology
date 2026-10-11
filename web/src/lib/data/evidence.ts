// Thesis evidence built by `demo/` (inference.json, ladder.json, sources.json), with provenance.
import { getJson } from './manifest';

export interface InferenceRecord {
  key: string;
  label: string;
  family: 't-line' | 'b-line' | 'teacher' | 'floor';
  rung?: string;
  params_m: number;
  params_src: string;
  flops_g: number;
  cpu_median_ms: number | null;
  cpu_speedup_vs_pillar0?: number;
  gpu_b1_ms?: number;
  gpu_b1_speedup_vs_pillar0?: number;
  gpu_b16_ms?: number;
  gpu_mem_gb?: number;
  flop_cut_vs_pillar0?: number;
  src: string;
}

export interface Inference {
  cpu_envelope: Record<string, string | number | boolean>;
  gpu_envelope: Record<string, string | number | boolean>;
  records: InferenceRecord[];
}

export interface LadderRung {
  rung: string;
  label: string;
  line: 't' | 'b';
  params_m: number;
  d_s: number;
  width: number;
  depth: number;
  blocks: number;
  gflops: number;
  train_gpu_hours: number;
}

export type Sources = Record<string, { file: string; sha256: string; label: string }>;

export const loadInference = () => getJson<Inference>('evidence/inference.json');
export const loadLadder = () => getJson<LadderRung[]>('evidence/ladder.json');
export const loadSources = () => getJson<Sources>('evidence/sources.json');

export function byLabel(inf: Inference, label: string): InferenceRecord {
  const r = inf.records.find((x) => x.label === label);
  if (!r) throw new Error(`no inference record for ${label}`);
  return r;
}
