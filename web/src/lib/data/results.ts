// Thesis results for the evidence page (built by `uv run mucodi-demo evidence`).
import { getJson } from './manifest';

export interface TradeoffRow {
  model: string;
  label: string;
  avg_ext: number;
  spread: number;
  units: Record<string, number>;
  checked: boolean;
}

export interface Tradeoff {
  units: string[];
  probes: { mlp: TradeoffRow[]; linear: TradeoffRow[] };
  spread_definition: string;
}

export interface CohortMeta {
  cohort: string;
  domain: string;
  n: string;
  n_test: string;
  sites: string;
  years: string;
  contrast: string;
  n_endpoints: string;
  role: string;
}

export interface CohortCell {
  model: string;
  cohort: string;
  probe: 'mlp' | 'linear';
  auroc: number;
  ci: [number, number];
}

export interface Cohorts {
  cohorts: CohortMeta[];
  columns: Record<string, string>;
  cells: CohortCell[];
}

export interface Endpoint {
  cohort: string;
  endpoint: string;
  title?: string;
  name?: string;
  definition?: string;
  threshold?: string;
  n?: string;
  prevalence?: string;
  order?: string;
  auroc: Record<string, Partial<Record<'mlp' | 'linear', [number, number]>>>;
}

export interface Curve {
  auroc: number;
  roc_tpr: number[];
  pr_precision: number[];
  ops: { name: string; sens: number; spec: number }[];
}

export interface CohortCurves {
  cohort: string;
  grid: number;
  endpoints: Record<
    string,
    {
      n_pos: number;
      n_neg: number;
      models: Record<string, Partial<Record<'mlp' | 'linear', Curve>>>;
    }
  >;
}

export interface DeployRow {
  model: string;
  mlp: { internal: number; external: number; external_ci: number };
  linear: { internal: number; external: number; external_ci: number };
}

export const loadTradeoff = () => getJson<Tradeoff>('evidence/tradeoff.json');
export const loadCohorts = () => getJson<Cohorts>('evidence/cohorts.json');
export const loadEndpoints = () => getJson<{ endpoints: Endpoint[] }>('evidence/endpoints.json');
export const loadDeploy = () => getJson<{ rows: DeployRow[] }>('evidence/deploy.json');
export const loadCurves = (cohort: string) =>
  getJson<CohortCurves>(`evidence/curves/${cohort}.json`);

/** Model colours from the thesis palette (code/experiments/analysis/style.py); students are ink. */
export const TEACHER_COLORS: Record<string, string> = {
  'Pillar-0': '#4477AA',
  'Curia-1': '#9E9AC8',
  'Curia-2': '#54278F',
  'CT-CLIP': '#74C476',
  'CT-CLIP-VF': '#006D2C',
  TANGERINE: '#EE6677',
  'Untrained (B6)': '#888888'
};

export const STUDENTS = [
  'TMIN',
  'T0',
  'T1',
  'T2',
  'T3',
  'T4',
  'T5',
  'T6',
  'B0',
  'B2',
  'B3',
  'B4',
  'B5',
  'B6'
];

/** Natural frequencies per 1,000 scans at a prevalence, from sensitivity and specificity. */
export function naturalFrequencies(sens: number, spec: number, prevalence: number, per = 1000) {
  const pos = per * prevalence;
  const neg = per - pos;
  const tp = sens * pos;
  const fn = pos - tp;
  const tn = spec * neg;
  const fp = neg - tn;
  return {
    tp,
    fn,
    fp,
    tn,
    ppv: tp + fp > 0 ? tp / (tp + fp) : NaN,
    npv: tn + fn > 0 ? tn / (tn + fn) : NaN
  };
}
