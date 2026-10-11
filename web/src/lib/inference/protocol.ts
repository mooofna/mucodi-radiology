// Messages between the page and an inference worker. One worker holds one ONNX Runtime
// instance; its thread count is fixed at start (ORT keeps one global pool per instance).

export interface ExternalFile {
  path: string;
  url: string;
}

export type Request =
  | { type: 'init'; threads: number; wasmBase: string }
  | { type: 'load'; key: string; url: string; external?: ExternalFile[] }
  | { type: 'input'; key: string; data: Float32Array; shape: number[] }
  | {
      type: 'run';
      model: string;
      input: string;
      warmup: number;
      iters: number;
      outputs?: string[];
    }
  | { type: 'dispose'; key: string };

export interface RunResult {
  times: number[];
  outputs: Record<string, { data: Float32Array; dims: number[] }>;
}

export type Response =
  | { type: 'ready'; threads: number; isolated: boolean }
  | { type: 'progress'; key: string; loaded: number; total: number }
  | { type: 'loaded'; key: string; ms: number; bytes: number }
  | { type: 'stored'; key: string }
  | { type: 'result'; result: RunResult }
  | { type: 'disposed'; key: string }
  | { type: 'error'; message: string };

export type Envelope<T> = T & { id: number };
