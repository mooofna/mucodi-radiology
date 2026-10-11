// Page-side client for an inference worker ("lane"). Each lane has its own ONNX Runtime
// instance and thread count; terminating the worker is the only way to cancel a running pass.
import { siteRoot } from '#lib/paths.ts';
import type { Envelope, ExternalFile, Request, Response, RunResult } from './protocol';

type Pending = {
  resolve: (r: Response) => void;
  reject: (e: Error) => void;
  onProgress?: (loaded: number, total: number) => void;
};

export class Lane {
  private worker: Worker;
  private next = 1;
  private pending = new Map<number, Pending>();
  private terminated = false;
  readonly threads: number;
  private readyPromise: Promise<{ threads: number; isolated: boolean }>;

  constructor(threads: number) {
    this.threads = threads;
    this.worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<Envelope<Response>>) => {
      const { id, ...msg } = event.data;
      const p = this.pending.get(id);
      if (!p) return;
      if (msg.type === 'progress') {
        p.onProgress?.(msg.loaded, msg.total);
        return;
      }
      this.pending.delete(id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg as Response);
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'the inference worker stopped (out of memory?)');
      for (const p of this.pending.values()) p.reject(error);
      this.pending.clear();
    };
    const wasmBase = new URL(`${siteRoot()}/ort/`, location.href).href;
    this.readyPromise = this.call({ type: 'init', threads, wasmBase }).then((r) => {
      if (r.type !== 'ready') throw new Error('unexpected reply');
      return { threads: r.threads, isolated: r.isolated };
    });
  }

  ready() {
    return this.readyPromise;
  }

  private call(
    msg: Request,
    transfer: Transferable[] = [],
    onProgress?: Pending['onProgress']
  ): Promise<Response> {
    if (this.terminated) return Promise.reject(new Error('lane was cancelled'));
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ ...msg, id } satisfies Envelope<Request>, transfer);
    });
  }

  async load(
    key: string,
    url: string,
    external: ExternalFile[] = [],
    onProgress?: Pending['onProgress']
  ) {
    await this.readyPromise;
    const r = await this.call({ type: 'load', key, url, external }, [], onProgress);
    if (r.type !== 'loaded') throw new Error('unexpected reply');
    return { ms: r.ms, bytes: r.bytes };
  }

  /** Copies the input into the worker once; later runs reference it by key. */
  async setInput(key: string, data: Float32Array, shape: number[]) {
    await this.readyPromise;
    await this.call({ type: 'input', key, data, shape });
  }

  async run(
    model: string,
    input: string,
    opts: { warmup?: number; iters?: number; outputs?: string[] } = {}
  ): Promise<RunResult> {
    const r = await this.call({
      type: 'run',
      model,
      input,
      warmup: opts.warmup ?? 1,
      iters: opts.iters ?? 1,
      outputs: opts.outputs
    });
    if (r.type !== 'result') throw new Error('unexpected reply');
    return r.result;
  }

  terminate() {
    if (this.terminated) return;
    this.terminated = true;
    this.worker.terminate();
    const error = new Error('lane was cancelled');
    for (const p of this.pending.values()) p.reject(error);
    this.pending.clear();
  }
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return d / Math.sqrt(na * nb);
}
