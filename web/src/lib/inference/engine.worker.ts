/// <reference lib="webworker" />
// ONNX Runtime Web (WASM, CPU) inside a dedicated worker. Timing is the wall time of
// session.run on an input that is already resident, after warm-up, as in thesis Table 6.5.
import * as ort from 'onnxruntime-web/wasm';
import type { Envelope, Request, Response } from './protocol';

const sessions = new Map<string, ort.InferenceSession>();
const inputs = new Map<string, ort.Tensor>();

function reply(id: number, msg: Response, transfer: Transferable[] = []): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage({ ...msg, id }, transfer);
}

async function fetchBytes(id: number, key: string, url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`could not load ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    const now = performance.now();
    if (now - last > 100) {
      reply(id, { type: 'progress', key, loaded, total });
      last = now;
    }
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }
  return out;
}

async function handle(id: number, msg: Request): Promise<void> {
  switch (msg.type) {
    case 'init': {
      ort.env.wasm.numThreads = msg.threads;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = msg.wasmBase;
      ort.env.logLevel = 'error';
      reply(id, { type: 'ready', threads: msg.threads, isolated: self.crossOriginIsolated });
      return;
    }
    case 'load': {
      if (sessions.has(msg.key)) {
        reply(id, { type: 'loaded', key: msg.key, ms: 0, bytes: 0 });
        return;
      }
      const model = await fetchBytes(id, msg.key, msg.url);
      let bytes = model.byteLength;
      const externalData: { path: string; data: Uint8Array }[] = [];
      for (const ext of msg.external ?? []) {
        const data = await fetchBytes(id, msg.key, ext.url);
        bytes += data.byteLength;
        externalData.push({ path: ext.path, data });
      }
      const t0 = performance.now();
      const session = await ort.InferenceSession.create(model, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
        ...(externalData.length ? { externalData } : {})
      });
      sessions.set(msg.key, session);
      reply(id, { type: 'loaded', key: msg.key, ms: performance.now() - t0, bytes });
      return;
    }
    case 'input': {
      inputs.set(msg.key, new ort.Tensor('float32', msg.data, msg.shape));
      reply(id, { type: 'stored', key: msg.key });
      return;
    }
    case 'run': {
      const session = sessions.get(msg.model);
      const input = inputs.get(msg.input);
      if (!session || !input) throw new Error('model or input not loaded');
      const feeds = { [session.inputNames[0]!]: input };
      const fetches = msg.outputs ?? session.outputNames;
      for (let i = 0; i < msg.warmup; i++) await session.run(feeds, fetches);
      const times: number[] = [];
      let out: ort.InferenceSession.OnnxValueMapType = {};
      for (let i = 0; i < msg.iters; i++) {
        const t = performance.now();
        out = await session.run(feeds, fetches);
        times.push(performance.now() - t);
      }
      const outputs: Record<string, { data: Float32Array; dims: number[] }> = {};
      const transfer: Transferable[] = [];
      for (const name of fetches) {
        const tensor = out[name] as ort.Tensor;
        const data = tensor.data as Float32Array;
        outputs[name] = { data, dims: [...tensor.dims] };
        transfer.push(data.buffer);
      }
      reply(id, { type: 'result', result: { times, outputs } }, transfer);
      return;
    }
    case 'dispose': {
      await sessions.get(msg.key)?.release();
      sessions.delete(msg.key);
      inputs.delete(msg.key);
      reply(id, { type: 'disposed', key: msg.key });
      return;
    }
  }
}

self.onmessage = (event: MessageEvent<Envelope<Request>>) => {
  const { id, ...msg } = event.data;
  handle(id, msg as Request).catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    reply(id, { type: 'error', message });
  });
};
