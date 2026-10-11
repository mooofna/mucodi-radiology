// Page-side wrapper around the preprocessing worker.
import type { PrepRequest, PrepResult } from './prep.worker';

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (r: PrepResult) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./prep.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent<PrepResult | { id: number; error: string }>) => {
    const p = pending.get(event.data.id);
    if (!p) return;
    pending.delete(event.data.id);
    if ('error' in event.data) p.reject(new Error(event.data.error));
    else p.resolve(event.data);
  };
  worker.onerror = (event) => {
    for (const p of pending.values()) p.reject(new Error(event.message || 'preprocessing failed'));
    pending.clear();
    worker = null;
  };
  return worker;
}

export function prepare(kind: PrepRequest['kind'], buffer: ArrayBuffer, name: string): Promise<PrepResult> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, kind, buffer, name } satisfies PrepRequest, [buffer]);
  });
}

export type { PrepResult };
