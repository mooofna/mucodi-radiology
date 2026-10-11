// Copies the runtime files that must be served unbundled from our own origin:
// the ONNX Runtime WASM binary and loader, and the cross-origin-isolation service worker.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const ortDist = join(web, 'node_modules/onnxruntime-web/dist');
mkdirSync(join(web, 'static/ort'), { recursive: true });
for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  copyFileSync(join(ortDist, f), join(web, 'static/ort', f));
}
copyFileSync(
  join(web, 'node_modules/coi-serviceworker/coi-serviceworker.min.js'),
  join(web, 'static/coi-serviceworker.js')
);
console.log('runtime files copied');
