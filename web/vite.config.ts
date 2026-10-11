import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// Cross-origin isolation for local dev and preview, so multi-threaded WASM can be tested.
// On GitHub Pages the same headers come from static/coi-serviceworker.js.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp'
};

export default defineConfig({
  plugins: [
    sveltekit({
      adapter: adapter({ fallback: '404.html' }),
      // GitHub Pages serves the site below /<repository>; CI sets BASE_PATH accordingly.
      paths: { base: (process.env.BASE_PATH ?? '') as '' | `/${string}` },
      prerender: { handleHttpError: 'warn' }
    })
  ],
  server: { fs: { allow: ['..'] }, headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web', '@niivue/dcm2niix'] },
  test: { include: ['src/**/*.test.ts'], environment: 'node' }
});
