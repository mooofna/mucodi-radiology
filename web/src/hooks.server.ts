import type { Handle } from '@sveltejs/kit/hooks';

// Dev and preview only (the site is prerendered): cross-origin isolation for multi-threaded WASM.
// On GitHub Pages the same headers come from static/coi-serviceworker.js.
export const handle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);
  response.headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  response.headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return response;
};
