// See https://svelte.dev/docs/kit/types#app.d.ts
declare global {
  namespace App {}
  interface Window {
    coi?: Record<string, unknown>;
  }
}

export {};
