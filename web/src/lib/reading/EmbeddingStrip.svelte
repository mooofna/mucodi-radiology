<script lang="ts">
  // The embedding as it is: d_S numbers, drawn as signed bars around zero.
  let { values }: { values: Float32Array | number[] } = $props();
  const max = $derived(Math.max(1e-9, ...Array.from(values, (v) => Math.abs(v))));
  const w = 600;
  const h = 56;
</script>

<svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`Embedding with ${values.length} values`}>
  <line x1="0" x2={w} y1={h / 2} y2={h / 2} class="zero" />
  {#each Array.from(values) as v, i (i)}
    <rect
      x={(i / values.length) * w}
      width={Math.max(w / values.length - 0.4, 0.4)}
      y={v >= 0 ? h / 2 - (v / max) * (h / 2 - 2) : h / 2}
      height={(Math.abs(v) / max) * (h / 2 - 2)}
      class={v >= 0 ? 'pos' : 'neg'}
    />
  {/each}
</svg>

<style>
  svg {
    display: block;
    width: 100%;
    height: 56px;
  }

  .zero {
    stroke: var(--rule);
    stroke-width: 1;
  }

  .pos {
    fill: var(--ink);
  }

  .neg {
    fill: var(--muted);
  }
</style>
