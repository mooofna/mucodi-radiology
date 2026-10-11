<script lang="ts">
  // Renders an Observable Plot figure and redraws it when the options or the width change.
  import type * as PlotNS from '@observablehq/plot';
  import { onMount } from 'svelte';

  interface Props {
    make: (plot: typeof PlotNS, width: number) => (SVGSVGElement | HTMLElement) | null;
    label: string;
  }

  let { make, label }: Props = $props();
  let host: HTMLDivElement;
  let width = $state(0);
  let lib: typeof PlotNS | null = $state(null);

  onMount(() => {
    import('@observablehq/plot').then((m) => (lib = m));
  });

  $effect(() => {
    if (!lib || !host || width === 0) return;
    const figure = make(lib, width);
    host.replaceChildren();
    if (!figure) return;
    // Expose each chart as one labelled image; Plot's internal group labels are not valid ARIA on <g>.
    for (const svg of figure.tagName.toLowerCase() === 'svg'
      ? [figure]
      : Array.from(figure.querySelectorAll('svg'))) {
      svg.querySelectorAll('[aria-label]').forEach((el) => el.removeAttribute('aria-label'));
      svg
        .querySelectorAll('[aria-description]')
        .forEach((el) => el.removeAttribute('aria-description'));
    }
    const main =
      figure.tagName.toLowerCase() === 'svg' ? figure : figure.querySelector('svg:last-of-type');
    main?.setAttribute('role', 'img');
    main?.setAttribute('aria-label', label);
    host.append(figure);
  });
</script>

<figure bind:clientWidth={width}>
  <div bind:this={host}></div>
</figure>

<style>
  figure {
    margin: 0;
    width: 100%;
  }

  figure :global(svg) {
    max-width: 100%;
    height: auto;
    overflow: visible;
    font-family: var(--font-sans);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  figure :global([aria-label='tip'] path) {
    fill: var(--bg);
  }
</style>
