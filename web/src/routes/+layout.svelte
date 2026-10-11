<script lang="ts">
  import '../app.css';
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import ThemeToggle from '#lib/components/ThemeToggle.svelte';

  let { children } = $props();

  const links = [
    { href: resolve('/reading-room'), route: '/reading-room', label: 'Reading room' },
    { href: resolve('/race'), route: '/race', label: 'CPU race' },
    { href: resolve('/results'), route: '/results', label: 'Results' },
    { href: resolve('/method'), route: '/method', label: 'Method' },
    { href: resolve('/about'), route: '/about', label: 'About' }
  ];

  const current = (route: string) => page.route.id === route;
  // The reading room fills the window like a workstation; a footer would push the viewer down.
  const workspace = $derived(page.route.id === '/reading-room');
</script>

<svelte:head>
  <title>MuCoDi Chest CT</title>
  <meta
    name="description"
    content="Compact chest CT encoders, distilled from radiology foundation models, running in the browser on your own CPU. Research demo of a bachelor thesis on the parameter-performance trade-off of radiology foundation models."
  />
</svelte:head>

<header class="bar">
  <a class="brand" href={resolve('/')}>
    <span class="name">MuCoDi Chest CT</span>
    <span class="tag">Research demo</span>
  </a>
  <nav aria-label="Main">
    {#each links as link (link.href)}
      <a href={link.href} aria-current={current(link.route) ? 'page' : undefined}>{link.label}</a>
    {/each}
  </nav>
  <div class="tools">
    <a class="author" href="https://mooofna.github.io/">Maurice Heide</a>
    <a
      class="icon"
      href="https://github.com/mooofna/mucodi-radiology"
      aria-label="Source code on GitHub"
    >
      <svg viewBox="0 0 16 16" width="17" height="17" aria-hidden="true"
        ><path
          fill="currentColor"
          d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
        /></svg
      >
    </a>
    <ThemeToggle />
  </div>
</header>

<main class:workspace>
  {@render children()}
</main>

{#if !workspace}
  <footer class="foot">
    <div class="foot-inner">
      <p>
        Research demo for the bachelor thesis <i
          >On the Parameter-Performance Trade-Off of Radiology Foundation Models</i
        >, TU Dresden, 2026. Not a medical device and not for diagnosis or any clinical decision.
      </p>
      <p class="foot-links">
        <a href={resolve('/about')}>About, licences and citation</a>
        <a href="https://github.com/mooofna/mucodi-radiology">Source code</a>
      </p>
    </div>
  </footer>
{/if}

<style>
  .bar {
    position: sticky;
    top: 0;
    z-index: 30;
    display: flex;
    align-items: center;
    gap: var(--space-4);
    height: var(--bar-height);
    padding: 0 var(--gutter);
    background: var(--bar);
    color: var(--bar-ink);
  }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    color: inherit;
    text-decoration: none;
    white-space: nowrap;
  }

  .name {
    font-weight: 600;
  }

  .tag {
    font-size: var(--text-xs);
    color: var(--bar-muted);
  }

  nav {
    display: flex;
    gap: var(--space-3);
  }

  nav a,
  .author {
    color: var(--bar-muted);
    text-decoration: none;
    font-size: var(--text-sm);
    white-space: nowrap;
  }

  nav a {
    padding: 0.75rem 0 0.65rem;
    border-bottom: 2px solid transparent;
  }

  nav a:hover,
  .author:hover,
  .icon:hover {
    color: var(--bar-ink);
  }

  nav a[aria-current='page'] {
    color: var(--bar-ink);
    border-bottom-color: var(--viewer-accent);
  }

  .tools {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .icon {
    display: inline-grid;
    place-items: center;
    width: 32px;
    height: 32px;
    color: var(--bar-muted);
  }

  main.workspace {
    height: calc(100dvh - var(--bar-height));
  }

  .foot {
    border-top: 1px solid var(--rule);
    margin-top: var(--space-5);
  }

  .foot-inner {
    max-width: var(--page);
    margin: 0 auto;
    padding: var(--space-3) var(--gutter) var(--space-4);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: var(--space-1) var(--space-4);
    font-size: var(--text-xs);
    color: var(--muted);
  }

  .foot-inner p {
    margin: 0;
    max-width: 44rem;
  }

  .foot-links {
    display: flex;
    gap: var(--space-2);
  }

  /* Narrow windows: brand and tools on the first row, the five pages on the second. */
  @media (max-width: 900px) {
    .bar {
      flex-wrap: wrap;
      height: auto;
      column-gap: var(--space-2);
      padding-top: 0.4rem;
    }

    nav {
      order: 3;
      width: 100%;
      gap: 1rem;
      overflow-x: auto;
      scrollbar-width: none;
    }

    nav a {
      padding: 0.45rem 0 0.4rem;
    }

    .tag,
    .author {
      display: none;
    }

    main.workspace {
      height: auto;
    }
  }
</style>
