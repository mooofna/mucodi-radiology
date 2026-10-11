<script lang="ts">
  import '../app.css';
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import ThemeToggle from '#lib/components/ThemeToggle.svelte';

  let { children } = $props();

  const links = [
    { href: resolve('/'), route: '/', label: 'Reading room' },
    { href: resolve('/race'), route: '/race', label: 'CPU race' },
    { href: resolve('/evidence'), route: '/evidence', label: 'Evidence' },
    { href: resolve('/method'), route: '/method', label: 'Method' },
    { href: resolve('/about'), route: '/about', label: 'About' }
  ];

  const current = (route: string) => page.route.id === route;
</script>

<svelte:head>
  <title>MuCoDi Chest CT</title>
  <meta
    name="description"
    content="Distilled chest CT encoders running on your own CPU, in the browser. Research demo of a bachelor thesis on the parameter-performance trade-off of radiology foundation models."
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
    <a class="ext" href="https://github.com/mooofna/mucodi-radiology">Code</a>
    <ThemeToggle />
  </div>
</header>
<p class="notice" role="note">
  Research demo. Not a medical device. Not for diagnosis or any clinical decision. Scans you open
  stay on this device.
</p>

<main>
  {@render children()}
</main>

<style>
  .bar {
    position: sticky;
    top: 0;
    z-index: 30;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    height: 44px;
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
    letter-spacing: 0.01em;
  }

  .tag {
    font-size: var(--text-xs);
    color: var(--bar-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  nav {
    display: flex;
    gap: var(--space-2);
    overflow-x: auto;
    scrollbar-width: none;
  }

  nav a,
  .ext {
    color: var(--bar-muted);
    text-decoration: none;
    font-size: var(--text-sm);
    padding: 0.7rem 0 0.6rem;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
  }

  nav a:hover,
  .ext:hover {
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

  .notice {
    margin: 0;
    max-width: none;
    padding: 0.3rem var(--gutter);
    font-size: var(--text-xs);
    color: var(--muted);
    border-bottom: 1px solid var(--rule);
    background: var(--bg-sunk);
  }

  @media (max-width: 760px) {
    .tag,
    .ext {
      display: none;
    }
  }
</style>
