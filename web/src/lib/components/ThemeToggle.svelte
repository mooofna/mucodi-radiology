<script lang="ts">
  import { onMount } from 'svelte';

  type Theme = 'light' | 'dark';
  let theme = $state<Theme>('light');

  function systemTheme(): Theme {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  onMount(() => {
    const stored = document.documentElement.dataset.theme as Theme | undefined;
    theme = stored ?? systemTheme();
  });

  function toggle() {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // Private mode: the choice lasts for this page only.
    }
    window.dispatchEvent(new CustomEvent('themechange', { detail: theme }));
  }

  const next = $derived(theme === 'dark' ? 'light' : 'dark');
</script>

<button
  class="toggle"
  onclick={toggle}
  aria-label={`Switch to the ${next} theme`}
  title={`Switch to the ${next} theme`}
>
  {#if theme === 'dark'}
    <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"
      ><circle cx="12" cy="12" r="4.5" fill="currentColor" /><g
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        ><path
          d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
        /></g
      ></svg
    >
  {:else}
    <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"
      ><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" fill="currentColor" /></svg
    >
  {/if}
</button>

<style>
  .toggle {
    display: inline-grid;
    place-items: center;
    width: 32px;
    height: 32px;
    padding: 0;
    border: 0;
    color: var(--bar-muted);
  }

  .toggle:hover {
    color: var(--bar-ink);
  }
</style>
