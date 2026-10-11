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
</script>

<button class="toggle" onclick={toggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
  {theme === 'dark' ? 'Light' : 'Dark'}
</button>

<style>
  .toggle {
    border: 1px solid var(--bar-muted);
    color: var(--bar-ink);
    font-size: var(--text-xs);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
  }
</style>
