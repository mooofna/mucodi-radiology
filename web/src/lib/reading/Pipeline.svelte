<script lang="ts">
  export interface Step {
    name: string;
    ms: number | null;
    state: 'pending' | 'running' | 'done' | 'skipped';
    note?: string;
  }

  let { steps }: { steps: Step[] } = $props();
  const max = $derived(Math.max(1, ...steps.map((s) => s.ms ?? 0)));
</script>

<ol class="pipeline">
  {#each steps as s (s.name)}
    <li class={s.state}>
      <span class="name">{s.name}</span>
      <span class="bar"
        ><span style={`width: ${s.ms ? Math.max(1, (s.ms / max) * 100) : 0}%`}></span></span
      >
      <span class="ms num">
        {#if s.state === 'running'}running{:else if s.ms !== null}{s.ms < 10
            ? s.ms.toFixed(1)
            : Math.round(s.ms)} ms{:else if s.state === 'skipped'}{s.note ??
            'skipped'}{:else}&nbsp;{/if}
      </span>
    </li>
  {/each}
</ol>

<style>
  .pipeline {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: var(--text-sm);
  }

  li {
    display: grid;
    grid-template-columns: 7.5rem 1fr 5.5rem;
    align-items: center;
    gap: 0.6rem;
    padding: 0.18rem 0;
    border-bottom: 1px solid var(--rule);
    color: var(--muted);
  }

  li.done,
  li.running {
    color: var(--ink);
  }

  .bar {
    height: 6px;
    background: var(--bg-sunk);
  }

  .bar span {
    display: block;
    height: 100%;
    background: var(--ink);
  }

  li.running .bar span {
    background: var(--accent);
  }

  .ms {
    text-align: right;
    font-size: var(--text-xs);
  }
</style>
