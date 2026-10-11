<script lang="ts">
  // The 14 students on a log10 parameter axis, as the defense "ladder rungs" figure.
  export interface Rung {
    rung: string;
    label: string;
    params_m: number;
    cpu_ms: number | null;
    bytes: number | null;
    available: boolean;
    reason?: string;
  }

  interface Props {
    rungs: Rung[];
    selected: string;
    onSelect: (rung: string) => void;
  }

  let { rungs, selected, onSelect }: Props = $props();

  const lo = Math.log10(0.02);
  const hi = Math.log10(90);
  const x = (p: number) => ((Math.log10(p) - lo) / (hi - lo)) * 100;
  const current = $derived(rungs.find((r) => r.rung === selected));
  const fmtParams = (p: number) =>
    p < 1 ? `${(p * 1000).toFixed(p < 0.1 ? 1 : 0)} k` : `${p.toFixed(p < 10 ? 2 : 1)} M`;
  const fmtBytes = (b: number) =>
    b < 1e6 ? `${(b / 1e3).toFixed(0)} kB` : `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;

  function key(event: KeyboardEvent) {
    const i = rungs.findIndex((r) => r.rung === selected);
    const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    for (let j = i + step; j >= 0 && j < rungs.length; j += step) {
      if (rungs[j]!.available) {
        onSelect(rungs[j]!.rung);
        return;
      }
    }
  }
</script>

<div class="ladder">
  <div class="axis" role="radiogroup" aria-label="Student model" tabindex="0" onkeydown={key}>
    <div class="line"></div>
    {#each rungs as r (r.rung)}
      <button
        role="radio"
        aria-checked={r.rung === selected}
        aria-label={`${r.label}, ${fmtParams(r.params_m)} parameters`}
        class:sel={r.rung === selected}
        class:tline={r.rung.startsWith('t')}
        disabled={!r.available}
        title={r.available ? `${r.label}: ${fmtParams(r.params_m)} parameters` : r.reason}
        style={`left: ${x(r.params_m)}%`}
        tabindex="-1"
        onclick={() => onSelect(r.rung)}
      ></button>
    {/each}
    <span class="tick" style={`left: ${x(0.1)}%`}>100 k</span>
    <span class="tick" style={`left: ${x(1)}%`}>1 M</span>
    <span class="tick" style={`left: ${x(10)}%`}>10 M</span>
    <span
      class="teacher-mark"
      style={`left: ${x(79)}%`}
      title="Pillar-0, the main teacher: 79 M parameters"
    ></span>
    <span class="teacher-label">Pillar-0 79 M</span>
  </div>
  {#if current}
    <dl class="facts">
      <div>
        <dt>Model</dt>
        <dd class="mono">{current.label}</dd>
      </div>
      <div>
        <dt>Parameters</dt>
        <dd class="num">{fmtParams(current.params_m)}</dd>
      </div>
      <div>
        <dt>File</dt>
        <dd class="num">{current.bytes ? fmtBytes(current.bytes) : 'n/a'}</dd>
      </div>
      <div>
        <dt>Thesis CPU</dt>
        <dd class="num">{current.cpu_ms ? `${(current.cpu_ms / 1000).toFixed(2)} s` : 'n/a'}</dd>
      </div>
    </dl>
  {/if}
</div>

<style>
  .ladder {
    display: grid;
    gap: 0.6rem;
  }

  .axis {
    position: relative;
    height: 46px;
    margin: 0.8rem 0.6rem 0;
  }

  .axis:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 4px;
  }

  .line {
    position: absolute;
    left: 0;
    right: 0;
    top: 14px;
    border-top: 1px solid var(--rule-strong);
  }

  button {
    position: absolute;
    top: 6px;
    width: 9px;
    height: 17px;
    margin-left: -4.5px;
    padding: 0;
    border: 1px solid var(--ink);
    background: var(--bg);
  }

  button.tline {
    background: var(--bg-sunk);
  }

  button.sel {
    background: var(--accent);
    border-color: var(--accent);
    height: 23px;
    top: 3px;
  }

  button:disabled {
    opacity: 0.3;
  }

  .tick {
    position: absolute;
    top: 26px;
    transform: translateX(-50%);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    white-space: nowrap;
  }

  .teacher-mark {
    position: absolute;
    top: 4px;
    height: 21px;
    border-left: 2px solid var(--c-pillar0);
  }

  .teacher-label {
    position: absolute;
    right: 0;
    top: -12px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--c-pillar0-text);
  }

  .facts {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 0;
    border-top: 1px solid var(--rule);
  }

  .facts div {
    padding-top: 0.35rem;
  }

  dt {
    font-size: var(--text-xs);
    color: var(--muted);
  }

  dd {
    margin: 0;
    font-size: var(--text-sm);
  }
</style>
