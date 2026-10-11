<script lang="ts">
  import { onMount } from 'svelte';
  import { asset, resolve } from '$app/paths';
  import { byLabel, loadInference, type Inference } from '#lib/data/evidence.ts';
  import { loadTradeoff, type Tradeoff } from '#lib/data/results.ts';

  let inference = $state<Inference | null>(null);
  let tradeoff = $state<Tradeoff | null>(null);

  onMount(() => {
    loadInference()
      .then((i) => (inference = i))
      .catch(() => undefined);
    loadTradeoff()
      .then((t) => (tradeoff = t))
      .catch(() => undefined);
  });

  const tmin = $derived(inference ? byLabel(inference, 'TMIN') : null);
  const p0 = $derived(inference ? byLabel(inference, 'Pillar-0') : null);
  const mlp = (label: string) => tradeoff?.probes.mlp.find((r) => r.label === label)?.avg_ext;
  const seconds = (ms: number | null | undefined) => (ms ? (ms / 1000).toFixed(2) : '');

  const PAGES = [
    {
      href: resolve('/reading-room'),
      title: 'Reading room',
      text: 'View one of three public chest CTs or your own scan, encode it with any of the fourteen students, and look at the feature maps of each network stage.'
    },
    {
      href: resolve('/race'),
      title: 'CPU race',
      text: 'Two students run side by side on this computer, next to the projected times of the teachers. A benchmark measures every student on its own.'
    },
    {
      href: resolve('/results'),
      title: 'Results',
      text: 'External AUROC by model size, cohort and endpoint, transfer to a hospital outside the training data, and inference speed.'
    },
    {
      href: resolve('/method'),
      title: 'Method',
      text: 'The distillation step by step, the student models, the preprocessing and the limitations of the study.'
    },
    {
      href: resolve('/about'),
      title: 'About',
      text: 'Intended use, privacy, licences, credits and how to cite the thesis.'
    }
  ];
</script>

<svelte:head>
  <title>MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <section class="hero">
    <div class="hero-text">
      <h1>MuCoDi Chest CT</h1>
      <p class="lede">
        Fourteen compact 3D encoders for chest CT, distilled from the radiology foundation models
        <span class="nb">Pillar-0</span> and <span class="nb">Curia-1</span> in a bachelor thesis at TU
        Dresden. This site runs them in the browser, on the CPU of the computer you are using.
      </p>

      <dl class="stats">
        <div>
          <dt>TMIN, 36 k parameters</dt>
          <dd class="num">{seconds(tmin?.cpu_median_ms) || '0.87'}&nbsp;s</dd>
        </div>
        <div>
          <dt><span class="nb">Pillar-0</span>, 79 M parameters</dt>
          <dd class="num">{seconds(p0?.cpu_median_ms) || '71.75'}&nbsp;s</dd>
        </div>
        <div>
          <dt>Speed-up of TMIN</dt>
          <dd class="num">{tmin?.cpu_speedup_vs_pillar0?.toFixed(1) ?? '82.6'}&thinsp;×</dd>
        </div>
      </dl>
      <p class="hint">
        Time to encode one chest CT on one thread of an Intel Xeon Silver 4214R, thesis Table 6.5.
      </p>

      <div class="actions">
        <a class="button primary" href={`${resolve('/reading-room')}?encode=1`}
          >Encode a scan on this computer</a
        >
        <a class="button" href={resolve('/race')}>Start the CPU race</a>
      </div>
      <p class="hint">
        Research demo. Not a medical device and not for diagnosis. Scans you open are processed in
        your browser and never uploaded.
      </p>
    </div>

    <figure class="hero-image">
      <img
        src={asset('images/reading-room.jpg')}
        alt="The reading room: coronal, sagittal and axial slices of a chest CT in the lung window, and a 3D view cut along the coronal plane that shows the ribs and the vessels of the lungs."
        width="1000"
        height="983"
      />
      <figcaption class="hint">
        The reading room with <span class="nb">LIDC-IDRI-0001</span> (CC BY 3.0).
      </figcaption>
    </figure>
  </section>

  {#if tradeoff}
    <section class="section">
      <div class="section-label">
        <p class="label">Accuracy</p>
        <h2>Performance on external cohorts</h2>
      </div>
      <div class="section-body">
        <p>
          With an MLP probe, the student T2, with 0.27 M parameters, reaches an average external
          AUROC of {mlp('T2')?.toFixed(1)}&nbsp;% across ten external cohorts, against
          {mlp('Pillar-0')?.toFixed(1)}&nbsp;% for <span class="nb">Pillar-0</span> with 79 M. How
          much of the teacher's performance a student keeps depends on the probe that reads its
          features; the <a href={resolve('/results')}>results</a> show both probes, every cohort and every
          endpoint.
        </p>
      </div>
    </section>
  {/if}

  <section class="section">
    <div class="section-label">
      <p class="label">This site</p>
      <h2>Pages</h2>
    </div>
    <div class="section-body">
      <ul class="pages">
        {#each PAGES as p (p.href)}
          <li>
            <a href={p.href}>{p.title}</a>
            <p>{p.text}</p>
          </li>
        {/each}
      </ul>
    </div>
  </section>
</div>

<style>
  .hero {
    display: grid;
    grid-template-columns: minmax(0, 6fr) minmax(0, 5fr);
    gap: var(--space-4);
    align-items: start;
  }

  h1 {
    font-size: var(--text-display);
    margin: 0 0 0.8rem;
  }

  .lede {
    font-size: var(--text-lg);
    max-width: 40rem;
  }

  .stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-2);
    margin: var(--space-3) 0 0.6rem;
    border-top: 1px solid var(--rule-strong);
    padding-top: var(--space-2);
  }

  .stats div {
    display: flex;
    flex-direction: column-reverse;
    justify-content: flex-end;
    gap: 0.2rem;
  }

  .stats dt {
    font-size: var(--text-sm);
    color: var(--muted);
  }

  .stats dd {
    margin: 0;
    font-size: var(--text-xl);
    font-weight: 600;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-2);
    margin: var(--space-3) 0 var(--space-2);
  }

  .button {
    display: inline-block;
    border: 1px solid var(--rule-strong);
    padding: 0.55rem 1.1rem;
    color: var(--ink);
    text-decoration: none;
    font-weight: 500;
  }

  .button:hover {
    background: var(--bg-sunk);
  }

  .button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-ink);
  }

  .button.primary:hover {
    filter: brightness(1.08);
  }

  .hero-image {
    margin: 0;
  }

  .hero-image img {
    display: block;
    width: 100%;
    height: auto;
    background: #000000;
  }

  figcaption {
    margin: 0.5rem 0 0;
  }

  .pages {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr));
    gap: var(--space-2) var(--space-3);
  }

  .pages li {
    border-top: 1px solid var(--rule);
    padding-top: 0.6rem;
  }

  .pages a {
    font-weight: 600;
    text-decoration: none;
  }

  .pages a:hover {
    text-decoration: underline;
  }

  .pages p {
    margin: 0.3rem 0 0;
    font-size: var(--text-sm);
    color: var(--muted);
  }

  @media (max-width: 900px) {
    .hero {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
