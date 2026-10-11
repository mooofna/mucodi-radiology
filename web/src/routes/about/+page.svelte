<script lang="ts">
  import { onMount } from 'svelte';
  import { loadCases } from '#lib/data/manifest.ts';

  const REPO = 'https://github.com/mooofna/mucodi-radiology';
  const blob = (file: string) => `${REPO}/blob/main/${file}`;

  const BIBTEX = `@thesis{heide2026parameter,
  author      = {Heide, Maurice},
  title       = {On the Parameter-Performance Trade-Off of Radiology Foundation Models},
  type        = {Bachelor's thesis},
  institution = {Technische Universit{\\"a}t Dresden},
  year        = {2026},
  month       = jul,
  url         = {${REPO}}
}`;

  const SOFTWARE = [
    {
      name: 'ONNX Runtime Web',
      use: 'runs the students on your CPU (WebAssembly)',
      licence: 'MIT'
    },
    { name: 'NiiVue', use: 'slice viewer and volume rendering', licence: 'BSD 2-Clause' },
    {
      name: 'dcm2niix',
      use: 'DICOM to NIfTI conversion, compiled to WebAssembly',
      licence: 'BSD 2-Clause'
    },
    { name: 'nifti-reader-js', use: 'NIfTI parsing', licence: 'MIT' },
    { name: 'Observable Plot', use: 'charts', licence: 'ISC' },
    { name: 'Svelte and SvelteKit', use: 'application framework', licence: 'MIT' },
    { name: 'coi-serviceworker', use: 'cross-origin isolation for multithreading', licence: 'MIT' },
    { name: 'IBM Plex Sans and Mono', use: 'typefaces', licence: 'SIL Open Font License 1.1' },
    { name: 'nibabel', use: 'orientation algorithm, ported to TypeScript', licence: 'MIT' },
    {
      name: 'rate-evals',
      use: 'evaluation preprocessing, ported to TypeScript',
      licence: 'Educational Community License 2.0'
    }
  ];

  let lidcAttribution = $state('');
  let copied = $state(false);

  onMount(() => {
    loadCases()
      .then((c) => (lidcAttribution = c.lidc_attribution))
      .catch(() => undefined);
  });

  async function copyCitation() {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } catch {
      copied = false;
    }
  }
</script>

<svelte:head>
  <title>About, MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <header class="intro">
    <p class="label">About</p>
    <h1>A research demo, built from a bachelor thesis.</h1>
    <p class="lede">
      MuCoDi Chest CT runs the distilled students of the thesis <i
        >On the Parameter-Performance Trade-Off of Radiology Foundation Models</i
      > on your own computer, next to the evidence the thesis rests on.
    </p>
  </header>

  <section class="section">
    <div class="section-label">
      <p class="label">Intended use</p>
      <h2>Not a medical device</h2>
    </div>
    <div class="section-body">
      <p>
        This site is a research demonstration. It is not a medical device, it has not been cleared
        or approved by any regulator, and it must not be used for diagnosis, treatment or any other
        clinical decision.
      </p>
      <p>
        The models compute embeddings: vectors that describe a scan. The site turns them into no
        finding, score or probability. The endpoint results on the evidence page are research
        evaluations on public cohorts, not the output of a diagnostic test.
      </p>
      <p>
        By using the models you agree to the use restrictions of the <a
          href={blob('WEIGHTS_LICENSE.md')}>weights licence</a
        >, which among other things forbid using them to provide medical advice or to make clinical
        decisions.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Privacy</p>
      <h2>Your scans stay on your device</h2>
    </div>
    <div class="section-body">
      <ul class="plain">
        <li>
          A scan you open is decoded, converted and encoded inside your browser, in background
          workers. It is never uploaded.
        </li>
        <li>
          No analytics, no cookies and no third-party requests: the fonts, the WebAssembly runtimes
          and the models are all served from this site.
        </li>
        <li>
          Your browser's local storage keeps three settings: the colour theme, the thread setting
          and your benchmark timings, which the race page uses for the device factor. Clearing the
          site data removes them.
        </li>
        <li>
          Choosing all cores registers a small service worker. It enables cross-origin isolation,
          which multithreaded WebAssembly requires, and stores nothing.
        </li>
        <li>
          The site is hosted on GitHub Pages; GitHub processes visitors' IP addresses in its server
          logs under its own privacy statement.
        </li>
      </ul>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Licences</p>
      <h2>Code, weights and data</h2>
    </div>
    <div class="section-body">
      <table class="licences">
        <thead>
          <tr><th scope="col">What</th><th scope="col">Licence</th><th scope="col">Terms</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Code of this site and of the thesis</td>
            <td>CC BY-NC 4.0</td>
            <td><a href={blob('LICENSE')}>LICENSE</a></td>
          </tr>
          <tr>
            <td>Student weights</td>
            <td>Research use only, with the use restrictions of Curia-1's RAIL-M licence</td>
            <td><a href={blob('WEIGHTS_LICENSE.md')}>WEIGHTS_LICENSE.md</a></td>
          </tr>
          <tr>
            <td>LIDC-IDRI scan and the method figure's scan panels</td>
            <td>CC BY 3.0</td>
            <td><a href={blob('DATA_LICENSE.md')}>DATA_LICENSE.md</a></td>
          </tr>
          <tr>
            <td>MIDRC-RICORD-1A scans</td>
            <td>CC BY-NC 4.0</td>
            <td><a href={blob('DATA_LICENSE.md')}>DATA_LICENSE.md</a></td>
          </tr>
        </tbody>
      </table>
      <p class="small">
        The students were distilled from Curia-1 (Raidium; Research-Only RAIL-M licence) and
        Pillar-0 (Yala Lab; Educational Community License 2.0). No teacher weights are distributed.
        CT-RATE, the training corpus, is not redistributed in any form; the method figure uses
        panels re-rendered from the LIDC-IDRI case.
      </p>
      {#if lidcAttribution}
        <p class="small">{lidcAttribution}</p>
      {/if}
      <p class="small">
        Bundled scans: Armato et al., Medical Physics 2011, and the LIDC-IDRI collection
        (doi:10.7937/K9/TCIA.2015.LO9QL9SX); Tsai et al., Radiology 2021, and the MIDRC-RICORD-1A
        collection (doi:10.7937/VTW4-X588); both from The Cancer Imaging Archive through the NCI
        Imaging Data Commons. Full citations and the changes made are listed in DATA_LICENSE.md.
      </p>
      <h3>Software</h3>
      <table class="software">
        <tbody>
          {#each SOFTWARE as s (s.name)}
            <tr><th scope="row">{s.name}</th><td>{s.use}</td><td>{s.licence}</td></tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">People</p>
      <h2>Thesis and credits</h2>
    </div>
    <div class="section-body">
      <p>
        Bachelor's thesis by Maurice Heide, Department of Computer Science, TU Dresden, written in
        the Clinical Artificial Intelligence group at the Else Kröner Fresenius Center for Digital
        Health and submitted on 19 July 2026.
      </p>
      <p>
        Supervised by Prof. Dr. med. Jakob N. Kather, Prof. Dr. Martin Weigert and Prof. Florian J.
        Fintelmann, whose Fintelmann Lab at Massachusetts General Hospital and Harvard Medical
        School hosted the research stay.
      </p>
      <p>
        Computational resources: the Athinoula A. Martinos Center at Massachusetts General Hospital
        and Harvard Medical School, and the Jülich Supercomputing Centre.
      </p>
      <p>
        MuCoDi was first developed for computational pathology in a companion study: Lenz, Heide,
        Gustav, Reitsam and Kather, <i
          >Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models</i
        >, arXiv:2607.05533, 2026. The teachers are described in Agrawal et al.,
        <i>Pillar-0: A New Frontier for Radiology Foundation Models</i>, arXiv:2511.17803, 2025, and
        Dancette et al., <i>Curia: A Multi-Modal Foundation Model for Radiology</i>,
        arXiv:2509.06830, 2025.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Disclosure</p>
      <h2>Advisor and teacher</h2>
    </div>
    <div class="section-body">
      <p>
        Prof. Florian J. Fintelmann, who supervised the thesis day to day, is a co-author of
        Pillar-0, the main teacher. Pillar-0 was chosen by the thesis's cross-cohort benchmark,
        which scores six radiology foundation models under one frozen-feature protocol and checks
        each against its published results before ranking them.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Citation</p>
      <h2>How to cite</h2>
    </div>
    <div class="section-body">
      <pre class="bib"><code>{BIBTEX}</code></pre>
      <div class="cite-row">
        <button onclick={copyCitation}>Copy BibTeX</button>
        <span class="small muted" aria-live="polite">{copied ? 'Copied.' : ''}</span>
        <a class="small" href={REPO}>Source code on GitHub</a>
      </div>
    </div>
  </section>
</div>

<style>
  .intro h1 {
    font-size: var(--text-display);
    margin: 0.3rem 0 0.8rem;
  }

  .lede {
    font-size: var(--text-lg);
    max-width: 46rem;
  }

  .plain {
    margin: 0;
    padding-left: 1.1rem;
    display: grid;
    gap: 0.5rem;
    max-width: var(--measure);
  }

  .licences {
    margin-bottom: var(--space-2);
  }

  h3 {
    font-size: var(--text-base);
    margin: var(--space-3) 0 var(--space-1);
  }

  .software th {
    font-weight: 500;
    color: var(--ink);
    white-space: nowrap;
  }

  .software td:last-child {
    color: var(--muted);
  }

  .bib {
    margin: 0 0 var(--space-2);
    padding: var(--space-2);
    background: var(--bg-sunk);
    border: 1px solid var(--rule);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow-x: auto;
    max-width: 46rem;
  }

  .cite-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .small {
    font-size: var(--text-xs);
  }
</style>
