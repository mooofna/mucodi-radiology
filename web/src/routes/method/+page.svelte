<script lang="ts">
  import { onMount } from 'svelte';
  import { assetUrl } from '#lib/data/manifest.ts';
  import { loadInference, loadLadder, type Inference, type LadderRung } from '#lib/data/evidence.ts';

  const STEPS = [
    { id: 'full', tab: 'Overview', title: 'The whole pipeline on one page' },
    { id: 'offbox', tab: '(a)', title: 'Both teachers run once, before training' },
    { id: 'pipebox', tab: '(b)', title: 'One volume becomes two crops the teachers never see' },
    { id: 'm1box', tab: '(c)', title: 'One projection head per teacher' },
    { id: 'objbox', tab: '(d)', title: 'The objective: pull to the positive, push off the negatives' },
    { id: 'm2box', tab: '(e)', title: 'Two teachers, one sum, no weights to tune' }
  ] as const;

  const ALT =
    'Diagram of the MuCoDi pipeline. (a) Pillar-0 and Curia-1 embed the training corpus once into two feature banks. ' +
    '(b) A cached CT volume yields two random crops, each turned into a four-window stack and encoded by the student. ' +
    "(c) and (e) A linear head per teacher maps the student embedding into that teacher's space. (d) The InfoNCE " +
    'objective pulls each query to its positive key and pushes it off sampled negatives. (e) The two losses are summed.';

  const WINDOWS = [
    { name: 'Lung', centre: -600, width: 1500 },
    { name: 'Mediastinum', centre: 50, width: 400 },
    { name: 'Soft tissue', centre: 50, width: 350 },
    { name: 'Bone', centre: 400, width: 1800 }
  ];

  let step = $state(0);
  let shown = $state(0);
  const current = $derived(STEPS[step]!);
  let tabs: HTMLButtonElement[] = $state([]);
  let ladder = $state<LadderRung[]>([]);
  let inference = $state<Inference | null>(null);

  const figure = (i: number) => assetUrl(`figures/mucodi_${STEPS[i]!.id}.svg`);

  // Swap the figure only once the next one is decoded, so the page never flashes empty.
  $effect(() => {
    const target = step;
    const img = new Image();
    img.src = figure(target);
    img
      .decode()
      .catch(() => undefined)
      .then(() => {
        if (step === target) shown = target;
      });
  });

  onMount(() => {
    Promise.all([loadLadder(), loadInference()])
      .then(([l, i]) => {
        ladder = l;
        inference = i;
      })
      .catch(() => undefined);
    // Warm the cache for the other panels once the page is idle.
    const idle = window.requestIdleCallback ?? ((f: () => void) => setTimeout(f, 1500));
    idle(() => STEPS.forEach((_, i) => (new Image().src = figure(i))));
  });

  function go(i: number, focus = false) {
    step = (i + STEPS.length) % STEPS.length;
    if (focus) tabs[step]?.focus();
  }

  function onTabKey(e: KeyboardEvent) {
    const keys: Record<string, number> = { ArrowRight: step + 1, ArrowLeft: step - 1, Home: 0, End: STEPS.length - 1 };
    if (e.key in keys) {
      e.preventDefault();
      go(keys[e.key]!, true);
    }
  }

  const cpu = (label: string) => inference?.records.find((r) => r.label === label)?.cpu_median_ms ?? null;
  const teachers = $derived((inference?.records ?? []).filter((r) => r.label === 'Pillar-0' || r.label === 'Curia-1'));
  const TEACHER_DIM: Record<string, string> = { 'Pillar-0': '1,152', 'Curia-1': '768' };
  const signed = (v: number) => (v < 0 ? `\u2212${-v}` : String(v));
  const fmtNum = (v: number, digits = 0) =>
    v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const fmtSeconds = (ms: number | null) => (ms === null ? '' : (ms / 1000).toFixed(2));
</script>

<svelte:head>
  <title>Method, MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <header class="intro">
    <p class="label">Method</p>
    <h1>Two teachers, fourteen students, one sum.</h1>
    <p class="lede">
      MuCoDi, multi-teacher contrastive distillation, trains small 3D convolutional encoders to place a chest CT where two
      frozen radiology foundation models place it. The teachers run once, offline. After training only the student's
      backbone remains, and that backbone is what this site runs in your browser.
    </p>
  </header>

  <section class="tour" aria-labelledby="tour-title">
    <p class="label">Distillation</p>
    <h2 id="tour-title">MuCoDi in five steps</h2>
    <div class="tablist" role="tablist" aria-label="Figure panels">
      {#each STEPS as s, i (s.id)}
        <button
          bind:this={tabs[i]}
          role="tab"
          id={`tab-${s.id}`}
          aria-selected={step === i}
          aria-controls="tour-panel"
          tabindex={step === i ? 0 : -1}
          class:on={step === i}
          onclick={() => go(i)}
          onkeydown={onTabKey}>{s.tab}</button
        >
      {/each}
    </div>
    <div class="tour-grid" role="tabpanel" id="tour-panel" aria-labelledby={`tab-${current.id}`}>
      <figure>
        <img
          src={figure(shown)}
          alt={step === 0 ? ALT : `${ALT} Panel ${current.tab} is highlighted.`}
          width="680"
          height="682"
        />
        <figcaption class="muted small">
          Thesis Figure 4.1, compiled from its TikZ source. The scan panels are re-rendered from the bundled LIDC-IDRI case
          with the thesis's own pipeline code, because the CT-RATE volume of the original may not be republished.
        </figcaption>
      </figure>
      <div class="tour-text">
        <p class="label">{#if step === 0}Overview{:else}Panel <span class="keep-case">{current.tab}</span>{/if}</p>
        <h3>{current.title}</h3>
        {#if step === 0}
          <p>
            Two frozen teachers, Pillar-0 and Curia-1, embed the training corpus once (a). Each training volume becomes two
            random crops in the four-window form the student reads (b). One linear head per teacher maps the student's
            embedding into that teacher's space (c, e), a contrastive objective scores it against the teacher's keys (d),
            and the two losses are summed.
          </p>
          <p class="muted small">Step through the panels with the tabs above, or with the arrow keys once a tab has focus.</p>
        {:else if step === 1}
          <ul>
            <li>
              Each teacher embeds all <span class="num">47,149</span> CT-RATE training volumes once, offline, under its own
              preprocessing.
            </li>
            <li>
              The ℓ<sub>2</sub>-normalised embeddings form a feature bank held in GPU memory, so no teacher forward pass
              happens during training.
            </li>
            <li>
              The teachers are frozen, so the stop-gradient is exact. Each query is pushed off
              <i>M</i> = <span class="num">16,384</span> keys sampled from the bank, where gathering the global batch of
              <span class="num">32</span> across the four GPUs would supply <span class="num">31</span>.
            </li>
          </ul>
        {:else if step === 2}
          <ul>
            <li>Every volume is resampled and cached once at <span class="num">160×320×320</span> in half precision.</li>
            <li>
              Two anisotropic random crops of the same volume, both resized to <span class="num">128×256×256</span>, and
              both are the student's. The teachers see neither, which is why the augmentation is crop-only.
            </li>
            <li>
              Each crop becomes the four-window Hounsfield stack the student reads: lung, mediastinum, soft tissue and bone.
            </li>
          </ul>
        {:else if step === 3}
          <ul>
            <li>
              A linear map takes the student's <i>d</i><sub>S</sub>-dimensional embedding into Pillar-0's own
              <span class="num">1,152</span> dimensions, so the comparison happens where the teacher lives. Panel (e) is its
              twin for Curia-1 at <span class="num">768</span>.
            </li>
            <li>
              Linear by design: a bare <i>W</i><sub>t</sub><i>x</i> + <i>b</i><sub>t</sub> leaves the representation work
              to the backbone, and the head is discarded after distillation. The evaluation's linear probe is chosen later to
              match it.
            </li>
          </ul>
        {:else if step === 4}
          <math display="block" class="formula">
            <msub><mi>ℒ</mi><mi>t</mi></msub>
            <mo>=</mo>
            <mo>−</mo>
            <mfrac><mrow><mn>2</mn><mi>τ</mi></mrow><mi>B</mi></mfrac>
            <munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover>
            <mi>log</mi>
            <mfrac>
              <msup><mi>e</mi><mrow><msub><mi>s</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><mo>/</mo><mi>τ</mi></mrow></msup>
              <mrow>
                <munderover
                  ><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>B</mi><mo>+</mo><mi>M</mi></mrow></munderover
                >
                <msup><mi>e</mi><mrow><msub><mi>s</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>/</mo><mi>τ</mi></mrow></msup>
              </mrow>
            </mfrac>
          </math>
          <ul>
            <li>
              The bank, not the batch: each row scores its positive against <i>B</i> + <i>M</i> = <span class="num">16,392</span> keys,
              only <i>B</i> = <span class="num">8</span> of them its own.
            </li>
            <li><i>s</i><sub>ik</sub> scores query <i>i</i> against key <i>k</i>, both ℓ<sub>2</sub>-normalised.</li>
            <li>
              The 2τ prefactor cancels the 1/τ of the logit scaling, so τ is no longer a hidden learning-rate multiplier;
              what remains of it sits in the softmax weights. Here τ = <span class="num">0.2</span>.
            </li>
          </ul>
        {:else}
          <ul>
            <li>
              The two terms combine by a uniform sum, ℒ = ℒ<sub>1</sub> + ℒ<sub>2</sub>, one per teacher, each evaluated
              on both crops of panel (b). No loss weights, no balancer, nothing fitted per teacher.
            </li>
            <li>
              Gradients reach the student and its projectors and stop at the keys, which is what makes the bank of panel (a)
              sound.
            </li>
          </ul>
        {/if}
        <div class="stepper">
          <button onclick={() => go(step - 1)}>Previous</button>
          <span class="num muted small" aria-live="polite">{step + 1} of {STEPS.length}</span>
          <button onclick={() => go(step + 1)}>Next</button>
        </div>
        <p class="muted small">Sources: thesis Sections 4.2 to 4.6, Figure 4.1 and Table A.1.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Students</p>
      <h2>Fourteen rungs, one family</h2>
    </div>
    <div class="section-body">
      <p>
        Every student is a MONAI 3D EfficientNet scaled by a width and a depth multiplier. The t-line extends the family below
        B0; the b-line keeps the canonical sizes, with B1 left out to keep the parameter spacing even. The input grid stays at
        <span class="num">128×256×256</span> for all of them, so parameter count is the only variable.
      </p>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" class="num">Parameters (M)</th>
              <th scope="col" class="num"><i>d</i><sub>S</sub></th>
              <th scope="col" class="num">Width</th>
              <th scope="col" class="num">Depth</th>
              <th scope="col" class="num">GFLOPs</th>
              <th scope="col" class="num">CPU, 1 thread (s)</th>
              <th scope="col" class="num">Training (GPU h)</th>
            </tr>
          </thead>
          {#each [['t', 't-line, below B0'], ['b', 'b-line, canonical sizes']] as [line, name] (line)}
            <tbody>
              <tr class="group"><th colspan="8" scope="colgroup">{name}</th></tr>
              {#each ladder.filter((r) => r.line === line) as r (r.rung)}
                <tr>
                  <th scope="row">{r.label}</th>
                  <td class="num">{r.params_m}</td>
                  <td class="num">{fmtNum(r.d_s)}</td>
                  <td class="num">{r.width}</td>
                  <td class="num">{r.depth}</td>
                  <td class="num">{fmtNum(r.gflops, 2)}</td>
                  <td class="num">{fmtSeconds(cpu(r.label))}</td>
                  <td class="num">{fmtNum(r.train_gpu_hours)}</td>
                </tr>
              {/each}
            </tbody>
          {/each}
          <tbody>
            <tr class="group"><th colspan="8" scope="colgroup">Teachers, frozen</th></tr>
            {#each teachers as t (t.key)}
              <tr>
                <th scope="row">{t.label}</th>
                <td class="num">{t.params_m}</td>
                <td class="num">{TEACHER_DIM[t.label]}</td>
                <td class="num"></td>
                <td class="num"></td>
                <td class="num">{fmtNum(t.flops_g, 2)}</td>
                <td class="num">{fmtSeconds(t.cpu_median_ms)}</td>
                <td class="num"></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <p class="muted small">
        Encoder only: the projection heads are dropped after distillation. FLOPs count one forward pass over the four-window
        volume at two per multiply-accumulate. CPU time is the median single-request latency on one pinned thread of an Intel
        Xeon Silver 4214R. Sources: thesis Tables 4.1, 6.5 and A.3.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Input</p>
      <h2>What the student reads</h2>
    </div>
    <div class="section-body">
      <ol class="steps">
        <li>
          <strong>Hounsfield units.</strong> The scan is read as a NIfTI volume with its affine; a DICOM folder is converted
          first with dcm2niix, compiled to WebAssembly.
        </li>
        <li>
          <strong>Canonical orientation.</strong> The voxel axes are permuted and flipped to the closest canonical orientation,
          as nibabel does, then ordered as depth, height and width.
        </li>
        <li>
          <strong>One grid.</strong> The whole field of view is resized to <span class="num">128×256×256</span> with
          trilinear interpolation, computed as PyTorch computes it.
        </li>
        <li>
          <strong>Four windows.</strong> Each window maps its Hounsfield range linearly onto 0 to 1 and clamps outside it,
          which gives the four input channels.
        </li>
      </ol>
      <table class="windows">
        <thead>
          <tr>
            <th scope="col">Window</th>
            <th scope="col" class="num">Centre (HU)</th>
            <th scope="col" class="num">Width (HU)</th>
            <th scope="col" class="num">Range (HU)</th>
          </tr>
        </thead>
        <tbody>
          {#each WINDOWS as w (w.name)}
            <tr>
              <td>{w.name}</td>
              <td class="num">{signed(w.centre)}</td>
              <td class="num">{w.width}</td>
              <td class="num">{signed(w.centre - w.width / 2)} to {signed(w.centre + w.width / 2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <p>
        The browser port follows the thesis's evaluation code step by step. Every encode of a bundled scan is compared live
        with the embedding the thesis pipeline computes in PyTorch, and the test suite requires a cosine similarity of at
        least <span class="num">0.9999</span> from the raw scan and from the bundled grid alike. The orientation step is
        checked against nibabel on all 48 axis permutations and flips, plus oblique affines.
      </p>
      <p class="muted small">
        Sources: the rate-evals harness (<span class="mono">readers.py</span>, <span class="mono">preprocess.py</span>,
        <span class="mono">models/common.py</span>) and thesis Table A.1.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Limitations</p>
      <h2>What the study does not show</h2>
    </div>
    <div class="section-body">
      <dl class="limits">
        <dt>Training corpus and scope</dt>
        <dd>
          The students are distilled on one corpus, CT-RATE, acquired at a single Turkish centre, while the teachers were
          pre-trained on far larger corpora, themselves single-institution. All evaluation is chest CT, and mostly
          non-contrast: contrast-enhanced CT is probed only by the RSPECT pulmonary angiography cohort. The public cohorts
          contain no paediatric chest CT, no occupational lung disease, no non-COVID infection, and no cohorts from
          Sub-Saharan Africa, South America or the Indian subcontinent.
        </dd>
        <dt>Teacher pool</dt>
        <dd>
          Every student learns from both teachers. A Pillar-0-only student was not trained, so Curia-1's marginal contribution
          is untested. Each teacher is read under its own preprocessing, and the effect of harmonising it was not isolated.
        </dd>
        <dt>Student architecture</dt>
        <dd>
          One backbone family is swept. Vision Transformers, MedNeXt and SwinUNETR were not, so their place on the curve is
          open. The fixed input grid departs from EfficientNet's compound scaling, which grows resolution together with depth
          and width.
        </dd>
        <dt>Evaluation and read-out</dt>
        <dd>
          Every result reads frozen features through a probe, so the transfer is probe-dependent by construction, and the MLP
          probe needs labelled data from each cohort; nothing here is zero-shot. End-to-end fine-tuning, which the frozen
          read-outs bound from below, was not measured. Each model is a single training run, so the intervals omit seed
          variance, and the cross-cohort average is a descriptive, unweighted mean without multiplicity correction. The OSIC
          fibrosis endpoint sits at chance for every model.
        </dd>
        <dt>Inference benchmark</dt>
        <dd>
          Latency was measured at two operating points: one single-threaded Xeon core and one A100 GPU. Multi-threaded CPUs,
          consumer GPUs, true edge devices and energy per inference were not measured. This demo adds a third point, your own
          CPU, under browser conditions; its teacher lanes are projections through a measured device factor, not runs.
        </dd>
      </dl>
      <p class="muted small">Source: thesis Section 7.5. The last sentence describes this demo, not the thesis.</p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Conclusion</p>
      <h2>What the thesis concludes</h2>
    </div>
    <div class="section-body">
      <p>
        The parameter-performance trade-off is probe-dependent. Under the linear probe, external transfer climbs with capacity
        and reaches the Pillar-0 teacher's linear level at the largest size on the point estimate. Under the MLP probe it
        saturates early: a compact student comes within about two points of the teacher and matches it only at the largest
        sizes. The divergence between the two read-outs, not either curve alone, is the finding.
      </p>
      <p>
        The parameter cut pays off most where no accelerator is available. On a single CPU core the smallest student embeds a
        chest-CT volume in under a second, while the Pillar-0 teacher takes over a minute. On a GPU the same cut surfaces as
        reduced memory rather than lower latency, so accuracy and inference cost must be read together.
      </p>
      <p>
        One question stands above the rest: whether the probe-dependent trade-off belongs to the framework or to the chest-CT
        setting in which it was measured. The thesis names the next steps:
      </p>
      <ul class="next">
        <li>single-teacher students, starting with Pillar-0 alone;</li>
        <li>other backbone families, such as MedNeXt and 3D Swin;</li>
        <li>end-to-end fine-tuning across the cohort matrix;</li>
        <li>graded endpoints, such as Lung-RADS categories, in place of binary targets;</li>
        <li>a benchmark with energy measurement on a true edge device;</li>
        <li>a larger, multi-institution training corpus, and the sweep beyond chest CT.</li>
      </ul>
      <p class="muted small">Sources: thesis Chapter 8 and Section 7.6.</p>
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

  .tour {
    border-top: 1px solid var(--rule-strong);
    padding-top: var(--space-2);
    margin-top: var(--space-4);
  }

  .tour h2 {
    margin-bottom: var(--space-2);
  }

  .tablist {
    display: flex;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--rule);
    margin-bottom: var(--space-3);
  }

  .tablist button {
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 0.45rem 0.9rem;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--muted);
    margin-bottom: -1px;
  }

  .tablist button.on {
    color: var(--ink);
    border-bottom-color: var(--accent);
  }

  .tour-grid {
    display: grid;
    grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
    gap: var(--space-4);
    align-items: start;
  }

  figure {
    margin: 0;
  }

  figure img {
    display: block;
    width: 100%;
    height: auto;
    background: #ffffff;
    border: 1px solid var(--rule);
  }

  figcaption {
    margin-top: 0.5rem;
    max-width: none;
  }

  .tour-text h3 {
    font-size: var(--text-xl);
    margin: 0.2rem 0 var(--space-2);
  }

  .tour-text ul {
    padding-left: 1.1rem;
    margin: 0 0 var(--space-2);
    display: grid;
    gap: 0.6rem;
  }

  .formula {
    font-size: 1.15rem;
    margin: 0 0 var(--space-2);
    overflow-x: auto;
  }

  .stepper {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: var(--space-3) 0 var(--space-2);
  }

  .scroll {
    overflow-x: auto;
    margin-bottom: var(--space-1);
  }

  tr.group th {
    padding-top: 0.9rem;
    color: var(--ink);
    font-weight: 600;
    border-bottom: 1px solid var(--rule-strong);
  }

  tbody th[scope='row'] {
    font-weight: 500;
    color: var(--ink);
  }

  .steps {
    padding-left: 1.2rem;
    margin: 0 0 var(--space-3);
    display: grid;
    gap: 0.5rem;
    max-width: var(--measure);
  }

  .windows {
    max-width: 32rem;
    margin-bottom: var(--space-3);
  }

  .limits {
    margin: 0 0 var(--space-2);
    max-width: var(--measure);
  }

  .limits dt {
    font-weight: 600;
    margin-top: var(--space-2);
  }

  .limits dt:first-child {
    margin-top: 0;
  }

  .limits dd {
    margin: 0.25rem 0 0;
  }

  .next {
    padding-left: 1.1rem;
    margin: 0 0 var(--space-2);
    max-width: var(--measure);
  }

  .small {
    font-size: var(--text-xs);
  }

  .keep-case {
    text-transform: none;
  }

  @media (max-width: 900px) {
    .tour-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
