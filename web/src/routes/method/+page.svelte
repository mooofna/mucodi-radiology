<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { assetUrl } from '#lib/data/manifest.ts';
  import {
    loadInference,
    loadLadder,
    type Inference,
    type LadderRung
  } from '#lib/data/evidence.ts';

  const STEPS = [
    { id: 'full', tab: 'Overview', title: 'The training pipeline' },
    { id: 'offbox', tab: '(a)', title: 'Teacher embeddings, computed once' },
    { id: 'pipebox', tab: '(b)', title: 'Two random crops of each scan' },
    { id: 'm1box', tab: '(c)', title: 'A linear projection for each teacher' },
    { id: 'objbox', tab: '(d)', title: 'The contrastive objective' },
    { id: 'm2box', tab: '(e)', title: 'The losses of both teachers, added' }
  ] as const;

  const ALT =
    'Diagram of the MuCoDi training pipeline. (a) Pillar-0 and Curia-1 embed the training scans once into two feature banks. ' +
    '(b) A stored scan yields two random crops, each converted into four windowed channels and encoded by the student. ' +
    "(c) and (e) A linear head per teacher projects the student's embedding into that teacher's embedding space. " +
    '(d) A contrastive loss compares it with the teacher embeddings. (e) The two losses are added.';

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
    const keys: Record<string, number> = {
      ArrowRight: step + 1,
      ArrowLeft: step - 1,
      Home: 0,
      End: STEPS.length - 1
    };
    if (e.key in keys) {
      e.preventDefault();
      go(keys[e.key]!, true);
    }
  }

  const cpu = (label: string) =>
    inference?.records.find((r) => r.label === label)?.cpu_median_ms ?? null;
  const teachers = $derived(
    (inference?.records ?? []).filter((r) => r.label === 'Pillar-0' || r.label === 'Curia-1')
  );
  const TEACHER_DIM: Record<string, string> = { 'Pillar-0': '1,152', 'Curia-1': '768' };
  const signed = (v: number) => (v < 0 ? `−${-v}` : String(v));
  const fmtNum = (v: number, digits = 0) =>
    v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const fmtSeconds = (ms: number | null) => (ms === null ? '' : (ms / 1000).toFixed(2));
</script>

<svelte:head>
  <title>Method, MuCoDi Chest CT</title>
</svelte:head>

<div class="page">
  <header class="intro">
    <h1>Method</h1>
    <p class="lede">
      MuCoDi, short for multi-teacher contrastive distillation, trains compact 3D convolutional
      encoders to reproduce the embeddings that two frozen radiology foundation models,
      <span class="nb">Pillar-0</span> and <span class="nb">Curia-1</span>, compute for the same
      chest CT. Only the trained student is needed afterwards, and it is the model that runs on this
      site.
    </p>
  </header>

  <section class="tour" aria-labelledby="tour-title">
    <p class="label">Training</p>
    <h2 id="tour-title">Distillation, step by step</h2>
    <div class="tabs" role="tablist" aria-label="Figure panels">
      {#each STEPS as s, i (s.id)}
        <button
          bind:this={tabs[i]}
          role="tab"
          id={`tab-${s.id}`}
          aria-selected={step === i}
          aria-controls="tour-panel"
          tabindex={step === i ? 0 : -1}
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
        <figcaption class="hint">
          Thesis Figure 4.1, compiled from its source. The scan panels are drawn from the bundled
          <span class="nb">LIDC-IDRI</span> case with the thesis's pipeline code, because the
          <span class="nb">CT-RATE</span> scan of the original may not be republished.
        </figcaption>
      </figure>
      <div class="tour-text">
        <p class="label">
          {#if step === 0}Overview{:else}Panel <span class="keep-case">{current.tab}</span>{/if}
        </p>
        <h3>{current.title}</h3>
        {#if step === 0}
          <p>
            <span class="nb">Pillar-0</span> and <span class="nb">Curia-1</span> embed every training
            scan once (a). During training, each scan is cut into two random crops in the four-channel
            form the student reads (b). For each teacher, a linear head projects the student's embedding
            into that teacher's embedding space (c), where a contrastive loss compares it with the teacher's
            embeddings (d). The losses of the two teachers are added (e).
          </p>
          <p class="hint">Use the tabs above, or the arrow keys once a tab has focus.</p>
        {:else if step === 1}
          <ul>
            <li>
              Before training, each teacher embeds all 47,149 training scans of
              <span class="nb">CT-RATE</span> once, with its own preprocessing.
            </li>
            <li>
              The embeddings, normalised to unit length, are kept in GPU memory as a feature bank.
              The teachers are not run during training.
            </li>
            <li>
              Because the teachers are frozen, their embeddings need no gradient. Each student
              embedding is compared with <i>M</i>&nbsp;=&nbsp;16,384 embeddings drawn from the bank.
              Gathering the batches of all four GPUs would have provided only 31.
            </li>
          </ul>
        {:else if step === 2}
          <ul>
            <li>
              Every scan is resampled once to 160 × 320 × 320 voxels and stored in half precision.
            </li>
            <li>
              Two random crops are taken from each stored scan and resized to 128 × 256 × 256
              voxels. Both go to the student. The teachers never see the crops, so cropping is the
              only augmentation.
            </li>
            <li>
              Each crop is converted into four windowed channels: lung, mediastinum, soft tissue and
              bone.
            </li>
          </ul>
        {:else if step === 3}
          <ul>
            <li>
              A linear map projects the student's <i>d</i><sub>S</sub>-dimensional embedding into
              the 1,152-dimensional embedding space of <span class="nb">Pillar-0</span>. Panel (e)
              shows the second map, into the 768 dimensions of <span class="nb">Curia-1</span>.
            </li>
            <li>
              The heads are linear on purpose, so that the student's backbone has to learn the
              representation. They are discarded after training. The linear probe of the evaluation
              was chosen later to match them.
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
              <msup
                ><mi>e</mi><mrow
                  ><msub><mi>s</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><mo>/</mo><mi>τ</mi
                  ></mrow
                ></msup
              >
              <mrow>
                <munderover
                  ><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow
                    ><mi>B</mi><mo>+</mo><mi>M</mi></mrow
                  ></munderover
                >
                <msup
                  ><mi>e</mi><mrow
                    ><msub><mi>s</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub><mo>/</mo><mi>τ</mi
                    ></mrow
                  ></msup
                >
              </mrow>
            </mfrac>
          </math>
          <ul>
            <li>
              Each projected student embedding is scored against <i>B</i>&nbsp;+&nbsp;<i>M</i
              >&nbsp;=&nbsp;16,392 teacher embeddings: its own, the seven others in the batch of its
              GPU (<i>B</i>&nbsp;=&nbsp;8), and 16,384 from the bank.
            </li>
            <li>
              <i>s</i><sub>ik</sub> is the cosine similarity of student embedding <i>i</i> and
              teacher embedding <i>k</i>.
            </li>
            <li>
              The factor 2τ cancels the 1/τ of the scaled similarities, so the temperature,
              τ&nbsp;=&nbsp;0.2, does not act as a hidden learning rate. It only shapes the softmax
              weights.
            </li>
          </ul>
        {:else}
          <ul>
            <li>
              The losses of the two teachers are added without weights, ℒ&nbsp;=&nbsp;ℒ<sub>1</sub
              >&nbsp;+&nbsp;ℒ<sub>2</sub>, each computed on both crops of panel (b).
            </li>
            <li>
              Gradients update the student and its two heads. The teacher embeddings stay fixed,
              which is what allows them to be computed once in panel (a).
            </li>
          </ul>
        {/if}
        <div class="stepper">
          <button onclick={() => go(step - 1)}>Previous</button>
          <span class="hint num" aria-live="polite">{step + 1} of {STEPS.length}</span>
          <button onclick={() => go(step + 1)}>Next</button>
        </div>
        <p class="hint">Thesis Sections 4.2 to 4.6, Figure 4.1 and Table A.1.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Students</p>
      <h2>The fourteen students</h2>
    </div>
    <div class="section-body">
      <p>
        All students are MONAI 3D EfficientNets, scaled by a width and a depth multiplier. The
        t-line extends the family below the smallest standard size, B0. The b-line uses the standard
        sizes B0 to B6 without B1, which keeps the steps in parameter count even. Every student
        reads the same 128 × 256 × 256 input, so the number of parameters is the only variable.
      </p>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" class="num">Parameters (M)</th>
              <th scope="col" class="num">Embedding size</th>
              <th scope="col" class="num">Width</th>
              <th scope="col" class="num">Depth</th>
              <th scope="col" class="num">GFLOPs</th>
              <th scope="col" class="num">CPU time (s)</th>
              <th scope="col" class="num">Training (GPU h)</th>
            </tr>
          </thead>
          {#each [['t', 't-line, below B0'], ['b', 'b-line, standard sizes']] as [line, name] (line)}
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
      <p class="hint">
        Encoder only: the projection heads are discarded after training. GFLOPs count one forward
        pass over the four-channel input, at two operations per multiply-accumulate. CPU time is the
        median single-request latency on one pinned thread of an Intel Xeon Silver 4214R. Thesis
        Tables 4.1, 6.5 and A.3.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Input</p>
      <h2>Preprocessing</h2>
    </div>
    <div class="section-body">
      <ol class="steps">
        <li>
          <strong>Hounsfield units.</strong> The scan is read as a NIfTI volume with its spatial orientation.
          A DICOM folder is first converted with dcm2niix, compiled to WebAssembly.
        </li>
        <li>
          <strong>Orientation.</strong> The voxel axes are permuted and flipped to the closest standard
          orientation, as nibabel does, and ordered as depth, height and width.
        </li>
        <li>
          <strong>Resizing.</strong> The whole field of view is resized to 128 × 256 × 256 voxels with
          trilinear interpolation, computed exactly as in PyTorch.
        </li>
        <li>
          <strong>Windows.</strong> Four windows map their Hounsfield range linearly to 0 to 1 and clamp
          values outside it. They form the four input channels.
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
              <td class="num"
                >{signed(w.centre - w.width / 2)} to {signed(w.centre + w.width / 2)}</td
              >
            </tr>
          {/each}
        </tbody>
      </table>
      <p>
        The browser runs the same steps as the thesis's evaluation code. Every encode of a bundled
        scan in the <a href={resolve('/reading-room')}>reading room</a> is compared with the embedding
        that the thesis pipeline computes in PyTorch, and the test suite requires a cosine similarity
        of at least 0.9999, both from the original scan and from the prepared grid. The orientation step
        is tested against nibabel on all 48 combinations of axis order and direction, and on oblique scans.
      </p>
      <p class="hint">
        Code: the rate-evals harness (<span class="mono">readers.py</span>,
        <span class="mono">preprocess.py</span>, <span class="mono">models/common.py</span>). Thesis
        Table A.1.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Limitations</p>
      <h2>What the study does not cover</h2>
    </div>
    <div class="section-body">
      <dl class="limits">
        <dt>Training data and scope</dt>
        <dd>
          The students are distilled on a single corpus, <span class="nb">CT-RATE</span>, acquired
          at one Turkish hospital. The teachers were pre-trained on far larger corpora, each also
          from a single institution. The evaluation covers chest CT only, mostly without contrast
          agent; contrast-enhanced CT is represented by the RSPECT pulmonary angiography cohort
          alone. The public cohorts include no paediatric chest CT, no occupational lung disease, no
          infection other than COVID-19, and no cohorts from Sub-Saharan Africa, South America or
          the Indian subcontinent.
        </dd>
        <dt>Teachers</dt>
        <dd>
          Every student learns from both teachers. A student distilled from
          <span class="nb">Pillar-0</span> alone was not trained, so the contribution of
          <span class="nb">Curia-1</span> is untested. Each teacher is read with its own preprocessing;
          the effect of a common preprocessing was not studied.
        </dd>
        <dt>Student architecture</dt>
        <dd>
          Only one backbone family is studied. Vision Transformers, MedNeXt and SwinUNETR were not
          trained, so their place on the curve is open. The fixed input size departs from
          EfficientNet's compound scaling, which increases the input resolution together with depth
          and width.
        </dd>
        <dt>Evaluation</dt>
        <dd>
          All results read frozen features through a probe, so they depend on the probe by
          construction. The MLP probe is trained on labelled data of each cohort; the results are
          not zero-shot. End-to-end fine-tuning, for which frozen features give a lower bound, was
          not measured. Each model was trained once, so the intervals do not include variation
          between training runs. The cross-cohort average is a descriptive, unweighted mean without
          correction for multiple comparisons. The OSIC fibrosis endpoint is at chance level for
          every model.
        </dd>
        <dt>Inference measurements</dt>
        <dd>
          Speed was measured on one thread of a server CPU and on one A100 GPU. Multi-threaded CPUs,
          consumer GPUs, edge devices and energy use were not measured. This site adds a third
          setting, the visitor's own CPU in a browser; its teacher times are projected, not
          measured.
        </dd>
      </dl>
      <p class="hint">Thesis Section 7.5. The last sentence describes this site.</p>
    </div>
  </section>

  <section class="section">
    <div class="section-label">
      <p class="label">Conclusions</p>
      <h2>What the thesis concludes</h2>
    </div>
    <div class="section-body">
      <p>
        The thesis finds that the trade-off between parameters and performance depends on the probe.
        With a linear probe, external transfer increases with model size and reaches the linear
        level of <span class="nb">Pillar-0</span> at the largest size, on the point estimate. With an
        MLP probe it levels off early: a compact student comes within about two points of the teacher
        and matches it only at the largest sizes.
      </p>
      <p>
        The reduction in parameters matters most where no accelerator is available. On one CPU core,
        the smallest student embeds a chest CT in under a second, while
        <span class="nb">Pillar-0</span> needs more than a minute. On a GPU the smaller model mainly saves
        memory rather than time.
      </p>
      <p>
        Whether the dependence on the probe is a property of the method or of chest CT remains open.
        The thesis proposes the following next steps:
      </p>
      <ul class="next">
        <li>
          students distilled from a single teacher, starting with <span class="nb">Pillar-0</span>;
        </li>
        <li>other backbone families, such as MedNeXt and 3D Swin;</li>
        <li>end-to-end fine-tuning across all cohorts;</li>
        <li>graded endpoints, such as Lung-RADS categories, instead of binary targets;</li>
        <li>speed and energy measurements on an edge device;</li>
        <li>a larger training corpus from several institutions, and anatomies beyond the chest.</li>
      </ul>
      <p class="hint">Thesis Chapter 8 and Section 7.6.</p>
    </div>
  </section>
</div>

<style>
  .intro h1 {
    font-size: var(--text-display);
    margin: 0 0 0.8rem;
  }

  .lede {
    font-size: var(--text-lg);
    max-width: 48rem;
  }

  .tour {
    border-top: 1px solid var(--rule-strong);
    padding-top: var(--space-2);
    margin-top: var(--space-4);
  }

  .tour h2 {
    margin-bottom: var(--space-2);
  }

  .tabs {
    font-family: var(--font-mono);
    margin-bottom: var(--space-3);
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

  .stepper .hint {
    margin: 0;
  }

  .keep-case {
    text-transform: none;
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

  @media (max-width: 900px) {
    .tour-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
