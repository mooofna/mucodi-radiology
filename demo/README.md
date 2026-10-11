# mucodi-demo

Offline asset pipeline for the in-browser MuCoDi chest CT demo in [`../web`](../web). It exports
the 3D-EfficientNet students to ONNX, prepares the demo scans, builds the evidence data from the
thesis artifacts and writes golden vectors for the web tests.

```bash
uv sync
uv run mucodi-demo --help
uv run pytest
```

Machine-specific paths (the Dropbox run tree, the thesis workspace) go in `configs/local.toml`,
which is not tracked; see `configs/local.example.toml`. Everything is written below
`.local/assets/`, which mirrors the site's `assets/` folder.

## Commands

| Command | Writes | Needs |
|---|---|---|
| `models` | ONNX graphs of all fourteen students, stage-tap graphs of TMIN, T2, T6 and B0, and `models.json` with hashes, tap geometry and parity against PyTorch | trained checkpoints |
| `cases` | the three demo scans, retrieved from the NCI Imaging Data Commons | internet access |
| `goldens` | PyTorch embeddings of every demo scan for every rung | models and cases |
| `web-goldens` | nibabel and PyTorch references for the TypeScript preprocessing tests | nothing else |
| `evidence` | the evidence page's data, checked against the thesis tables; `--no-curves` skips the ROC and PR curves | thesis result files; the curves also read the out-of-fold predictions |
| `method-figure` | thesis Figure 4.1, compiled from its TikZ source, with panels from the LIDC-IDRI case | `lualatex`, `pdftocairo`, the thesis sources |
| `bundle` | `web-assets-core.tar.gz`, `web-assets-large.tar` and `SHA256SUMS` in `.local/release/` | the built assets |

Every build step stops on a mismatch: ONNX parity below a cosine of 0.99999, an evidence number
that differs from its thesis table, or a curve whose recomputed AUROC differs from the thesis.

## Running the site locally

```bash
cd ../web
npm ci
npm run assets   # hard-links .local/assets into static/assets
npm run dev
```

## Publishing the assets

The site and CI download the assets from the `demo-assets` release. The core archive holds what
the tests need (the t-line and B0 models, the scans, the evidence and the figures); the large one
holds the B2 to B6 models.

```bash
uv run mucodi-demo bundle
gh release upload demo-assets .local/release/* --clobber
gh workflow run pages.yml
```

## Tests

`uv run pytest` runs offline, without checkpoints or scans: ONNX parity and the gradient check of
the network-view geometry on a seeded, untrained TMIN; the reference preprocessing against its
formulas; the ROC and PR summaries; and the release archives.
