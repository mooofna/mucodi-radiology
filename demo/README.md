# mucodi-demo

Offline asset pipeline for the in-browser MuCoDi chest CT demo in `../web`.
It exports the 3D-EfficientNet students to ONNX, prepares the demo scans, builds the
evidence data from the thesis artifacts and writes golden vectors for the web tests.

```bash
uv sync
uv run mucodi-demo --help
uv run pytest
```

Machine-specific paths (Dropbox share, thesis workspace) go in `configs/local.toml`,
which is not tracked; see `configs/local.example.toml`.
