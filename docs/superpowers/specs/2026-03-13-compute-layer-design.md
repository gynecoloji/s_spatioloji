# Compute Layer Design — s_spatioloji

**Date:** 2026-03-13
**Status:** Approved

---

## Overview

The compute layer adds single-cell analysis operations on top of the data layer.
It follows a functional design: flat functions that accept an `s_spatioloji` object,
perform a computation, write results to the `maps/` store, and return the result key.
A thin `Maps` accessor on `s_spatioloji` provides clean result read-back.

---

## File Layout

```
src/s_spatioloji/
├── compute/
│   ├── __init__.py
│   ├── normalize.py         # normalize_total, log1p, scale, pearson_residuals
│   ├── feature_selection.py # highly_variable_genes
│   ├── reduction.py         # pca, umap, tsne, diffmap
│   ├── clustering.py        # leiden, louvain, kmeans, hierarchical
│   ├── batch_correction.py  # harmony, combat, regress_out, scvi_batch
│   ├── imputation.py        # magic, alra, knn_smooth, scvi_impute
│   ├── _scvi.py             # private: shared scVI train helper
│   └── _runner.py           # private: subprocess entrypoint for conda bridge
```

---

## Result Store: `maps/`

Results are written to a `maps/` directory under the dataset root.
Two formats are used depending on result size:

| Result type | Size | Format | Example path |
|---|---|---|---|
| Embeddings, labels, HVG-subset corrected matrices | n_cells × 2–n_hvg | Parquet | `maps/X_pca.parquet` |
| Full expression matrices | n_cells × n_genes | Zarr | `expression_scvi.zarr/` |

All Parquet files include `cell_id` as the first column for joining back to `cells.parquet`.

### Overwrite policy

All compute functions **always overwrite** existing output files by default.
A `force=False` parameter is available on every function: when `force=False` and **all**
expected primary output files already exist, the function returns the primary output key
immediately without recomputing. For functions with multiple outputs (e.g., `pca` writes
both `X_pca.parquet` and `X_pca_loadings.parquet`), `force=False` only skips recomputation
if **both** files are present. Default is `force=True` (always recompute).

For `scvi_batch` and `scvi_impute`, `force=False` skips recomputation if
`maps/X_scvi_latent.parquet` exists. When the primary skip fires, any secondary outputs
requested by `save_expression=True` (e.g. `expression_scvi.zarr/`) are still written if
they are missing — the skip only prevents re-running the model, not writing already-requested
side outputs. The model cache in
`maps/_scvi_model/` has its own fingerprint-based invalidation (see `_scvi_train`) and is
independent of the caller's `force` parameter. `force` applies only to the immediate call
— shared model reuse between `scvi_impute` and `scvi_batch` is governed by the fingerprint,
not by either caller's `force` value.

### Write atomicity

All writes use **write-to-temp-then-rename** to avoid partial files:
Parquet is written to `maps/.<key>.tmp.parquet`, then renamed to `maps/<key>.parquet`.
Zarr stores are written to `.<key>.tmp.zarr/`, then renamed. Concurrent writes
from multiple threads or processes are **not supported** in the default local scheduler.

### Canonical key names

```
maps/X_norm.parquet           # normalize_total output
maps/X_log1p.parquet          # log1p output
maps/X_scaled.parquet         # scale output (zero-centred, HVG subset, n_cells × n_hvg)
maps/X_residuals.parquet      # pearson_residuals output
maps/hvg.parquet              # HVG table (columns: gene, highly_variable, mean, variance, dispersion)
maps/X_pca.parquet            # PCA embedding (n_cells × n_components)
maps/X_pca_loadings.parquet   # PCA gene loadings (n_genes × n_components)
maps/X_umap.parquet           # UMAP embedding (n_cells × 2)
maps/X_tsne.parquet           # tSNE embedding (n_cells × 2)
maps/X_diffmap.parquet        # diffusion map (n_cells × n_components)
maps/leiden.parquet           # Leiden cluster labels (columns: cell_id, leiden)
maps/louvain.parquet          # Louvain cluster labels (columns: cell_id, louvain)
maps/kmeans.parquet           # KMeans cluster labels (columns: cell_id, kmeans)
maps/hierarchical.parquet     # Hierarchical cluster labels (columns: cell_id, hierarchical)
maps/X_pca_harmony.parquet    # Harmony-corrected PCA (n_cells × n_components)
maps/X_combat.parquet         # ComBat-corrected expression HVG subset (n_cells × n_hvg)
maps/X_regressed.parquet      # regress_out result HVG subset (n_cells × n_hvg)
maps/X_scvi_latent.parquet    # scVI latent embedding (n_cells × n_latent; shared by impute + batch)
maps/_scvi_model/             # cached scVI model directory
maps/_scvi_model/params.json  # fingerprint: input_key, n_latent, batch_key, n_epochs

maps/X_magic.parquet          # MAGIC denoised expression HVG subset (n_cells × n_hvg)
maps/X_alra.parquet           # ALRA imputed expression HVG subset (n_cells × n_hvg)
maps/X_knnsmooth.parquet      # KNN-smoothed expression HVG subset (n_cells × n_hvg)
```

The following Zarr stores are written at the **dataset root** (not under `maps/`):

```
expression_scvi.zarr/         # scVI denoised full expression (if save_expression=True)
expression_combat.zarr/       # ComBat batch-corrected full expression (if save_expression=True)
expression_magic.zarr/        # MAGIC imputed full expression (if save_expression=True)
expression_alra.zarr/         # ALRA imputed full expression (if save_expression=True)
```

**Note on HVG-subset Parquet vs full-expression Zarr:**
The following functions produce corrected/denoised expression, compute over all genes,
and write two outputs: `combat`, `magic`, `alra`.
- `maps/X_<name>.parquet` — result subset to HVGs (n_cells × n_hvg; suitable as PCA input)
- `expression_<name>.zarr/` — full n_cells × n_genes result (if `save_expression=True`)

The `hvg_key` parameter on these functions controls which HVG table is used for subsetting.
For conda-bridge functions (`magic`, `alra`), the HVG gene list is serialised into the input
temp file before the subprocess is launched — the runner subprocess does not read the
`maps/` store directly.

`regress_out` and `knn_smooth` write **Parquet only** (n_cells × n_hvg via `hvg_key`) —
no full-expression Zarr is written and neither function has a `save_expression` parameter.

`scvi_batch` and `scvi_impute` write a **latent embedding** (not expression) to Parquet
(`n_cells × n_latent`). Latent embeddings are not expression matrices and do not require
HVG subsetting. These functions have no `hvg_key` parameter.

---

## `Maps` Accessor

Added to `s_spatioloji` in `data/core.py`:

```python
sj.maps["X_pca"]           # → dask.dataframe (lazy) — bare key name, no extension
sj.maps["expression_scvi"] # → ExpressionStore — bare key, resolved to .zarr on disk
sj.maps.keys()             # → list[str] of bare key names (no extensions)
sj.maps.has("X_umap")      # → bool
sj.maps.delete("X_pca")    # → removes maps/X_pca.parquet or <root>/<key>.zarr/ from disk
```

### Key resolution rules

- Keys are always **bare names** (no `.parquet` or `.zarr` suffix).
- On lookup, the accessor checks in order:
  1. `maps/<key>.parquet` → returns `dask.dataframe`
  2. `maps/<key>.zarr/` → returns `ExpressionStore` (opened with `sj._config.chunks`)
  3. `<root>/<key>.zarr/` → returns `ExpressionStore` (opened with `sj._config.chunks`)
- If none exist, `KeyError` is raised.
- The following keys resolve via rule 3 (bare-root Zarr):
  `expression_scvi`, `expression_combat`, `expression_magic`, `expression_alra`
- `_scvi_model` is **not** a user-accessible key; it is an internal directory and will
  raise `KeyError` if looked up via `sj.maps`.
- The `maps/` directory is created on first write by any compute function (not by the
  accessor itself). Each compute function calls `maps_dir.mkdir(exist_ok=True)` before
  writing its temp file.

### `delete` behaviour

`delete` follows the same lookup order as `__getitem__` and removes **only the first match**:
1. If `maps/<key>.parquet` exists → removes it.
2. Else if `maps/<key>.zarr/` exists → removes it recursively.
3. Else if `<root>/<key>.zarr/` exists → removes it recursively.
4. If none exist → raises `KeyError`.

If both `maps/<key>.parquet` and a root-level Zarr for the same key somehow coexist,
only the Parquet is removed (first match wins). No current code path produces this state.

### `keys()` scan scope

`sj.maps.keys()` returns all discoverable bare key names by scanning:
1. `maps/*.parquet` → bare name without extension
2. `maps/*.zarr/` → bare name without extension
3. `<root>/expression_*.zarr/` → bare name without extension

All three lists are merged and deduplicated. The result includes `expression_scvi`,
`expression_combat`, `expression_magic`, and `expression_alra` when those directories exist.

---

## Standard Pipeline Order

```
from_xenium() / from_merscope()
    ↓
normalize_total(sj)                  → maps/X_norm.parquet
log1p(sj)                            → maps/X_log1p.parquet
    ↓
highly_variable_genes(sj)            → maps/hvg.parquet  (returns key "hvg")
    ↓
scale(sj, hvg=True, hvg_key="hvg")   → maps/X_scaled.parquet  ← zero-centred, HVG subset only
    ↓
pca(sj, hvg=True, hvg_key="hvg")     → maps/X_pca.parquet
                                        maps/X_pca_loadings.parquet (side effect)
    ↓
umap(sj)                             → maps/X_umap.parquet
tsne(sj)                             → maps/X_tsne.parquet     (optional)
    ↓
leiden(sj)                           → maps/leiden.parquet
  (or: louvain / kmeans / hierarchical — alternative clustering methods)

── optional ──────────────────────────────────────────────────────────────────
harmony(sj, batch_key)               → maps/X_pca_harmony.parquet
scvi_impute(sj)                      → maps/X_scvi_latent.parquet
                                        expression_scvi.zarr/  (if save_expression=True)
scvi_batch(sj, batch_key)            → maps/X_scvi_latent.parquet  (shared model)
                                        expression_scvi.zarr/  (if save_expression=True)
```

---

## Function Signatures

### normalize.py

```python
def normalize_total(sj, target_sum=1e4, input_key="expression",
                    output_key="X_norm", force=True) -> str
def log1p(sj, input_key="X_norm", output_key="X_log1p", force=True) -> str
def scale(sj, input_key="X_log1p", hvg=True, hvg_key="hvg",
          max_value=10.0, output_key="X_scaled", force=True) -> str
def pearson_residuals(sj, theta=100.0, input_key="expression",
                      output_key="X_residuals", force=True) -> str
```

**Note on `scale`:** When `hvg=True`, genes are subset using the HVG list from
`maps/<hvg_key>.parquet` **before** any densification. This prevents OOM on large
gene panels. The output Parquet contains only HVG columns.

### feature_selection.py

```python
def highly_variable_genes(sj, n_top=2000, input_key="X_log1p",
                           output_key="hvg", force=True) -> str
```

Returns the output key (`"hvg"`) consistent with all other functions.
The HVG table at `maps/hvg.parquet` has columns: `gene`, `highly_variable` (bool),
`mean`, `variance`, `dispersion`. Retrieve the gene list via:
`sj.maps["hvg"].compute().query("highly_variable")["gene"].tolist()`

### reduction.py

```python
def pca(sj, n_components=50, n_subsample=100_000, hvg=True, hvg_key="hvg",
        input_key="X_scaled", output_key="X_pca",
        output_loadings_key="X_pca_loadings", force=True) -> str
def umap(sj, n_neighbors=15, input_key="X_pca", output_key="X_umap", force=True) -> str
def tsne(sj, perplexity=30, input_key="X_pca", output_key="X_tsne", force=True) -> str
def diffmap(sj, n_components=15, input_key="X_pca",
            output_key="X_diffmap", force=True) -> str
```

**`pca` return value:** Returns `output_key` (the primary embedding key, e.g. `"X_pca"`).
Gene loadings are written as a side effect to `maps/<output_loadings_key>.parquet`.
`force=False` skips recomputation only when **both** `maps/<output_key>.parquet` and
`maps/<output_loadings_key>.parquet` already exist.

**`pca` parameter clamping:** `n_components` is automatically clamped to
`min(n_components, n_cells - 1, n_features)` where `n_features` is the number of
columns in `input_key` (when `hvg=True`, this equals n_hvg; when `hvg=False`, this equals
n_genes). Tests must assert the clamped value is used without error.

**`diffmap` dependency:** Implemented using `scipy.sparse.linalg.eigs` on the
diffusion operator — no extra install required beyond `scipy` (already in core deps).
No additional optional group entry needed.

### clustering.py

```python
def leiden(sj, resolution=1.0, n_neighbors=15, input_key="X_pca",
           output_key="leiden", force=True) -> str
def louvain(sj, resolution=1.0, n_neighbors=15, input_key="X_pca",
            output_key="louvain", force=True) -> str
def kmeans(sj, n_clusters=10, input_key="X_pca",
           output_key="kmeans", force=True) -> str
def hierarchical(sj, n_clusters=10, input_key="X_pca",
                 output_key="hierarchical", force=True) -> str
```

**`n_neighbors` clamping for graph-based methods:** `leiden` and `louvain` build a KNN
graph internally. The `n_neighbors` count is clamped to `min(n_neighbors, n_cells - 1)`.
Tests must assert this with small fixtures. `kmeans` and `hierarchical` do not build
KNN graphs and have no `n_neighbors` parameter.

### batch_correction.py `[pip install s_spatioloji[batch]]`

```python
def harmony(sj, batch_key="fov_id", input_key="X_pca",
            output_key="X_pca_harmony", force=True) -> str
def combat(sj, batch_key="fov_id", input_key="X_log1p", hvg_key="hvg",
           output_key="X_combat", save_expression=True, force=True) -> str
def regress_out(sj, keys=["transcript_counts"], input_key="X_log1p", hvg_key="hvg",
                output_key="X_regressed", force=True) -> str
def scvi_batch(sj, batch_key="fov_id", input_key="expression", n_latent=30, n_epochs=400,
               output_key="X_scvi_latent", save_expression=True,
               conda_env=None, timeout=7200, force=True) -> str
```

`combat` runs **in-process** (no conda bridge). It reads `maps/<hvg_key>.parquet` to
identify HVGs, operates on the full input matrix, and writes:
- `maps/X_combat.parquet` — corrected expression subset to HVGs (n_cells × n_hvg)
- `expression_combat.zarr/` — full corrected matrix (if `save_expression=True`)

`regress_out` similarly uses `hvg_key` to subset its Parquet output.

`scvi_batch` uses `input_key="expression"` (raw counts required by scVI). It always writes
`maps/X_scvi_latent.parquet`. When `save_expression=True` (default), also writes
`expression_scvi.zarr/`.

### imputation.py `[pip install s_spatioloji[imputation]]`

```python
def magic(sj, knn=5, input_key="X_log1p", hvg_key="hvg", output_key="X_magic",
          save_expression=True, conda_env=None, timeout=7200, force=True) -> str
def alra(sj, input_key="X_log1p", hvg_key="hvg", output_key="X_alra",
         save_expression=True, conda_env=None, timeout=7200, force=True) -> str
def knn_smooth(sj, k=15, input_key="X_log1p", hvg_key="hvg",
               output_key="X_knnsmooth", force=True) -> str
def scvi_impute(sj, batch_key=None, input_key="expression", n_latent=30, n_epochs=400,
                output_key="X_scvi_latent",
                save_expression=True, conda_env=None, timeout=7200, force=True) -> str
```

`magic` and `alra` operate on the full expression matrix and write two outputs:
- `maps/X_<name>.parquet` — result subset to HVGs using `hvg_key` (n_cells × n_hvg)
- `expression_<name>.zarr/` — full result matrix (if `save_expression=True`)

`knn_smooth` writes Parquet only (n_cells × n_hvg subset via `hvg_key`).
It uses only `scikit-learn` (core dependency) — no conda bridge, no `save_expression`.

`scvi_impute` and `scvi_batch` use `input_key="expression"` (raw counts required by scVI).
`scvi_impute` always writes `maps/X_scvi_latent.parquet` (latent embedding).

`batch_key=None` on `scvi_impute` means unsupervised scVI (pure imputation, no batch
conditioning). When `batch_key` is provided, the model is batch-conditioned.
`scvi_impute(batch_key="fov_id")` and `scvi_batch(batch_key="fov_id")` with matching
`n_latent` / `n_epochs` will share the cached model in `maps/_scvi_model/`.

---

## scVI Shared Helper (`_scvi.py`)

```python
def _scvi_train(sj, input_key, batch_key, n_latent, n_epochs, conda_env, timeout, force) -> Path
```

Returns path to `maps/_scvi_model/`. When `force=False`, skips training if
`maps/_scvi_model/params.json` exists **and** its fingerprint matches all four of:
`input_key`, `n_latent`, `batch_key`, `n_epochs` — the same fields stored in `params.json`.
When `force=True`, deletes the existing model directory and retrains regardless of cache.
**This means `force=True` destroys the shared model cache.** A subsequent
`scvi_batch(force=False)` or `scvi_impute(force=False)` with matching parameters will
find no cache and retrain. This is by design: each caller's `force` flag is independent.

`batch_key=None` trains an unsupervised scVI model (no batch conditioning).
`None` is serialised as the literal string `"null"` in `params.json`, ensuring
that unsupervised and batch-conditioned models never share a cached model even
if `n_latent` and `n_epochs` are identical.

If parameters differ from a cached model (but `force=False`), the old model
directory is deleted and retraining occurs.

**Obtaining the denoised expression matrix:** After `_scvi_train` returns, the caller
uses the `scvi-tools` Python API to load the trained model from the returned path and
call `model.get_normalized_expression()` to obtain the denoised full expression matrix.
The caller then writes this matrix to `expression_scvi.zarr/` via `ExpressionStore.create()`
(always available as a core data-layer class; no optional install needed for Zarr writes).

**Full-expression Zarr writes for `combat`, `magic`, `alra`:** After computing the full
corrected/denoised matrix, the caller calls `ExpressionStore.create(path, n_cells, n_genes,
chunk_config=sj._config.chunks)` and writes the matrix via `store.write_chunk()`.
`ExpressionStore` is a core class and is always available.

**scVI outputs three artefacts — both `scvi_batch` and `scvi_impute` write all three:**

| Artefact | Path | Written by |
|---|---|---|
| Latent embedding | `maps/X_scvi_latent.parquet` | always |
| Denoised expression | `expression_scvi.zarr/` | if `save_expression=True` |
| Model weights | `maps/_scvi_model/` | always |

If both `scvi_impute` and `scvi_batch` are called on the same `sj` with matching
parameters, the second call reuses the cached model and only writes any missing artefacts.

---

## Conda Bridge (`_runner.py`)

When `conda_env` is provided to any compute function that supports it
(`magic`, `alra`, `scvi_batch`, `scvi_impute`):

1. **Validate env exists:** Run `conda env list` and raise `ValueError` if env not found
   (fails fast before any data serialisation).
2. **Serialise inputs:** Write expression subset + kwargs to temp files
   (Parquet for tabular data, NPZ for dense arrays). Kwargs written to
   `<tmpdir>/kwargs.json` (not passed on command line — avoids shell injection).
3. **Spawn subprocess:**
   ```
   conda run -n {conda_env} python -m s_spatioloji.compute._runner \
       --fn {fn_name} \
       --input {tmp_input_path} \
       --output {tmp_output_path} \
       --kwargs-file {tmp_kwargs_json}
   ```
   Timeout controlled by the `timeout` parameter (default 7200 seconds).
4. **On non-zero exit:** Raise `RuntimeError` with subprocess stderr captured. For `alra`,
   non-zero exits include R/rpy2 failures (R not on PATH, ALRA package not installed) —
   the runner does not distinguish these; all failures surface as `RuntimeError`.
5. **Read output:** Load temp output file, write to `maps/` via normal store.
6. **Cleanup:** Temp files removed in a `finally` block.

**Minimum conda env requirements per function** (must be installed by user):

| Function | Conda bridge supported | Required packages in conda env |
|---|---|---|
| `scvi_impute` / `scvi_batch` | yes | `scvi-tools>=1.0`, `pytorch`, `anndata` |
| `magic` | yes | `magic-impute`, `scikit-learn` |
| `alra` | yes | `rpy2`, `R` (ALRA is an R package called via rpy2) |
| `combat` | no (in-process only) | — |
| `knn_smooth` | no (core deps only) | — |

`_runner.py` has **no dependency on the main `s_spatioloji` stack** — only
`numpy`, `pandas`, and the target tool are imported inside the runner.

---

## Optional Install Groups

```toml
[project.optional-dependencies]
reduction  = ["umap-learn", "openTSNE"]
clustering = ["leidenalg", "igraph", "python-louvain"]
batch      = ["harmonypy", "pycombat>=0.3"]
imputation = ["scvi-tools>=1.0", "magic-impute", "rpy2"]
all        = ["s_spatioloji[reduction,clustering,batch,imputation]"]
```

**Notes:**
- `python-louvain` (PyPI) provides the `community` module required by `louvain()`.
- `pycombat` (PyPI) is the correct package name for ComBat batch correction.
- `alra` is called via `rpy2` + R — users must install R + ALRA separately.
- `diffmap` uses `scipy.sparse.linalg.eigs` — no extra dependency.
- `knn_smooth` uses `scikit-learn` — no extra dependency.

All optional imports are guarded at call time with descriptive install messages:
```python
try:
    import harmonypy
except ImportError:
    raise ImportError("Install with: pip install s_spatioloji[batch]")
```

---

## Testing Strategy

- **Fixture size:** 200 cells × 30 genes minimum (avoids KNN/PCA degenerate cases).
- **Parameter clamping:** Each function that auto-clamps parameters (PCA `n_components`,
  KNN `n_neighbors` in leiden/louvain) has a dedicated test asserting the clamped run
  completes without error.
- **Output verification:** Each test asserts: output key written to `maps/`, shape correct,
  `cell_id` column present, dtype appropriate.
- **force=False multi-output:** For `pca`, verify that `force=False` skips recomputation
  only when both `X_pca.parquet` and `X_pca_loadings.parquet` exist.
- **force=False single-output:** Tests verify that calling a function twice with `force=False`
  does not recompute (mock the compute step and assert it is not called on second invocation).
- **force=False scVI:** Verify `force=False` skips when `X_scvi_latent.parquet` exists;
  model cache logic is tested separately via `_scvi_train` unit tests.
- **Conda bridge:** Mocked via `unittest.mock.patch("subprocess.run")` — no real conda
  env required in CI. Tests verify: env validation called, kwargs written to file (not CLI),
  temp files cleaned up on success and on error.
- **Optional deps:** `pytest.importorskip("harmonypy")` etc. — skipped in CI if not installed.
- **scVI cache:** Test that calling `scvi_impute` twice with same params skips retraining
  (assert `_scvi_train` called once). Test that changed `n_latent` triggers retraining.
  Test that `batch_key=None` and `batch_key="fov_id"` produce different fingerprints.
- **Maps.delete Zarr:** Verify `sj.maps.delete("expression_scvi")` removes `<root>/expression_scvi.zarr/`.
- **Tests:** `tests/unit/test_compute_normalize.py`, `test_compute_feature_selection.py`,
  `test_compute_reduction.py`, `test_compute_clustering.py`, `test_compute_batch.py`,
  `test_compute_imputation.py`, `test_compute_runner.py`

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Result format | Parquet for small (HVG-subset), Zarr for full matrices | Columnar Parquet for embeddings/HVG matrices; chunked Zarr for full n_cells × n_genes |
| PCA strategy | Subsample exact PCA + project all cells | Fast, standard in practice (Scanpy/Seurat pattern) |
| `n_components` clamping | Auto-clamp at runtime | Prevents silent failure on small datasets |
| HVG mechanism | Explicit `hvg_key` param on all expression functions | Avoids implicit magic; consistent with input_key pattern |
| `hvg_key` on imputation/batch | Subsetting happens at write time | Full denoised matrix computed first; HVG subset written to Parquet for PCA input |
| scVI `input_key="expression"` | Raw counts required by scVI | scVI's ZINB model requires integer counts; log-normalised input would be incorrect |
| scVI outputs | Latent + full expression + model | Full utility; `save_expression` flag controls expression write |
| scVI model cache | `params.json` fingerprint | Avoids stale reuse; deterministic invalidation |
| scVI `batch_key=None` sentinel | Serialised as `"null"` in JSON | Prevents unsupervised/batch models from sharing cache |
| `combat` in-process only | No conda bridge for pycombat | pycombat is pip-installable; conda bridge overhead not justified |
| Conda bridge | Subprocess + temp files + kwargs file | Clean isolation; avoids shell injection |
| Conda env validation | Eager (before serialisation) | Fails fast; no wasted I/O |
| `knn_smooth` no conda bridge | Uses sklearn (core dep) only | No external tool needed; conda overhead unnecessary |
| `timeout` per-function param | Default 7200 s | Long-running tools (scVI) may exceed shorter timeouts |
| `force` in `_scvi_train` | Passed explicitly from caller | Keeps cache-delete logic inside helper; callers don't manage model dir directly |
| `force=False` scVI skip | Check `X_scvi_latent.parquet` only | Model cache has its own invalidation; separating concerns avoids double-checking |
| Overwrite policy | `force=True` default | Always fresh; `force=False` for idempotent pipelines |
| Write atomicity | Temp-then-rename | Prevents partial file reads if process interrupted |
| Concurrent writes | Not supported | Stated explicitly; avoids false safety assumptions |
| `diffmap` impl | `scipy.sparse.linalg.eigs` | No extra dependency; scipy already in core |
| `louvain` dep | `python-louvain` (PyPI) | Correct package name; in `clustering` group |
| `combat` dep | `pycombat` (PyPI) | Correct package name |
| `alra` dep | `rpy2` + R | ALRA is R-native; conda bridge recommended |
| Maps.delete Zarr | Deletes `<root>/<key>.zarr/` recursively | Consistent with key resolution rules for bare-root Zarr stores |
