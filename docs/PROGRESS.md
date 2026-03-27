# Dataloader++ Progress Tracker

> **Copilot**: Read this file at the start of every session. Update it after
> completing each implementation step, decision, or design change.

---

## Current Phase: Phase 2 — Adapters & Integration Prep

### Phase Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Interface Design | ✅ Complete | ABCs, types, manifest joiner, transforms, collator, dataset |
| Phase 2 — Adapters & Refactors | 🔨 In Progress | Wrap pipeline stages, extract loaders, prepare for upstream |
| Phase 3 — Concrete Feature Loader | ⬜ Not Started | WebDataset loader, raw file loader, sampler |
| Phase 4 — Validation | ⬜ Not Started | End-to-end tests, benchmarks |

---

## Decisions Log

### D1: Waveform Processing — Dual-Mode Class (2026-03-26)

**Decision**: Build a single `WaveformProcessor` class that supports both:
- **Online mode**: Modifies waveforms at dataload time (inside the `DataProcessor`
  transform pipeline). Lightweight, no disk writes.
- **Offline mode**: During feature extraction, saves the modified waveform to
  disk alongside provenance metadata (what transform was applied, parameters,
  source file). Future loads skip the transform and read the pre-processed file.

**Rationale**: The user wants flexibility — apply transforms on-the-fly for
experimentation, or bake them in for production. One class, two code paths.

**Status**: ✅ Implemented (`dataloader/transform/waveform.py`).

### D2: Masking — Deferred (2026-03-26)

**Decision**: Park the question of where masking lives (transform vs. collator)
until we have more information on what masking is needed for and how it would be
implemented in the upstream codebase.

**Status**: ⏸ Parked.

### D3: metasr-internal / fs2 Interface Compatibility (2026-03-26)

**Decision**: We do not currently have access to the upstream signatures. Build a
thin **compatibility shim layer** with clearly documented interface boundaries.
When upstream signatures become available, we adapt the shim without rewriting core
logic.

**Approach**:
- Define our own `DataBatch` / `SpeechDataset` / `SpeechCollator` with clean
  interfaces.
- Add a `dataloader/compat/` module (or similar) that can translate between
  our types and upstream types once known.
- Document every public-facing return type and method signature so the mapping
  is explicit.

**Status**: 🔨 To implement (compat layer is Phase 2).

### D4: Storage Format — .pt as Primary Everywhere (2026-03-26)

**Decision**: Use `.pt` (PyTorch serialization) as the **sole default** metadata
storage format. All new metadata flows through `PtStore`. Legacy backends
(`NpzStore`, `ParquetStore`, `JsonStore`) are retained for backward compat with
existing pipeline outputs only.

**Rationale**:
- `.pt` supports dicts, tensors, scalars, lists natively — one format for everything.
- Avoids the pipeline having to deal with format differences.
- `default_store(root)` factory always returns `PtStore`.
- Parquet stays only for manifest joins (columnar data); JSON only for
  human-readable provenance/config files written by `WaveformProcessor`.

**Action items**:
- [x] Add `PtStore(MetadataStore)` implementation.
- [x] Add `default_store()` factory.
- [x] Update `MetadataFormat` enum (`.pt` already present).
- [ ] Migrate adapter `save()`/`load()` to use `PtStore` in Phase 2 adapters.

**Status**: ✅ Implemented.

### D5: Phoneme Alignments — Deferred (2026-03-26)

**Decision**: Not in scope for Phase 2. Infrastructure supports it; will add
`PhonemeProcessor` when needed.

**Status**: ⏸ Parked.

### D6: DataBatch — More Tensor-Centric (2026-03-26)

**Decision**: Refactor `DataBatch` to favor named tensor fields over
`metadata: list[MetadataDict]`. Per-sample metadata dicts should be
projected into batch-level tensors wherever possible. Keep a
`metadata: list[MetadataDict]` escape hatch for non-tensorizable data.

**Action items**:
- [x] Add explicit tensor fields: `snr_db`, `c50_db`, `durations_s`.
- [x] Collator populates these fields; model code reads tensors directly.
- [x] Keep `metadata` list for debugging / non-tensor info only.
- [x] Add `wav_ids: list[str]` for sample identification.

**Status**: ✅ Implemented.

### D7: Distributed / Streaming — WebDataset IterableDataset (2026-03-26)

**Decision**: Implement `WebDatasetSpeechDataset` as a proper `IterableDataset`
following the user's proven pattern from their token training repo:
- `wds.WebDataset(urls, resampled=True)` for infinite epoch streaming.
- `wds.shardlists.split_by_node` for multi-node partitioning.
- `wds.shardlists.split_by_worker` for DataLoader worker partitioning.
- Shuffle buffer for sample-level randomness.
- Map-style `EvalDataset` variant for deterministic evaluation.

**Reference**: User's `TokenDataset` / `EvalDataset` implementation.

**Status**: 🔨 To implement (Phase 2–3).

### D8: C50 Clarity Metric (2026-03-26)

**Decision**: Add `c50_db` as a first-class tensor field in `DataBatch`
alongside `snr_db`. Both are per-sample scalar metrics extracted by the
Brouhaha pipeline.

**Status**: ✅ Implemented.

### D9: Codebase Cleanup & Technical Debt Reduction (2026-03-27)

**Decision**: Systematic cleanup before continuing Phase 2 adapters. Driven
by an audit of all files in the repo.

**Sub-decisions**:

1. **Delete `archive/`** — Retired adaptive-thresholding experiment.  Zero
   live references.  Git history preserves it.

2. **Extract shared Brouhaha helpers** — `_extract_brouhaha()` and
   `_load_brouhaha_pipeline()` are duplicated verbatim between `snr.py` and
   `segment_snr.py`.  Extract to `src/core/brouhaha.py`.

3. **Extract `resample_block()`** — Inline reimplementation in
   `src/packaging/writer.py` duplicates `src/core/vad_processing.py`.  Move
   the canonical function to `src/core/audio.py`, import from both sites.

4. **Add resume helper to `src/utils.py`** — 5 pipeline scripts repeat the
   same checkpoint/resume pattern.  Extract to a shared helper.

5. **Move `compare.py` → `src/plotting/compare.py`** — VAD vs VTC
   comparison is only useful for plotting/diagnostics, not production
   inference.  Relocate and update imports in `package.py`.

6. **Move `vtc_clip_alignment.py` → `src/pipeline/`** — This is a
   production validation tool (clips must snap to the 3.98 s VTC chunk
   grid).  The remaining `src/analysis/` scripts stay as-is for future
   exploratory work.

7. **Consolidate plotting boilerplate** — Create `src/plotting/utils.py`
   with `lazy_pyplot()`, `save_figure()`, `hist_with_median()`.  Replace
   6× `_setup()` copies and 8× save-figure blocks across plotting modules.

8. **Break up `package.py::main()`** — Split ~400-line `main()` into named
   functions (`_run_comparison()`, `_build_clips()`, `_compute_and_render()`,
   `_write_shards_and_manifest()`).  Same file, no new modules.

9. **Revamp plotting suite** — Many plots are exploratory leftovers.
   Identify the essential dashboards and prune the rest.  *(Future task —
   separate from the current cleanup.)*

10. **Keep `snr.py` and `segment_snr.py` separate** — `snr.py` collects
    full-duration SNR/C50 arrays; `segment_snr.py` collects speech-only
    segment-level averages (used in packaging).  Both are needed.  Shared
    Brouhaha code is extracted per item 2 above.

**Status**: ✅ Complete.

**D9 additional work (2026-03-28)**:
- Moved `set_seeds()` from `src/core/vad_processing.py` → `src/utils.py`.
  All pipeline scripts and tests updated to import from `src.utils`.
- Updated `src/plotting/compare.py` to use `lazy_pyplot()` / `save_figure()`.
- Extracted plotting code (~230 LOC) from `src/pipeline/vtc_clip_alignment.py`
  into `src/plotting/clip_alignment.py`.
- Replaced old per-topic plotting modules (overview, snr_noise, snr_vtc,
  speech_turns, packaging) with two master dashboards in `src/plotting/master.py`.
  Old modules moved to `archive/plotting/`.
- `figures.py` now dispatches to `save_master_overview()` / `save_master_quality()`
  + `print_dataset_summary()` (all in `master.py`).
- Removed `docs/CLEANUP_PROMPTS.md` and `docs/PLOTTING_AUDIT.md` (tasks done).

---

## Implementation Queue (Phase 2)

Priority order:

1. ~~**`PtStore`** — Add `.pt` metadata storage backend~~ ✅
2. ~~**`WaveformProcessor`** — Dual-mode (online/offline) waveform transforms~~ ✅
3. ~~**`DataBatch` refactor** — Tensor-centric fields (`snr_db`, `c50_db`, `durations_s`, `wav_ids`)~~ ✅
4. ~~**`WebDatasetSpeechDataset`** — IterableDataset with distributed support~~ ✅
5. ~~**Compat shim** — Placeholder for upstream type mapping~~ ✅
6. ~~**Codebase cleanup (D9)** — DRY fixes, file moves, plotting utils, package.py decomposition~~ ✅
7. ~~**`dataloader/adapters/`** — Wrap VAD, VTC, SNR, Noise as `FeatureProcessor`~~ ✅
8. ~~**Config system** — `PipelineConfig` (versioned extraction params) + `FilterConfig` (load-time data selection) + `build_manifest()` convenience function~~ ✅
9. **Loader utilities** — Extract load functions from `package.py`

**Rename**: Briefly renamed `ManifestJoiner` → `FeatureJoiner`, then reverted.
`ManifestJoiner` is correct — it joins manifests (metadata tables), not features.

---

## Open Questions

- **Q1**: What are the exact masking requirements? (attention masks, prediction
  masks, label masks — which are needed, at what granularity?)
- **Q2**: What are the metasr-internal `SpeechDataset` / `SpeechCollatorWithMasking`
  signatures? (blocked until access is granted)
- **Q3**: Should offline waveform processing produce a new manifest entry linking
  `wav_id` → processed file path, or overwrite the original?

---

## File Inventory

### `dataloader/` — Dataloader++ package (Phase 1–2)

| File | Phase | Status |
|------|-------|--------|
| `dataloader/__init__.py` | 1 | ✅ Complete |
| `dataloader/types.py` | 1 | ✅ Complete |
| `dataloader/processor/base.py` | 1 | ✅ Complete |
| `dataloader/processor/registry.py` | 1 | ✅ Complete |
| `dataloader/loader/base.py` | 1 | ✅ Complete |
| `dataloader/loader/waveform.py` | 1 | ✅ Complete |
| `dataloader/loader/metadata.py` | 1 | ✅ Complete |
| `dataloader/manifest/schema.py` | 1 | ✅ Complete |
| `dataloader/manifest/joiner.py` | 1 | ✅ Complete |
| `dataloader/manifest/store.py` | 1→2 | ✅ Complete (PtStore + default_store) |
| `dataloader/transform/base.py` | 1 | ✅ Complete |
| `dataloader/transform/audio.py` | 1 | ✅ Complete |
| `dataloader/transform/label.py` | 1 | ✅ Complete |
| `dataloader/transform/waveform.py` | 2 | ✅ WaveformProcessor + Denoiser |
| `dataloader/batch/base.py` | 1 | ✅ Complete |
| `dataloader/batch/data_batch.py` | 1→2 | ✅ Tensor-centric (snr_db, c50_db, durations_s, wav_ids) |
| `dataloader/batch/speech.py` | 1→2 | ✅ Collates snr_db, c50_db, durations_s |
| `dataloader/dataset/base.py` | 1 | ✅ Complete |
| `dataloader/dataset/webdataset.py` | 2 | ✅ WebDatasetSpeechDataset + EvalSpeechDataset |
| `dataloader/compat/__init__.py` | 2 | ✅ Created |
| `dataloader/compat/upstream.py` | 2 | ✅ Shim (to/from upstream batch/sample) |
| `dataloader/adapters/__init__.py` | 2 | ✅ Adapter package |
| `dataloader/adapters/vad.py` | 2 | ✅ VADAdapter — reads vad_meta, vad_raw, vad_merged |
| `dataloader/adapters/vtc.py` | 2 | ✅ VTCAdapter — reads vtc_meta, vtc_raw, vtc_merged |
| `dataloader/adapters/snr.py` | 2 | ✅ SNRAdapter — reads snr_meta, snr/*.npz |
| `dataloader/adapters/esc.py` | 2 | ✅ ESCAdapter — reads esc_meta, esc/*.npz |
| `dataloader/config.py` | 2 | ✅ PipelineConfig (versioned) + FilterConfig (load-time) |
| `dataloader/build.py` | 2 | ✅ build_manifest() — Big Join + filters |

### `src/` — Pipeline & shared modules (D9 cleanup)

| File | Change | Status |
|------|--------|--------|
| `src/utils.py` | Added `set_seeds()`, `load_completed_ids()` | ✅ |
| `src/core/brouhaha.py` | New: shared Brouhaha model loading + extraction | ✅ |
| `src/core/audio.py` | New: `resample_block()` | ✅ |
| `src/core/vad_processing.py` | Removed `set_seeds()` (→ utils), `resample_block()` (→ audio) | ✅ |
| `src/plotting/utils.py` | New: `lazy_pyplot()`, `save_figure()`, `hist_with_median()` | ✅ |
| `src/plotting/master.py` | New: 2 master dashboards + `print_dataset_summary()` | ✅ |
| `src/plotting/clip_alignment.py` | New: extracted from vtc_clip_alignment.py | ✅ |
| `src/plotting/figures.py` | Rewired to call master.py | ✅ |
| `src/plotting/compare.py` | Moved from pipeline/; uses lazy_pyplot/save_figure | ✅ |
| `src/pipeline/vtc_clip_alignment.py` | Moved from analysis/; plotting extracted | ✅ |
| `src/pipeline/package.py` | Decomposed main() into named functions | ✅ |
| `src/pipeline/snr.py` | Uses shared brouhaha/resume helpers | ✅ |
| `src/pipeline/segment_snr.py` | Uses shared brouhaha/resume helpers | ✅ |
| `src/pipeline/vad.py` | set_seeds from utils | ✅ |
| `src/pipeline/vtc.py` | set_seeds from utils | ✅ |
| `src/pipeline/esc.py` | set_seeds + resume from utils | ✅ |

### Archived

| File | Reason |
|------|--------|
| `archive/plotting/overview.py` | Replaced by master.py |
| `archive/plotting/snr_noise.py` | Replaced by master.py |
| `archive/plotting/snr_vtc.py` | Replaced by master.py |
| `archive/plotting/speech_turns.py` | Replaced by master.py |
| `archive/plotting/packaging.py` | Replaced by master.py |
| `archive/CLEANUP_PROMPTS.md` | All tasks completed |
| `archive/PLOTTING_AUDIT.md` | Audit completed |

### Docs

| File | Status |
|------|--------|
| `docs/PROGRESS.md` | ✅ Active (this file) |
| `docs/DATALOADER_DESIGN.md` | ✅ Architecture blueprint |
| `docs/FEATURE_EXTRACTION_SUMMARY.md` | ✅ Production reference |
| `.github/copilot-instructions.md` | ✅ Updated |
