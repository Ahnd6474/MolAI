# CASMI acquisition conditioning (2026-10-12)

Audit of the local competition files: train has 2,539,575 spectra; test has
1,213 spectra. Test has seven adduct types, both polarities, `timsTOF` instruments,
and eV collision energies. These are observations of the files, not assumptions
that future competition files must satisfy.

The old packer retained only six metadata values: polarity, label-derived
precursor error, collision-energy mean/std/missing, and peak count. In particular,
`precursor_error_ppm` does not exist in test and must not condition the model.
Historical validation used that feature and is not a clean estimate of test-time
performance. No claim is made here about the magnitude of its effect.

## New inputs

The acquisition schema uses only information available at inference:

| Information | Encoding |
| --- | --- |
| Fragment m/z, normalized intensity | Existing sparse peak path, all valid peaks |
| Precursor m/z | Existing precursor MLP plus linear-scaled metadata value |
| Adduct, polarity, instrument, original energy units | Train-fitted categorical one-hot, separate missing/unseen states |
| Collision-energy eV list | Sorted values with per-slot masks, not only mean/std |
| Original collision energy | Parsed signed numeric values and unit category, missing/parse-failure flags |
| Base-peak intensity | log1p value and presence mask |
| Peak count | Computed from the valid observed peaks, same logic in train/test |
| Charge, multimer count, mass shift, estimated neutral mass | Calculated from adduct and measured precursor; validity mask for unsupported strings |

Original collision-energy free text is **not** fully semantically interpreted.
Its numeric values are encoded; units are not assumed to be equivalent to eV.
Train-only structure labels, formula, precursor error, source library, and
`adduct_orig` are excluded. IDs are for grouping/alignment, never neural inputs.
No MS1/isotope-envelope/retention-time columns exist in these supplied files.

The fitted schema has 243 metadata features, six eV slots and six original-energy
slots. All 1,213 test rows encode with finite values, no unseen categories, and
valid adduct mass conversion. Lists exceeding the fitted capacity raise an error
instead of silently truncating. Numeric metadata and peak intensities are stored
as float32. This increases storage relative to old float16 shards.

The new default model config uses `metadata_fusion: preconv_cross`. Acquisition
features become **seven typed tokens per spectrum**: polarity, adduct, precursor,
instrument, eV energy list, original energy/units, and intensity/peak count. Each
group has its own MLP and a learned type embedding. Missing values remain explicit
features in valid type tokens; they do not create an all-masked attention row.

Peak embeddings are Q and these seven tokens are K,V. A pre-norm cross-attention
updates the peaks through `peak + scalar_gate * cross_update`, with signed gate
initialized to 0.1. K/V are projected once per spectrum, then gathered by each
chunk's spectrum owner; they are not shared across different spectra. There is
**no unconditional additive metadata bypass** in this mode. The update precedes
both convolution/pooling stages; no second metadata cross is added after pooling.

Peak width stays 64; Conv/pool twice, self-attention twice and image-Q/MS-KV cross
attention are unchanged. A 243-feature input is not a 243-dimensional MS token.
The exact metadata feature order is bound from the dataset into the saved model
config, so checkpoint reconstruction uses identical group layouts. The training
entry points reject legacy v1 metadata when preconv_cross is requested.

## Preparing a new dataset

```powershell
.venv/Scripts/python.exe scripts/audit_spectrum_inputs.py `
  --output artifacts/diagnostics/competition_ms_20261012

.venv/Scripts/python.exe scripts/prepare_spectrum_field_dataset.py `
  --fields data/expected-charge-128 `
  --spectra data/train.parquet `
  --metadata-schema artifacts/diagnostics/competition_ms_20261012/metadata-schema.json `
  --output data/expected-charge-128-spectrum-v2
```

Use the same `SpectrumMetadataSchema.from_dict(...).encode(row, valid_peak_count)`
at inference and group all spectra by `molecule_id`. Do not fit on test or allocate
independent category IDs. The audit fits on the supplied train file; for strict
fold experiments, fit vocabulary/capacity on the training fold alone and reuse it
for the held-out fold. Unknown categories are explicitly supported.

Old packed datasets/checkpoints are not rewritten. Encoders without an explicit
`metadata_fusion` setting retain the historical `add` path and identical state
keys. They do not gain the new features or cross automatically. Repack into a
**new** output directory. Typed metadata projections and cross-attention are new
parameters, so the old optimizer/encoder is not a strict resume for this topology.
Start a new run or explicitly migrate the affected weights. This change does not
start training, modify the loss, or demonstrate a generation-quality improvement.

Verify a completed repack against the old alignment and raw acquisition rows:

```powershell
.venv/Scripts/python.exe scripts/validate_spectrum_repack.py `
  --data data/expected-charge-128-spectrum-v2 `
  --reference data/expected-charge-128-spectrum `
  --spectra data/train.parquet `
  --output artifacts/preparations/competition_ms_v2_20261012/validation.json
```

The validator checks every shard's CSR structure, finite float32 tensors,
per-molecule spectrum and peak counts, and unchanged image/noise-view references.
It compares order-independent hashes of all spectra for eight sampled molecules
against peaks and metadata re-encoded directly from raw Parquet. It does not
claim a full raw-row recomputation of all 2.5 million spectra.

Remaining possible bottlenecks: the absolute peak affine lookup uses 0.01 m/z
bins (raw m/z gaps still reach the distance convolution); precise neutral-loss
tokens, spectrum reliability weighting, and test-domain-aware validation have
not been added by this change.
