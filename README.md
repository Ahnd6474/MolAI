# MolAI

Local workspace for the [Enveda CASMI 2026 molecule identification competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

## Environment

The project uses Python 3.12, PyTorch with the official CUDA 13.2 runtime, and the same
RDKit version (`2026.03.3`) used by the competition evaluator.

```powershell
uv python install 3.12
uv sync
uv run python scripts/check_environment.py
```

Start JupyterLab with:

```powershell
uv run jupyter lab
```

Competition files belong in `data/` and are intentionally ignored by Git:

- `data/train.parquet`
- `data/test.parquet`
- `data/sample_submission.csv`

The submitted file must be named `submission.csv`. Each test `molecule_id` appears once,
with up to 25 ranked SMILES joined by semicolons.

## Molecular field pretraining

The field teacher has two modes:

- `orbital_free`: fast 2D Thomas-Fermi-Weizsaecker approximation for large-scale warm-up.
- `kohn_sham`: occupied-orbital SCF teacher for a smaller, higher-quality subset.

The Kohn-Sham target is one phase-invariant signed charge-density image. Effective nuclei
are deposited into at most four neighboring pixels with charge-conserving bilinear
interpolation, so their grid integral equals `Z_eff`; nuclear area is not used as a proxy
for charge. The continuous negative component is the Kohn-Sham electron density. A
symmetric softsign provides a bounded, invertible image encoding, and no synthetic bond
paths are drawn. ELF and projected bond order remain separate diagnostics.
Raw orbital stacks are retained only during a solve because their channel count depends on
the molecule and orbital phase is not a stable image target. Density, deformation density,
ELF, and frontier orbitals remain available as diagnostics.

The default Kohn-Sham grid is `192 x 192`. The Cloud Matching model keeps this full spatial
resolution and reduces only its feature width; `max_resolution: 256` leaves room for later
resolution ablations.

### Graph-derived electron cloud

For structure-first pretraining, the fast electron-cloud renderer converts atom identity,
explicit hydrogen populations, and bond order into smooth signed basis functions. It is not
presented as a quantum-chemical observable: it is a deterministic, graph-faithful image
with compact positive cores and connected negative clouds suitable for exact-structure
ablation experiments. Atom-cloud width follows the RDKit van der Waals radius, bonded and
nonbonding valence populations are separated, and bond clouds shift toward the more
electronegative endpoint. The current chemistry channels include Gasteiger partial charge,
Pauling electronegativity, hybridization-directed lone-pair lobes, aromatic/conjugated
delocalization, and signed R/S plus E/Z stereochemical marks. The channels are fused into
one scalar field; the component tensors remain available for diagnostics.

Dataset generation should use `ElectronCloud2D.render_batch(smiles)`. RDKit prepares the
variable-size graphs on CPU, after which all smooth primitives are pooled and rasterized
on the GPU in memory-bounded chunks. `primitive_chunk_size` controls peak GPU memory,
while `render(smiles)` uses the identical batched path for one molecule.

```powershell
uv run python scripts/preview_electron_cloud.py `
  --smiles "CC(=O)Oc1ccccc1C(=O)O" `
  --resolution 192
```

Test whether the representation retains enough information to reconstruct a molecule:

```powershell
uv run python scripts/evaluate_electron_cloud.py `
  --input data/train.parquet `
  --limit 2000 `
  --resolution 96 `
  --render-batch-size 32 `
  --epochs 8
```

The JSON report contains exact 8-bit image collisions, nearest-neighbor image RMSE,
canonical-SMILES recovery, connectivity recovery, and SMILES validity. A paired checkpoint
stores the full-resolution CNN probe and tokenizer vocabulary. The small probe measures
information retention; it is not the final competition spectrum-to-structure model.

Generate resumable float16 shards with:

```powershell
uv run python scripts/generate_dft_dataset.py `
  --config configs/kohn_sham.yaml `
  --input data/train.parquet `
  --output data/kohn_sham_fields
```

The manifest records convergence and final density change for every sample. Non-converged
samples remain auditable instead of silently being treated as converged. Pretrain the
single-channel, full-resolution axial/local Cloud Matching model with:

```powershell
uv run python scripts/train_cloud.py `
  --data data/kohn_sham_fields `
  --config configs/model_ks.yaml
```

Preview a molecule and its HOMO diagnostic with:

```powershell
uv run python scripts/preview_kohn_sham.py --smiles "c1ccccc1"
uv run python scripts/preview_kohn_sham.py --smiles "c1ccccc1" --diagnostics
```

This is a two-dimensional effective-valence teacher, not a replacement for a converged
three-dimensional quantum-chemistry package. Direct Kohn-Sham generation over all PubChem
structures is intentionally not the plan: use a curated converged subset to train a fast
field surrogate, then use the surrogate for scale.
