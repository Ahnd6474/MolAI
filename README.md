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
