"""Verify the local CASMI development environment and competition data."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import polars as pl
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def check_torch() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU")

    device = torch.device("cuda")
    lhs = torch.randn((1024, 1024), device=device)
    rhs = torch.randn((1024, 1024), device=device)
    result = lhs @ rhs
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"GPU smoke test: {result.shape=} {result.device=}")


def check_rdkit() -> None:
    print(f"RDKit: {rdBase.rdkitVersion}")
    molecule = Chem.MolFromSmiles("OC[C@H]1OC(O)C(O)C(O)C1O")
    if molecule is None:
        raise RuntimeError("RDKit failed to parse the smoke-test SMILES")
    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical = enumerator.Canonicalize(molecule)
    print(f"RDKit canonical SMILES: {Chem.MolToSmiles(canonical)}")


def check_data() -> None:
    expected = {
        "train.parquet": {"inchikey14", "ms2_mzs", "normalized_smiles"},
        "test.parquet": {"molecule_id", "ms2_mzs", "precursor_mz"},
        "sample_submission.csv": {"molecule_id", "smiles"},
    }

    for filename, required_columns in expected.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing competition file: {path}")

        scan = pl.scan_csv(path) if path.suffix == ".csv" else pl.scan_parquet(path)
        schema = scan.collect_schema()
        missing = required_columns.difference(schema.names())
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        row_count = scan.select(pl.len()).collect().item()
        print(f"{filename}: {row_count:,} rows, {len(schema)} columns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="check Python, Torch/CUDA, and RDKit without mounted competition files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_torch()
    check_rdkit()
    if not args.skip_data:
        check_data()
    print("Environment check passed.")
