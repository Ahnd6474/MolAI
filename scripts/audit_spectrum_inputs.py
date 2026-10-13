"""Audit real CASMI train/test conditioning without reading structure labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from molai.data.spectrum_metadata import CATEGORY_COLUMNS, SpectrumMetadataSchema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/test.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = SpectrumMetadataSchema.fit_training_parquet(args.train)
    train, test = pq.ParquetFile(args.train), pq.ParquetFile(args.test)
    rows = test.read().to_pylist()
    encoded = torch.stack([schema.encode(row, len(row["ms2_mzs"])) for row in rows])
    if not torch.isfinite(encoded).all():
        raise ValueError("non-finite test acquisition features")
    names = schema.feature_names
    report = {
        "train_rows": train.metadata.num_rows, "test_rows": test.metadata.num_rows,
        "train_columns": train.schema_arrow.names, "test_columns": test.schema_arrow.names,
        "train_only_columns": sorted(set(train.schema_arrow.names) - set(test.schema_arrow.names)),
        "metadata_dim": len(names), "metadata_features": names,
        "test_unseen_categories": {
            column: int(encoded[:, names.index(f"{column}:unseen")].sum())
            for column in CATEGORY_COLUMNS
        },
        "test_adduct_mass_valid_rows": int(encoded[:, names.index("adduct_mass_valid")].sum()),
        "test_metadata_finite": True,
        "test_categories": {column: sorted({str(row[column]) for row in rows})
                            for column in CATEGORY_COLUMNS},
        "schema": schema.to_dict(),
        "notes": ["Only acquisition features are encoded; IDs are used for grouping/alignment only.",
                  "precursor_error_ppm is train-only and excluded from conditioning.",
                  "Original collision text is numerically parsed, not a learned text embedding.",
                  "This checks input compatibility, not generation accuracy."],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata-schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "train_rows", "test_rows", "metadata_dim", "test_unseen_categories",
        "test_adduct_mass_valid_rows", "test_metadata_finite")}, indent=2))


if __name__ == "__main__":
    main()
