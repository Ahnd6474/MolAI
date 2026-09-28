"""Generate resumable, sharded 2D pseudo-DFT fields from molecular structures."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
import yaml

from molai.dft import (
    DFT2DConfig,
    KohnSham2D,
    KohnSham2DConfig,
    OrbitalFreeDFT2D,
    canonical_nuclei,
    collate_nuclei,
)
from molai.dft.dataset import FieldShardWriter, StructureRecord, iter_structure_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/dft_fields"))
    parser.add_argument("--config", type=Path, default=Path("configs/dft.yaml"))
    parser.add_argument("--solver", choices=("auto", "orbital_free", "kohn_sham"), default="auto")
    parser.add_argument("--smiles-column", default="normalized_smiles")
    parser.add_argument("--id-column", default="inchikey14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scf-iterations", type=int)
    parser.add_argument("--orbital-steps", type=int)
    return parser.parse_args()


def _chunks(records: list[StructureRecord], size: int) -> list[list[StructureRecord]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


def main() -> None:
    args = parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    solver_name = args.solver
    if solver_name == "auto":
        solver_name = "kohn_sham" if "kohn_sham" in configuration else "orbital_free"
    config_key = "kohn_sham" if solver_name == "kohn_sham" else "dft"
    dft_values = configuration[config_key]
    if args.resolution is not None:
        dft_values["resolution"] = args.resolution
    if args.steps is not None and solver_name == "orbital_free":
        dft_values["steps"] = args.steps
    if args.scf_iterations is not None and solver_name == "kohn_sham":
        dft_values["scf_iterations"] = args.scf_iterations
    if args.orbital_steps is not None and solver_name == "kohn_sham":
        dft_values["orbital_steps"] = args.orbital_steps
    dft_config = (
        KohnSham2DConfig(**dft_values) if solver_name == "kohn_sham" else DFT2DConfig(**dft_values)
    )
    dataset_config = configuration["dataset"]
    layout_config = configuration["layout"]
    batch_size = args.batch_size or int(dataset_config["batch_size"])
    shard_size = args.shard_size or int(dataset_config["shard_size"])

    writer = FieldShardWriter(args.output, dft_config, shard_size, args.input, solver_name)
    records = iter_structure_records(args.input, args.smiles_column, args.id_column)
    records = itertools.islice(records, writer.source_offset, None)
    if args.limit is not None:
        records = itertools.islice(records, args.limit)
    solver = (
        KohnSham2D(dft_config, args.device)
        if isinstance(dft_config, KohnSham2DConfig)
        else OrbitalFreeDFT2D(dft_config, args.device)
    )

    source_offset = writer.source_offset
    processed_this_run = 0
    failed_since_flush = 0
    start_time = time.perf_counter()
    pending: list[StructureRecord] = []
    for record in records:
        pending.append(record)
        if len(pending) < batch_size:
            continue
        for group in _chunks(pending, batch_size):
            source_offset += len(group)
            processed_this_run += len(group)
            molecules = []
            identifiers = []
            for item in group:
                try:
                    molecules.append(
                        canonical_nuclei(
                            item.smiles,
                            extent=dft_config.extent,
                            margin=float(layout_config["margin"]),
                            target_bond_length=float(layout_config["target_bond_length"]),
                            collapse_hydrogens=bool(layout_config["collapse_hydrogens"]),
                        )
                    )
                    identifiers.append(item.identifier)
                except (ValueError, RuntimeError) as error:
                    failed_since_flush += 1
                    print(f"Skipping {item.identifier}: {error}")
            if molecules:
                result = solver.solve(collate_nuclei(molecules, args.device))
                writer.add_batch(
                    molecules,
                    identifiers,
                    result,
                    source_offset,
                    failed_since_flush,
                )
                if not writer.buffer:
                    failed_since_flush = 0
                elapsed = time.perf_counter() - start_time
                rate = processed_this_run / max(elapsed, 1e-6)
                print(
                    f"source={source_offset:,} stored={writer.manifest['successful_records'] + len(writer.buffer):,} "
                    f"iterations={result.iterations} converged={result.converged} rate={rate:.2f}/s"
                )
        pending.clear()

    if pending:
        source_offset += len(pending)
        processed_this_run += len(pending)
        molecules = []
        identifiers = []
        for item in pending:
            try:
                molecules.append(
                    canonical_nuclei(
                        item.smiles,
                        extent=dft_config.extent,
                        margin=float(layout_config["margin"]),
                        target_bond_length=float(layout_config["target_bond_length"]),
                        collapse_hydrogens=bool(layout_config["collapse_hydrogens"]),
                    )
                )
                identifiers.append(item.identifier)
            except (ValueError, RuntimeError) as error:
                failed_since_flush += 1
                print(f"Skipping {item.identifier}: {error}")
        if molecules:
            result = solver.solve(collate_nuclei(molecules, args.device))
            writer.add_batch(molecules, identifiers, result, source_offset, failed_since_flush)
    writer.flush(source_offset, failed_since_flush)
    print(f"Finished: {writer.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
