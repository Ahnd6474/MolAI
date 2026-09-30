"""Streaming structure readers and resumable sharded field storage."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import torch
from rdkit import Chem

from molai.dft.kohn_sham import KohnSham2DConfig, KohnSham2DResult
from molai.dft.layout import MoleculeNuclei
from molai.dft.solver import DFT2DConfig, DFT2DResult

SolverConfig = DFT2DConfig | KohnSham2DConfig
SolverResult = DFT2DResult | KohnSham2DResult
_READ_BUFFER_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class StructureRecord:
    identifier: str
    smiles: str


@contextmanager
def _open_binary_stream(path: Path) -> Iterator[BinaryIO]:
    with path.open("rb", buffering=_READ_BUFFER_BYTES) as raw:
        if path.suffix != ".gz":
            yield raw
            return
        with (
            gzip.GzipFile(fileobj=raw) as compressed,
            io.BufferedReader(compressed, buffer_size=_READ_BUFFER_BYTES) as buffered,
        ):
            yield buffered


@contextmanager
def _open_text_stream(path: Path, newline: str | None = None) -> Iterator[TextIO]:
    with (
        _open_binary_stream(path) as binary,
        io.TextIOWrapper(binary, encoding="utf-8", newline=newline) as text,
    ):
        yield text


def iter_structure_records(
    path: Path,
    smiles_column: str = "normalized_smiles",
    id_column: str = "inchikey14",
    *,
    deduplicate_ids: bool = True,
    parquet_batch_size: int = 65_536,
) -> Iterator[StructureRecord]:
    """Stream structures from Parquet, CSV, SMI, or PubChem SDF files.

    Identifier deduplication is retained for compatibility, but large-scale callers
    may disable it and deduplicate exact molecular keys downstream.
    """

    suffixes = path.suffixes
    seen: set[str] | None = set() if deduplicate_ids else None

    def unseen(identifier: str) -> bool:
        if seen is None:
            return True
        if identifier in seen:
            return False
        seen.add(identifier)
        return True

    if path.suffix == ".parquet":
        try:
            from pyarrow import parquet
        except ImportError as error:  # pragma: no cover - declared project dependency
            raise RuntimeError("Parquet streaming requires pyarrow") from error
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=parquet_batch_size, columns=[id_column, smiles_column]
        ):
            identifiers = batch.column(0).to_pylist()
            smiles_values = batch.column(1).to_pylist()
            for identifier, smiles in zip(
                identifiers, smiles_values, strict=True
            ):
                identifier_text = str(identifier)
                if smiles is not None and unseen(identifier_text):
                    yield StructureRecord(identifier_text, str(smiles))
        return
    if path.suffix == ".csv" or suffixes[-2:] == [".csv", ".gz"]:
        with _open_text_stream(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader):
                smiles = row.get(smiles_column)
                identifier = row.get(id_column) or str(row_number)
                if smiles and unseen(identifier):
                    yield StructureRecord(identifier, smiles)
        return
    text_suffix = suffixes[-2] if path.suffix == ".gz" and len(suffixes) >= 2 else path.suffix
    if text_suffix in {".smi", ".smiles", ".txt"}:
        with _open_binary_stream(path) as handle:
            for index, line in enumerate(handle):
                value = line.split(maxsplit=2)
                if value:
                    smiles = value[0].decode("utf-8")
                    identifier = value[1].decode("utf-8") if len(value) > 1 else str(index)
                    if unseen(identifier):
                        yield StructureRecord(identifier, smiles)
        return
    if path.suffix == ".sdf" or suffixes[-2:] == [".sdf", ".gz"]:
        with _open_binary_stream(path) as handle:
            supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=True)
            for index, molecule in enumerate(supplier):
                if molecule is None:
                    continue
                smiles = Chem.MolToSmiles(molecule, canonical=True)
                identifier = (
                    molecule.GetProp("PUBCHEM_COMPOUND_CID")
                    if molecule.HasProp("PUBCHEM_COMPOUND_CID")
                    else str(index)
                )
                if unseen(identifier):
                    yield StructureRecord(identifier, smiles)
        return
    raise ValueError(f"unsupported structure input: {path}")


class FieldShardWriter:
    """Accumulate compact float16 field shards with an atomic resume manifest."""

    def __init__(
        self,
        output_dir: Path,
        solver_config: SolverConfig,
        shard_size: int,
        source: Path,
        solver_name: str = "orbital_free",
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir / "manifest.json"
        self.shard_size = shard_size
        self.configuration = {"solver": solver_name, **solver_config.to_dict()}
        self.config_hash = hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.buffer: list[dict[str, object]] = []
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["config_hash"] != self.config_hash:
                raise ValueError("existing dataset uses a different DFT configuration")
            if Path(self.manifest["source"]).resolve() != source.resolve():
                raise ValueError("existing dataset uses a different source")
        else:
            existing = list(output_dir.iterdir())
            if existing:
                raise ValueError("output directory is non-empty but has no manifest")
            self.manifest = {
                "format": "molai-dft-fields-v2",
                "solver": solver_name,
                "field_channels": (
                    ["kohn_sham_fused"] if solver_name == "kohn_sham" else ["signed_charge"]
                ),
                "source": str(source.resolve()),
                "config": self.configuration,
                "config_hash": self.config_hash,
                "source_offset": 0,
                "successful_records": 0,
                "failed_records": 0,
                "shards": [],
            }
            self._write_manifest()

    @property
    def source_offset(self) -> int:
        return int(self.manifest["source_offset"])

    def add_batch(
        self,
        nuclei: list[MoleculeNuclei],
        identifiers: list[str],
        result: SolverResult,
        source_offset: int,
        failed_since_flush: int,
    ) -> None:
        for index, molecule in enumerate(nuclei):
            item: dict[str, object] = {
                "field": result.field[index].detach().cpu().to(torch.float16),
                "integrated_charge": result.integrated_charge[index].detach().cpu(),
                "formal_charge": molecule.formal_charge,
                "electron_count": molecule.electron_count,
                "smiles": molecule.canonical_smiles,
                "inchikey14": molecule.inchikey14,
                "source_id": identifiers[index],
                "iterations": result.iterations,
                "converged": result.converged,
                "final_change": (
                    result.final_relative_energy_change
                    if isinstance(result, DFT2DResult)
                    else result.final_density_change
                ),
            }
            if isinstance(result, DFT2DResult):
                item["total_energy"] = result.total_energy[index].detach().cpu()
            else:
                occupied = torch.nonzero(result.occupancies[index] > 0, as_tuple=False).flatten()
                homo_index = int(occupied[-1])
                item["homo_energy"] = result.orbital_energies[index, homo_index].detach().cpu()
                item["occupied_orbitals"] = len(occupied)
            self.buffer.append(item)
        if len(self.buffer) >= self.shard_size:
            self.flush(source_offset, failed_since_flush)

    def flush(self, source_offset: int, failed_since_flush: int = 0) -> None:
        if not self.buffer and source_offset == self.source_offset:
            return
        if self.buffer:
            shard_index = len(self.manifest["shards"])
            shard_name = f"fields-{shard_index:06d}.pt"
            shard_path = self.output_dir / shard_name
            temporary_path = self.output_dir / f".{shard_name}.tmp"
            payload = {
                "field": torch.stack([item["field"] for item in self.buffer]),
                "integrated_charge": torch.stack(
                    [item["integrated_charge"] for item in self.buffer]
                ),
                "formal_charge": torch.tensor(
                    [item["formal_charge"] for item in self.buffer], dtype=torch.int16
                ),
                "electron_count": torch.tensor(
                    [item["electron_count"] for item in self.buffer], dtype=torch.float32
                ),
                "smiles": [item["smiles"] for item in self.buffer],
                "inchikey14": [item["inchikey14"] for item in self.buffer],
                "source_id": [item["source_id"] for item in self.buffer],
                "iterations": torch.tensor(
                    [item["iterations"] for item in self.buffer], dtype=torch.int16
                ),
                "converged": torch.tensor(
                    [item["converged"] for item in self.buffer], dtype=torch.bool
                ),
                "final_change": torch.tensor(
                    [item["final_change"] for item in self.buffer],
                    dtype=torch.float32,
                ),
            }
            if "total_energy" in self.buffer[0]:
                payload["total_energy"] = torch.stack(
                    [item["total_energy"] for item in self.buffer]
                )
            if "homo_energy" in self.buffer[0]:
                payload["homo_energy"] = torch.stack([item["homo_energy"] for item in self.buffer])
                payload["occupied_orbitals"] = torch.tensor(
                    [item["occupied_orbitals"] for item in self.buffer], dtype=torch.int16
                )
            torch.save(payload, temporary_path)
            temporary_path.replace(shard_path)
            self.manifest["shards"].append({"file": shard_name, "records": len(self.buffer)})
            self.manifest["successful_records"] += len(self.buffer)
            self.buffer.clear()
        self.manifest["source_offset"] = source_offset
        self.manifest["failed_records"] += failed_since_flush
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_path.replace(self.manifest_path)
