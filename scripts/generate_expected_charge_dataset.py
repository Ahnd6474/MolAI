"""Build resumable single-channel expected-charge tensor shards from SMILES."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from rdkit import RDLogger

from molai.dft.dataset import StructureRecord, iter_structure_records
from molai.fields import (
    CompiledExpectedChargeMolecule,
    ExpectedCharge2D,
    ExpectedChargeConfig,
    ExpectedChargeTrainingBatchResult,
    TrainingChannel,
)


def replace_file_with_retry(source: Path, destination: Path, attempts: int = 6) -> None:
    """Atomically replace a file, tolerating short-lived Windows scanner locks."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * 2**attempt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--channel",
        choices=("field", "signed_charge", "electrostatic_potential"),
        default="electrostatic_potential",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--softsign-scale", type=float, default=32.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 4) // 4)),
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--id-column", default="molecule_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validation-previews", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate exact InChIKeys; disable for a source already known to be unique",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class SingleChannelShardWriter:
    """Write compact tensor shards and maintain restart/deduplication state."""

    def __init__(
        self,
        output_dir: Path,
        source: Path,
        config: ExpectedChargeConfig,
        channel: TrainingChannel,
        dtype: torch.dtype,
        shard_size: int,
        deduplicate: bool,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir / "manifest.json"
        self.keys_path = output_dir / "molecule_keys.txt"
        self.channel = channel
        self.dtype = dtype
        self.shard_size = shard_size
        self.deduplicate = deduplicate
        self.configuration = {
            "renderer": "expected_charge_v5",
            "channel": channel,
            "dtype": str(dtype).removeprefix("torch."),
            "deduplicate": deduplicate,
            "config": config.to_dict(),
        }
        self.config_hash = hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.channel_batches: list[torch.Tensor] = []
        self.canonical_smiles: list[str] = []
        self.molecule_keys: list[str] = []
        self.source_ids: list[str] = []
        self.formal_charge_batches: list[torch.Tensor] = []
        self.electron_count_batches: list[torch.Tensor] = []
        self.integrated_charge_batches: list[torch.Tensor] = []
        self.pending_records = 0
        self.pending_failed = 0
        self.pending_duplicates = 0
        self.pending_source_offset = 0
        self.pending_elements = 0
        self.pending_sum = 0.0
        self.pending_sum_of_squares = 0.0
        self.pending_minimum: float | None = None
        self.pending_maximum: float | None = None

        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["config_hash"] != self.config_hash:
                raise ValueError("existing output uses a different rendering configuration")
            if Path(self.manifest["source"]).resolve() != source.resolve():
                raise ValueError("existing output uses a different source")
        else:
            if any(output_dir.iterdir()):
                raise ValueError("output directory is non-empty but has no manifest")
            self.manifest = {
                "format": "molai-expected-charge-v5",
                "source": str(source.resolve()),
                "config_hash": self.config_hash,
                **self.configuration,
                "source_offset": 0,
                "successful_records": 0,
                "failed_records": 0,
                "duplicate_records": 0,
                "statistics": {
                    "elements": 0,
                    "sum": 0.0,
                    "sum_of_squares": 0.0,
                    "minimum": None,
                    "maximum": None,
                },
                "shards": [],
            }
            self._write_manifest()
        self.seen_keys = self._load_seen_keys() if deduplicate else None

    @property
    def source_offset(self) -> int:
        return int(self.manifest["source_offset"])

    @property
    def successful_records(self) -> int:
        return int(self.manifest["successful_records"]) + self.pending_records

    def _load_seen_keys(self) -> set[str]:
        committed_records = int(self.manifest["successful_records"])
        keys: list[str] = []
        if self.keys_path.exists():
            with self.keys_path.open("r", encoding="ascii") as handle:
                keys = [line.strip() for line in handle if line.strip()]
        if len(keys) > committed_records:
            keys = keys[:committed_records]
            self._replace_keys_file(keys)
        elif len(keys) < committed_records:
            keys = []
            for shard in self.manifest["shards"]:
                payload = torch.load(
                    self.output_dir / shard["file"], map_location="cpu", weights_only=True
                )
                keys.extend(payload["molecule_key"])
            if len(keys) != committed_records:
                raise RuntimeError("could not recover molecule keys from committed shards")
            self._replace_keys_file(keys)
        return set(keys)

    def _replace_keys_file(self, keys: list[str]) -> None:
        temporary_path = self.keys_path.with_suffix(".txt.tmp")
        temporary_path.write_text("".join(f"{key}\n" for key in keys), encoding="ascii")
        replace_file_with_retry(temporary_path, self.keys_path)

    def reserve(self, molecule: CompiledExpectedChargeMolecule) -> bool:
        if self.seen_keys is None:
            return True
        if molecule.molecule_key in self.seen_keys:
            self.pending_duplicates += 1
            return False
        self.seen_keys.add(molecule.molecule_key)
        return True

    def add_batch(
        self,
        result: ExpectedChargeTrainingBatchResult,
        source_ids: list[str],
        source_offset: int,
        failed: int,
    ) -> None:
        channel = result.channel.detach().to(device="cpu", dtype=self.dtype)
        self.channel_batches.append(channel)
        statistics_values = channel.float()
        self.pending_elements += statistics_values.numel()
        self.pending_sum += float(statistics_values.sum(dtype=torch.float64))
        self.pending_sum_of_squares += float(
            statistics_values.square().sum(dtype=torch.float64)
        )
        batch_minimum = float(statistics_values.min())
        batch_maximum = float(statistics_values.max())
        self.pending_minimum = (
            batch_minimum
            if self.pending_minimum is None
            else min(self.pending_minimum, batch_minimum)
        )
        self.pending_maximum = (
            batch_maximum
            if self.pending_maximum is None
            else max(self.pending_maximum, batch_maximum)
        )
        self.canonical_smiles.extend(result.canonical_smiles)
        self.molecule_keys.extend(result.molecule_keys)
        self.source_ids.extend(source_ids)
        self.formal_charge_batches.append(result.formal_charge.detach().cpu().to(torch.int16))
        self.electron_count_batches.append(
            result.expected_electron_count.detach().cpu().to(torch.float32)
        )
        self.integrated_charge_batches.append(
            result.integrated_charge.detach().cpu().to(torch.float32)
        )
        self.pending_records += len(result.canonical_smiles)
        self.pending_failed += failed
        self.pending_source_offset = source_offset
        if self.pending_records >= self.shard_size:
            self.flush()

    def note_empty_batch(self, source_offset: int, failed: int) -> None:
        self.pending_failed += failed
        self.pending_source_offset = source_offset

    def flush(self) -> None:
        if not self.pending_records:
            if self.pending_source_offset != self.source_offset:
                self.manifest["source_offset"] = self.pending_source_offset
                self.manifest["failed_records"] += self.pending_failed
                self.manifest["duplicate_records"] += self.pending_duplicates
                self.pending_failed = 0
                self.pending_duplicates = 0
                self._write_manifest()
            return

        channel = torch.cat(self.channel_batches)
        formal_charge = torch.cat(self.formal_charge_batches)
        electron_count = torch.cat(self.electron_count_batches)
        integrated_charge = torch.cat(self.integrated_charge_batches)
        shard_index = len(self.manifest["shards"])
        shard_name = f"fields-{shard_index:06d}.pt"
        shard_path = self.output_dir / shard_name
        temporary_path = self.output_dir / f".{shard_name}.tmp"
        torch.save(
            {
                self.channel: channel,
                "formal_charge": formal_charge,
                "electron_count": electron_count,
                "integrated_charge": integrated_charge,
                "canonical_smiles": self.canonical_smiles,
                "molecule_key": self.molecule_keys,
                "source_id": self.source_ids,
            },
            temporary_path,
        )
        replace_file_with_retry(temporary_path, shard_path)

        if self.deduplicate:
            with self.keys_path.open("a", encoding="ascii", newline="\n") as handle:
                handle.writelines(f"{key}\n" for key in self.molecule_keys)
                handle.flush()
                os.fsync(handle.fileno())

        statistics = self.manifest["statistics"]
        statistics["elements"] += self.pending_elements
        statistics["sum"] += self.pending_sum
        statistics["sum_of_squares"] += self.pending_sum_of_squares
        if self.pending_minimum is None or self.pending_maximum is None:  # pragma: no cover
            raise RuntimeError("missing statistics for a non-empty shard")
        statistics["minimum"] = (
            self.pending_minimum
            if statistics["minimum"] is None
            else min(float(statistics["minimum"]), self.pending_minimum)
        )
        statistics["maximum"] = (
            self.pending_maximum
            if statistics["maximum"] is None
            else max(float(statistics["maximum"]), self.pending_maximum)
        )
        self.manifest["shards"].append(
            {"file": shard_name, "records": self.pending_records}
        )
        self.manifest["successful_records"] += self.pending_records
        self.manifest["failed_records"] += self.pending_failed
        self.manifest["duplicate_records"] += self.pending_duplicates
        self.manifest["source_offset"] = self.pending_source_offset
        self._write_manifest()

        self.channel_batches.clear()
        self.canonical_smiles.clear()
        self.molecule_keys.clear()
        self.source_ids.clear()
        self.formal_charge_batches.clear()
        self.electron_count_batches.clear()
        self.integrated_charge_batches.clear()
        self.pending_records = 0
        self.pending_failed = 0
        self.pending_duplicates = 0
        self.pending_elements = 0
        self.pending_sum = 0.0
        self.pending_sum_of_squares = 0.0
        self.pending_minimum = None
        self.pending_maximum = None

    def _write_manifest(self) -> None:
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        replace_file_with_retry(temporary_path, self.manifest_path)


def compile_record(
    renderer: ExpectedCharge2D,
    record: StructureRecord,
) -> tuple[StructureRecord, CompiledExpectedChargeMolecule | None, str | None]:
    try:
        return record, renderer.compile_molecule(record.smiles), None
    except (RuntimeError, ValueError) as error:
        return record, None, str(error)


def save_validation_previews(
    output_dir: Path,
    result: ExpectedChargeTrainingBatchResult,
    molecules: list[CompiledExpectedChargeMolecule],
    source_ids: list[str],
    remaining: int,
    start_index: int,
) -> int:
    if remaining <= 0:
        return 0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_dir = output_dir / "validation_previews"
    preview_dir.mkdir(exist_ok=True)
    count = min(remaining, len(molecules))
    for index in range(count):
        image = result.channel[index, 0].detach().float().cpu()
        if result.channel_name == "field":
            limit = 1.0
        else:
            limit = max(float(image.abs().quantile(0.995)), 1e-8)
        figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
        artist = axis.imshow(
            image,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            extent=(-1.0, 1.0, -1.0, 1.0),
        )
        coordinates = molecules[index].coordinates
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], marker="x", s=12, color="black"
        )
        axis.set_title(
            f"{source_ids[index]} | {result.channel_name}\n"
            f"Q={float(result.integrated_charge[index]):.4f}"
        )
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        figure.savefig(preview_dir / f"sample-{start_index + index:04d}.png", dpi=150)
        plt.close(figure)
    return count


def main() -> None:
    args = parse_args()
    if args.shard_size < 1 or args.workers < 1:
        raise ValueError("shard size and worker count must be positive")
    device = torch.device(args.device)
    batch_size = args.batch_size or (128 if device.type == "cuda" else 32)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    config = ExpectedChargeConfig(
        resolution=args.resolution,
        softsign_scale=args.softsign_scale,
    )
    renderer = ExpectedCharge2D(config, device)
    writer = SingleChannelShardWriter(
        args.output,
        args.input,
        config,
        args.channel,
        dtype,
        args.shard_size,
        args.deduplicate,
    )
    records = iter_structure_records(
        args.input,
        args.smiles_column,
        args.id_column,
        deduplicate_ids=False,
    )
    records = itertools.islice(records, writer.source_offset, None)
    if args.limit is not None:
        records = itertools.islice(records, args.limit)

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    start_offset = writer.source_offset
    source_offset = start_offset
    preview_dir = args.output / "validation_previews"
    preview_count = (
        min(len(list(preview_dir.glob("sample-*.png"))), args.validation_previews)
        if preview_dir.exists()
        else 0
    )
    reported_failures = 0
    started = time.perf_counter()
    next_progress = source_offset + args.progress_every

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        batch = list(itertools.islice(records, batch_size))
        futures = [
            executor.submit(compile_record, renderer, record) for record in batch
        ]
        while batch:
            compiled_results = [future.result() for future in futures]
            if device.type == "cuda":
                next_batch = list(itertools.islice(records, batch_size))
                next_futures = [
                    executor.submit(compile_record, renderer, record)
                    for record in next_batch
                ]
            else:
                next_batch = []
                next_futures = []
            source_offset += len(batch)
            failures = 0
            molecules: list[CompiledExpectedChargeMolecule] = []
            source_ids: list[str] = []
            for record, molecule, error in compiled_results:
                if molecule is None:
                    failures += 1
                    if reported_failures < 10:
                        print(f"Skipping {record.identifier}: {error}")
                        reported_failures += 1
                    continue
                if not writer.reserve(molecule):
                    continue
                molecules.append(molecule)
                source_ids.append(record.identifier)

            if molecules:
                with torch.inference_mode():
                    result = renderer.render_training_compiled_batch(molecules, args.channel)
                preview_count += save_validation_previews(
                    args.output,
                    result,
                    molecules,
                    source_ids,
                    args.validation_previews - preview_count,
                    preview_count,
                )
                writer.add_batch(result, source_ids, source_offset, failures)
            else:
                writer.note_empty_batch(source_offset, failures)

            if source_offset >= next_progress:
                elapsed = time.perf_counter() - started
                processed = source_offset - start_offset
                rate = processed / max(elapsed, 1e-9)
                print(
                    f"source={source_offset:,} stored={writer.successful_records:,} "
                    f"rate={rate:.2f}/s"
                )
                next_progress = source_offset + args.progress_every

            if device.type != "cuda":
                next_batch = list(itertools.islice(records, batch_size))
                next_futures = [
                    executor.submit(compile_record, renderer, record)
                    for record in next_batch
                ]
            batch = next_batch
            futures = next_futures

    writer.pending_source_offset = source_offset
    writer.flush()
    elapsed = time.perf_counter() - started
    processed = source_offset - start_offset
    print(
        f"Finished source={source_offset:,} stored={writer.manifest['successful_records']:,} "
        f"elapsed={elapsed:.1f}s rate={processed / max(elapsed, 1e-9):.2f}/s"
    )
    print(f"Manifest: {writer.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
