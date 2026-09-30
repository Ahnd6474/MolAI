"""Pack every training spectrum into CSR shards aligned with molecular fields."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import duckdb
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-rows", type=int, default=2048)
    return parser.parse_args()


def _resolve_raw_fields(root: Path) -> tuple[Path, dict[str, object]]:
    current = root.resolve()
    while True:
        manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "molai-noise-view-v1":
            return current, manifest
        source = Path(str(manifest["source"])).expanduser()
        current = source if source.is_absolute() else (current / source).resolve()


def _load_ordered_keys(raw_root: Path, manifest: dict[str, object]) -> list[str]:
    keys_path = raw_root / "molecule_keys.txt"
    if keys_path.is_file():
        keys = keys_path.read_text(encoding="ascii").splitlines()
    else:
        keys = []
        for shard in manifest["shards"]:
            payload = torch.load(
                raw_root / shard["file"], map_location="cpu", weights_only=False
            )
            key_name = "source_id" if "source_id" in payload else "molecule_key"
            keys.extend(str(value) for value in payload[key_name])
    expected = sum(int(shard["records"]) for shard in manifest["shards"])
    if len(keys) != expected:
        raise ValueError(f"found {len(keys):,} keys for {expected:,} field records")
    if keys != sorted(keys):
        raise ValueError("field keys must be sorted for streaming spectrum preparation")
    return keys


class SpectrumShardBuffers:
    def __init__(self, records: int, metadata_dim: int) -> None:
        self.records = records
        self.metadata_dim = metadata_dim
        self.mz: list[torch.Tensor] = []
        self.intensity: list[torch.Tensor] = []
        self.peak_counts: list[int] = []
        self.spectrum_counts = torch.zeros(records, dtype=torch.int64)
        self.metadata: list[torch.Tensor] = []
        self.precursor_mz: list[float] = []

    def add(
        self,
        local_index: int,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        metadata: torch.Tensor,
        precursor_mz: float,
    ) -> None:
        self.mz.append(mz)
        self.intensity.append(intensity)
        self.peak_counts.append(len(mz))
        self.spectrum_counts[local_index] += 1
        self.metadata.append(metadata)
        self.precursor_mz.append(precursor_mz)

    def payload(self) -> dict[str, torch.Tensor]:
        peak_counts = torch.tensor(self.peak_counts, dtype=torch.int64)
        peak_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), peak_counts.cumsum(0)))
        spectrum_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int64), self.spectrum_counts.cumsum(0))
        )
        return {
            "mz": torch.cat(self.mz),
            "intensity": torch.cat(self.intensity),
            "peak_offsets": peak_offsets,
            "spectrum_offsets": spectrum_offsets,
            "metadata": torch.stack(self.metadata).half(),
            "precursor_mz": torch.tensor(self.precursor_mz, dtype=torch.float32),
        }


def _write_shard(output: Path, index: int, buffers: SpectrumShardBuffers) -> str:
    name = f"spectra-{index:06d}.pt"
    temporary = output / f".{name}.tmp"
    torch.save(buffers.payload(), temporary)
    os.replace(temporary, output / name)
    return name


def _metadata_features(
    ionization_mode: str | None,
    precursor_error_ppm: float | None,
    collision_energy: list[float] | None,
    num_peaks: int,
) -> torch.Tensor:
    polarity = {"positive": 1.0, "negative": -1.0}.get(ionization_mode or "", 0.0)
    error = (
        math.tanh(precursor_error_ppm / 20.0)
        if precursor_error_ppm is not None and math.isfinite(precursor_error_ppm)
        else 0.0
    )
    finite_collision = [
        float(value)
        for value in (collision_energy or [])
        if value is not None and math.isfinite(float(value))
    ]
    if finite_collision:
        mean = sum(finite_collision) / len(finite_collision)
        variance = sum((value - mean) ** 2 for value in finite_collision) / len(
            finite_collision
        )
        collision_mean = mean / 100.0
        collision_std = math.sqrt(variance) / 100.0
        collision_missing = 0.0
    else:
        collision_mean = 0.0
        collision_std = 0.0
        collision_missing = 1.0
    peak_scale = min(math.log1p(max(num_peaks, 0)) / math.log1p(100_000), 1.0)
    return torch.tensor(
        [polarity, error, collision_mean, collision_std, collision_missing, peak_scale]
    )


def _clean_peaks(
    mz_values: list[float], intensity_values: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(len(mz_values), len(intensity_values))
    mz = torch.tensor(
        [float("nan") if value is None else value for value in mz_values[:count]],
        dtype=torch.float32,
    )
    intensity = torch.tensor(
        [float("nan") if value is None else value for value in intensity_values[:count]],
        dtype=torch.float32,
    )
    valid = torch.isfinite(mz) & torch.isfinite(intensity) & mz.ge(0.0)
    mz = mz[valid]
    intensity = intensity[valid].clamp(0.0, 1.0)
    order = mz.argsort()
    return mz[order], intensity[order].half()


def main() -> None:
    args = parse_args()
    fields_root = args.fields.resolve()
    raw_root, raw_manifest = _resolve_raw_fields(fields_root)
    keys = _load_ordered_keys(raw_root, raw_manifest)
    key_to_index = {key: index for index, key in enumerate(keys)}
    shard_records = [int(shard["records"]) for shard in raw_manifest["shards"]]
    shard_starts: list[int] = []
    total = 0
    for records in shard_records:
        shard_starts.append(total)
        total += records

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError(f"output dataset already exists: {args.output}")

    query = """
        SELECT
            inchikey,
            precursor_mz,
            ms2_mzs,
            ms2_normalized_intensities,
            ionization_mode,
            precursor_error_ppm,
            collision_energy_ev,
            coalesce(num_peaks, array_length(ms2_mzs)) AS observed_peaks
        FROM read_parquet(?)
        WHERE ms2_mzs IS NOT NULL
          AND ms2_normalized_intensities IS NOT NULL
          AND array_length(ms2_mzs) > 0
        ORDER BY inchikey
    """
    connection = duckdb.connect()
    cursor = connection.execute(query, [str(args.spectra.resolve())])

    metadata_dim = 6
    output_shards: list[dict[str, object]] = []
    current_shard = 0
    buffers = SpectrumShardBuffers(shard_records[current_shard], metadata_dim)
    previous_global_index = -1
    matched_molecules = torch.zeros(len(keys), dtype=torch.bool)
    spectrum_count = 0
    peak_count = 0

    while rows := cursor.fetchmany(args.fetch_rows):
        for row in rows:
            molecule_key = str(row[0])
            global_index = key_to_index.get(molecule_key)
            if global_index is None:
                continue
            if global_index < previous_global_index:
                raise RuntimeError("spectrum query order does not match sorted field keys")
            previous_global_index = global_index
            while global_index >= shard_starts[current_shard] + shard_records[current_shard]:
                name = _write_shard(args.output, current_shard, buffers)
                output_shards.append({"file": name, "records": shard_records[current_shard]})
                print(f"wrote {name}", flush=True)
                current_shard += 1
                buffers = SpectrumShardBuffers(shard_records[current_shard], metadata_dim)

            mz, intensity = _clean_peaks(row[2], row[3])
            if len(mz) < 1:
                continue
            local_index = global_index - shard_starts[current_shard]
            precursor_mz = float(row[1] or 0.0)
            if not math.isfinite(precursor_mz):
                precursor_mz = 0.0
            buffers.add(
                local_index,
                mz,
                intensity,
                _metadata_features(row[4], row[5], row[6], int(row[7])),
                precursor_mz,
            )
            matched_molecules[global_index] = True
            spectrum_count += 1
            peak_count += len(mz)

    while current_shard < len(shard_records):
        name = _write_shard(args.output, current_shard, buffers)
        output_shards.append({"file": name, "records": shard_records[current_shard]})
        print(f"wrote {name}", flush=True)
        current_shard += 1
        if current_shard < len(shard_records):
            buffers = SpectrumShardBuffers(shard_records[current_shard], metadata_dim)

    missing = int((~matched_molecules).sum())
    if missing:
        raise RuntimeError(f"{missing:,} field molecules have no usable spectra")
    field_manifest = json.loads((fields_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "molai-spectrum-field-v1",
        "field_source": os.path.relpath(fields_root, args.output.resolve()),
        "spectra_source": str(args.spectra.resolve()),
        "records": len(keys),
        "spectra": spectrum_count,
        "peaks": peak_count,
        "metadata_dim": metadata_dim,
        "field_channels": raw_manifest.get("field_channels", ["signed_charge"]),
        "metadata_features": [
            "ionization_polarity",
            "tanh_precursor_error_ppm_over_20",
            "collision_energy_mean_over_100",
            "collision_energy_std_over_100",
            "collision_energy_missing",
            "log_peak_count",
        ],
        "seed": field_manifest.get("seed", 0),
        "noise_schedule": field_manifest.get("noise_schedule"),
        "shards": output_shards,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"prepared {len(keys):,} molecules, {spectrum_count:,} spectra, "
        f"and {peak_count:,} peaks",
        flush=True,
    )


if __name__ == "__main__":
    main()
