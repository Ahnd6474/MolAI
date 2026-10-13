"""Verify a complete v2 repack against old molecule alignment and raw spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import duckdb
import torch
from prepare_spectrum_field_dataset import _clean_peaks, _resolve_raw_fields

from molai.data.spectrum_metadata import ACQUISITION_COLUMNS, SpectrumMetadataSchema


def signature(mz, intensity, metadata, precursor):
    digest = hashlib.sha256()
    for value in (mz, intensity, metadata):
        digest.update(value.contiguous().numpy().tobytes())
    digest.update(struct.pack("<f", precursor))
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data: Path, reference: Path, spectra: Path) -> dict:
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((reference / "manifest.json").read_text())
    schema = SpectrumMetadataSchema.from_dict(manifest["metadata_schema"])
    names = schema.feature_names
    check(manifest["metadata_features"] == names, "metadata feature order mismatch")
    check(manifest["metadata_dim"] == len(names), "metadata dimension mismatch")
    check(not any("precursor_error_ppm" in name for name in names), "label-derived condition")
    field_path = (data / manifest["field_source"]).resolve()
    check(field_path == (reference / original["field_source"]).resolve(), "image source changed")
    for key in ("records", "spectra", "peaks", "seed", "noise_schedule"):
        check(manifest.get(key) == original.get(key), f"reference {key} changed")
    check([s["records"] for s in manifest["shards"]] ==
          [s["records"] for s in original["shards"]], "molecular shard layout changed")
    record_count = int(manifest["records"])
    selected = sorted({round(index * (record_count - 1) / 7) for index in range(8)})
    raw_root, field_manifest = _resolve_raw_fields(field_path)
    sampled_signatures = {}
    molecule_total = spectrum_total = peak_total = 0
    total_bytes = 0
    for index, shard in enumerate(manifest["shards"]):
        path = data / shard["file"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        previous = torch.load(reference / original["shards"][index]["file"],
                              map_location="cpu", weights_only=False)
        peak_offsets, spectrum_offsets = payload["peak_offsets"], payload["spectrum_offsets"]
        mz, intensity, metadata, precursor = (payload[name] for name in
                                              ("mz", "intensity", "metadata", "precursor_mz"))
        count, spectra_count, peaks_count = shard["records"], len(metadata), len(mz)
        for name in ("mz", "intensity", "metadata", "precursor_mz"):
            value = payload[name]
            check(value.dtype == torch.float32, f"{shard['file']}: {name} not float32")
            check(bool(torch.isfinite(value).all()), f"{shard['file']}: nonfinite {name}")
        check(metadata.shape == (spectra_count, len(names)), "metadata shape mismatch")
        check(intensity.shape == mz.shape and precursor.shape == (spectra_count,), "data shape mismatch")
        check(len(peak_offsets) == spectra_count + 1 and len(spectrum_offsets) == count + 1,
              "CSR shape mismatch")
        check(peak_offsets[0] == spectrum_offsets[0] == 0, "CSR must start at zero")
        check(peak_offsets[-1] == peaks_count and spectrum_offsets[-1] == spectra_count,
              "CSR end mismatch")
        check(bool((peak_offsets.diff() > 0).all()), "empty spectrum")
        check(bool((spectrum_offsets.diff() > 0).all()), "molecule without spectra")
        check(torch.equal(spectrum_offsets.diff(), previous["spectrum_offsets"].diff()),
              "per-molecule spectrum count changed")
        check(torch.equal(peak_offsets[spectrum_offsets].diff(),
                          previous["peak_offsets"][previous["spectrum_offsets"]].diff()),
              "per-molecule peak count changed")
        local_samples = [value - molecule_total for value in selected
                         if molecule_total <= value < molecule_total + count]
        if local_samples:
            fields = torch.load(raw_root / field_manifest["shards"][index]["file"],
                                map_location="cpu", weights_only=False)
            key_name = "source_id" if "source_id" in fields else "molecule_key"
            for local in local_samples:
                key = str(fields[key_name][local])
                start, end = int(spectrum_offsets[local]), int(spectrum_offsets[local + 1])
                sampled_signatures[key] = Counter(
                    signature(mz[peak_offsets[s]:peak_offsets[s + 1]],
                              intensity[peak_offsets[s]:peak_offsets[s + 1]], metadata[s],
                              float(precursor[s])) for s in range(start, end)
                )
            del fields
        molecule_total += count
        spectrum_total += spectra_count
        peak_total += peaks_count
        total_bytes += path.stat().st_size
        del payload, previous
        print(f"verified shard {index + 1}/{len(manifest['shards'])}", flush=True)
    expected = (manifest["records"], manifest["spectra"], manifest["peaks"])
    check((molecule_total, spectrum_total, peak_total) == expected, "global counts mismatch")
    keys = list(sampled_signatures)
    check(len(keys) == len(selected), "missing sampled molecule keys")
    raw_signatures = {key: Counter() for key in keys}
    placeholders = ",".join("?" for _ in keys)
    query = (f"SELECT inchikey, ms2_mzs, ms2_normalized_intensities, "
             f"{', '.join(ACQUISITION_COLUMNS)} FROM read_parquet(?) "
             f"WHERE inchikey IN ({placeholders})")
    with duckdb.connect() as connection:
        for row in connection.execute(query, [str(spectra), *keys]).fetchall():
            if row[1] is None or row[2] is None:
                continue
            clean_mz, clean_intensity = _clean_peaks(row[1], row[2])
            if len(clean_mz) == 0:
                continue
            acquisition = dict(zip(ACQUISITION_COLUMNS, row[3:], strict=True))
            measured = float(acquisition["precursor_mz"] or 0.)
            if not math.isfinite(measured) or measured <= 0:
                measured = 0.
            raw_signatures[row[0]][signature(
                clean_mz, clean_intensity, schema.encode(acquisition, len(clean_mz)), measured,
            )] += 1
    check(raw_signatures == sampled_signatures, "sampled peaks/metadata differ from raw Parquet")
    return {"verified": True, "molecules": molecule_total, "spectra": spectrum_total,
            "peaks": peak_total, "shards": len(manifest["shards"]), "metadata_dim": len(names),
            "bytes": total_bytes, "all_numeric_tensors_finite": True,
            "all_molecule_spectrum_and_peak_counts_match_reference": True,
            "sampled_molecules_matching_raw": len(keys),
            "sampled_spectra_matching_raw": sum(sum(c.values()) for c in raw_signatures.values()),
            "image_source_unchanged": True, "seed_and_noise_view_unchanged": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.data, args.reference, args.spectra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
