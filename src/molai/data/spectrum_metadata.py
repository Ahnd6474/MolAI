"""Train-fitted acquisition metadata available at CASMI inference time.

Labels, library identifiers and label-derived precursor errors are deliberately
not accepted as features. Reuse the serialized schema unchanged for validation
and test; never fit category IDs separately on those splits.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import duckdb
import torch
from rdkit import Chem

CATEGORY_COLUMNS = ("ionization_mode", "adduct", "instrument_type", "collision_energy_orig_units")
ACQUISITION_COLUMNS = (
    *CATEGORY_COLUMNS, "precursor_mz", "base_peak_intensity", "collision_energy_ev",
    "collision_energy_orig",
)
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ELECTRON_MASS = 0.000548579909065


def _category(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return None if text in {"", "nan", "none", "null", "n/a", "unknown"} else text


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _energies(value: object) -> list[float]:
    if value is None:
        return []
    return sorted(number for item in value if (number := _finite(item)) is not None)


def _original_energies(value: object) -> list[float]:
    if value is None:
        return []
    return [number for token in re.findall(NUMBER_PATTERN, str(value))
            if (number := _finite(token)) is not None]


@lru_cache(maxsize=1024)
def _adduct_parameters(adduct: str) -> tuple[int, int, float] | None:
    match = re.fullmatch(r"\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])",
                         adduct.strip())
    if match is None:
        return None
    multiplicity = int(match[1] or 1)
    charge = int(match[3] or 1) * (1 if match[4] == "+" else -1)
    if multiplicity < 1 or charge == 0:
        return None
    shift = 0.0
    table = Chem.GetPeriodicTable()
    for sign, term in re.findall(r"([+-])([A-Za-z0-9]+)", match[2]):
        formula_match = re.fullmatch(r"(\d*)([A-Z].*)", term)
        if formula_match is None:
            return None
        coefficient = int(formula_match[1] or 1)
        formula = formula_match[2]
        atoms = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if "".join(symbol + count for symbol, count in atoms) != formula:
            return None
        atom_mass = 0.0
        for symbol, count in atoms:
            try:
                mass = table.GetMostCommonIsotopeMass(symbol)
            except RuntimeError:
                return None
            atom_mass += mass * int(count or 1)
        shift += (1 if sign == "+" else -1) * coefficient * atom_mass
    return charge, multiplicity, shift


def adduct_mass_features(adduct: object, precursor_mz: object) -> list[float]:
    """Return charge, multimer count, atom-mass shift, neutral mass, validity.

    This is a deterministic inference-time calculation, not a formula label.
    Unsupported/non-formula adducts have an explicit invalid mask.
    """
    mz = _finite(precursor_mz)
    parameters = _adduct_parameters(str(adduct))
    if parameters is None or mz is None or mz <= 0:
        return [0.0] * 5
    charge, multiplicity, shift = parameters
    neutral_mass = (mz * abs(charge) - shift + charge * ELECTRON_MASS) / multiplicity
    if neutral_mass <= 0:
        return [0.0] * 5
    return [float(charge), float(multiplicity), shift / 100.0, neutral_mass / 1000.0, 1.0]


@dataclass
class SpectrumMetadataSchema:
    categories: dict[str, list[str]]
    energy_slots: int
    original_energy_slots: int

    @classmethod
    def fit_training_parquet(cls, path: Path | str) -> SpectrumMetadataSchema:
        """Read only acquisition columns; category dictionaries never use labels."""
        with duckdb.connect() as connection:
            categories = {}
            for column in CATEGORY_COLUMNS:
                values = connection.execute(
                    f"SELECT DISTINCT {column} FROM read_parquet(?)", [str(path)]
                ).fetchall()
                categories[column] = sorted({text for (value,) in values
                                             if (text := _category(value)) is not None})
            capacity = connection.execute(
                "SELECT coalesce(max(array_length(collision_energy_ev)), 0), "
                "coalesce(max(array_length(regexp_extract_all(collision_energy_orig, ?))), 0) "
                "FROM read_parquet(?)", [NUMBER_PATTERN, str(path)],
            ).fetchone()
        return cls(categories, max(1, int(capacity[0])), max(1, int(capacity[1])))

    @property
    def feature_names(self) -> list[str]:
        names = []
        for column in CATEGORY_COLUMNS:
            names += [f"{column}:missing", f"{column}:unseen"]
            names += [f"{column}:{value}" for value in self.categories[column]]
        names += ["precursor_mz_over_1000", "precursor_mz_present",
                  "log1p_base_peak_intensity", "base_peak_intensity_present", "log1p_peak_count",
                  "adduct_charge", "adduct_multimer_count", "adduct_atom_mass_shift_over_100",
                  "adduct_neutral_mass_over_1000", "adduct_mass_valid"]
        for prefix, slots in (("collision_ev", self.energy_slots),
                              ("collision_orig", self.original_energy_slots)):
            names += [f"{prefix}_count"]
            for index in range(slots):
                names += [f"{prefix}_{index}_over_100", f"{prefix}_{index}_present"]
        names += ["collision_orig_text_present", "collision_orig_numeric_parse_failed"]
        return names

    def to_dict(self) -> dict[str, object]:
        return {"format": "casmi-acquisition-metadata-v2", "categories": self.categories,
                "energy_slots": self.energy_slots,
                "original_energy_slots": self.original_energy_slots}

    @classmethod
    def from_dict(cls, value: dict) -> SpectrumMetadataSchema:
        if value.get("format") != "casmi-acquisition-metadata-v2":
            raise ValueError("unsupported acquisition metadata schema")
        return cls(value["categories"], int(value["energy_slots"]),
                   int(value["original_energy_slots"]))

    def encode(self, row: Mapping[str, object], num_peaks: int) -> torch.Tensor:
        features = []
        for column in CATEGORY_COLUMNS:
            value = _category(row.get(column))
            vocabulary = self.categories[column]
            one_hot = [0.0] * (len(vocabulary) + 2)
            index = 0 if value is None else (vocabulary.index(value) + 2
                                             if value in vocabulary else 1)
            one_hot[index] = 1.0
            features.extend(one_hot)
        precursor = _finite(row.get("precursor_mz"))
        base_peak = _finite(row.get("base_peak_intensity"))
        precursor = precursor if precursor is not None and precursor > 0 else None
        base_peak = base_peak if base_peak is not None and base_peak >= 0 else None
        features += [(precursor or 0.0) / 1000.0, float(precursor is not None),
                     math.log1p(base_peak or 0.0), float(base_peak is not None),
                     math.log1p(max(num_peaks, 0))]
        features += adduct_mass_features(row.get("adduct"), precursor)
        original = row.get("collision_energy_orig")
        original_values = _original_energies(original)
        for values, capacity in ((_energies(row.get("collision_energy_ev")), self.energy_slots),
                                 (original_values, self.original_energy_slots)):
            if len(values) > capacity:
                raise ValueError("collision energy list exceeds fitted schema; refusing truncation")
            features.append(float(len(values)))
            for index in range(capacity):
                features += [values[index] / 100.0, 1.0] if index < len(values) else [0.0, 0.0]
        original_present = _category(original) is not None
        features += [float(original_present), float(original_present and not original_values)]
        return torch.tensor(features, dtype=torch.float32)
