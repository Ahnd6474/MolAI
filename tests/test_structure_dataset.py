import gzip
from pathlib import Path

from molai.dft.dataset import StructureRecord, iter_structure_records


def test_streams_smi_and_optionally_deduplicates_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "molecules.smi"
    path.write_text("O water\nCCO ethanol\nN water\n", encoding="utf-8")

    deduplicated = list(iter_structure_records(path))
    all_records = list(iter_structure_records(path, deduplicate_ids=False))

    assert deduplicated == [
        StructureRecord("water", "O"),
        StructureRecord("ethanol", "CCO"),
    ]
    assert len(all_records) == 3


def test_streams_gzipped_csv(tmp_path: Path) -> None:
    path = tmp_path / "molecules.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,smiles\n1,O\n2,c1ccccc1\n")

    records = list(iter_structure_records(path, "smiles", "id"))

    assert records == [StructureRecord("1", "O"), StructureRecord("2", "c1ccccc1")]
