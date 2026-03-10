"""Unit 1 -- Data Ingestion for GSEA preranked output files."""

from pathlib import Path
from dataclasses import dataclass, field
import csv
import re
import sys


@dataclass
class TermRecord:
    """Enrichment data for one GO term in one mutant."""
    term_name: str
    go_id: str
    nes: float
    fdr: float
    nom_pval: float
    size: int


@dataclass
class MutantProfile:
    """Complete enrichment profile for one mutant (merged pos + neg)."""
    mutant_id: str
    records: dict[str, TermRecord]


@dataclass
class CohortData:
    """All enrichment data for the entire mutant cohort."""
    mutant_ids: list[str]  # sorted alphabetically
    profiles: dict[str, MutantProfile]  # keyed by mutant_id
    all_term_names: set[str]  # union of all GO term names across all mutants
    all_go_ids: set[str]  # union of all GO IDs across all mutants


class DataIngestionError(Exception):
    """Raised when input data violates structural expectations."""
    pass


# Regex pattern to match a GO ID: GO: followed by exactly 7 digits
_GO_ID_PATTERN = re.compile(r"GO:\d{7}")


def discover_mutant_folders(data_dir: Path) -> list[tuple[str, Path]]:
    """Discover level-1 mutant subfolders and extract mutant identifiers.

    Returns list of (mutant_id, folder_path) sorted alphabetically by mutant_id.
    """
    results = []
    for entry in data_dir.iterdir():
        if not entry.is_dir():
            continue
        folder_name = entry.name
        if ".GseaPreranked" not in folder_name:
            continue
        # Extract mutant_id as the portion before the first .GseaPreranked
        mutant_id = folder_name.split(".GseaPreranked")[0]
        results.append((mutant_id, entry))
    # Sort alphabetically by mutant_id
    results.sort(key=lambda x: x[0])
    return results


def locate_report_files(mutant_folder: Path, mutant_id: str) -> tuple[Path, Path]:
    """Locate exactly one pos and one neg TSV file in a mutant subfolder.

    Returns (pos_file_path, neg_file_path).
    Raises DataIngestionError if zero or more than one match for either pattern.
    """
    pos_files = list(mutant_folder.glob("gsea_report_for_na_pos_*.tsv"))
    neg_files = list(mutant_folder.glob("gsea_report_for_na_neg_*.tsv"))

    if len(pos_files) == 0:
        raise DataIngestionError(
            f"Missing positive report file for mutant '{mutant_id}': "
            f"no files matching 'gsea_report_for_na_pos_*.tsv' in {mutant_folder}"
        )
    if len(pos_files) > 1:
        raise DataIngestionError(
            f"Ambiguous positive report file for mutant '{mutant_id}': "
            f"found {len(pos_files)} files matching 'gsea_report_for_na_pos_*.tsv' in {mutant_folder}"
        )

    if len(neg_files) == 0:
        raise DataIngestionError(
            f"Missing negative report file for mutant '{mutant_id}': "
            f"no files matching 'gsea_report_for_na_neg_*.tsv' in {mutant_folder}"
        )
    if len(neg_files) > 1:
        raise DataIngestionError(
            f"Ambiguous negative report file for mutant '{mutant_id}': "
            f"found {len(neg_files)} files matching 'gsea_report_for_na_neg_*.tsv' in {mutant_folder}"
        )

    return (pos_files[0], neg_files[0])


def parse_gsea_report(tsv_path: Path) -> list[TermRecord]:
    """Parse a single GSEA preranked TSV report file.

    Extracts GO ID and term name from NAME column. Handles HTML artifact in
    column headers and trailing tabs. Skips rows without valid GO ID with warning.
    """
    records = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    if not lines:
        return records

    # Parse header to find column indices
    header_line = lines[0].rstrip("\t")
    headers = header_line.split("\t")
    # Normalize headers: strip whitespace and handle HTML artifacts
    normalized_headers = [h.strip() for h in headers]

    # Build a map of column name -> index
    # We need: NAME, NES, FDR q-val, NOM p-val, SIZE
    col_map = {}
    for i, h in enumerate(normalized_headers):
        col_map[h] = i

    name_idx = col_map.get("NAME")
    nes_idx = col_map.get("NES")
    fdr_idx = col_map.get("FDR q-val")
    nom_pval_idx = col_map.get("NOM p-val")
    size_idx = col_map.get("SIZE")

    for line_num, line in enumerate(lines[1:], start=2):
        # Skip empty lines
        if not line.strip():
            continue

        # Strip trailing tabs to avoid spurious empty fields
        stripped_line = line.rstrip("\t")
        fields = stripped_line.split("\t")

        # Get the NAME field
        if name_idx is None or name_idx >= len(fields):
            print(
                f"Warning: skipping row {line_num} in {tsv_path}: missing NAME field",
                file=sys.stderr,
            )
            continue

        name_value = fields[name_idx].strip()

        # Extract GO ID using regex
        go_match = _GO_ID_PATTERN.search(name_value)
        if not go_match:
            print(
                f"Warning: skipping row {line_num} in {tsv_path}: "
                f"no valid GO ID found in NAME '{name_value}'",
                file=sys.stderr,
            )
            continue

        go_id = go_match.group(0)
        # Term name is everything after the GO ID, stripped, normalized to uppercase
        term_name = name_value[go_match.end():].strip().upper()

        if not term_name:
            print(
                f"Warning: skipping row {line_num} in {tsv_path}: "
                f"empty term name after GO ID in NAME '{name_value}'",
                file=sys.stderr,
            )
            continue

        # Extract numeric fields
        try:
            nes_val = float(fields[nes_idx]) if nes_idx is not None and nes_idx < len(fields) else None
            fdr_val = float(fields[fdr_idx]) if fdr_idx is not None and fdr_idx < len(fields) else None
            nom_pval_val = float(fields[nom_pval_idx]) if nom_pval_idx is not None and nom_pval_idx < len(fields) else None
            size_val = int(fields[size_idx]) if size_idx is not None and size_idx < len(fields) else None
        except (ValueError, TypeError):
            print(
                f"Warning: skipping row {line_num} in {tsv_path}: "
                f"non-numeric values in NES, FDR q-val, NOM p-val, or SIZE",
                file=sys.stderr,
            )
            continue

        if any(v is None for v in [nes_val, fdr_val, nom_pval_val, size_val]):
            print(
                f"Warning: skipping row {line_num} in {tsv_path}: "
                f"missing values in NES, FDR q-val, NOM p-val, or SIZE",
                file=sys.stderr,
            )
            continue

        record = TermRecord(
            term_name=term_name,
            go_id=go_id,
            nes=nes_val,
            fdr=fdr_val,
            nom_pval=nom_pval_val,
            size=size_val,
        )
        records.append(record)

    return records


def merge_pos_neg(pos_records: list[TermRecord], neg_records: list[TermRecord]) -> dict[str, TermRecord]:
    """Merge positive and negative report records into a single profile dict keyed by term_name.

    If a term appears in both pos and neg records, the entry with the smaller
    nominal p-value is retained (conflict resolution per spec Section 6.2 Step 1).
    """
    merged: dict[str, TermRecord] = {}

    for record in pos_records:
        if record.term_name in merged:
            existing = merged[record.term_name]
            if record.nom_pval < existing.nom_pval:
                merged[record.term_name] = record
        else:
            merged[record.term_name] = record

    for record in neg_records:
        if record.term_name in merged:
            existing = merged[record.term_name]
            if record.nom_pval < existing.nom_pval:
                merged[record.term_name] = record
        else:
            merged[record.term_name] = record

    return merged


def ingest_data(data_dir: Path) -> CohortData:
    """Top-level ingestion entry point. Discovers folders, validates, parses, and merges.

    Raises DataIngestionError on structural violations including fewer than 2 mutant lines.
    """
    if not data_dir.is_dir():
        raise DataIngestionError(f"data_dir must be an existing directory: {data_dir}")

    mutant_folders = discover_mutant_folders(data_dir)

    if len(mutant_folders) < 2:
        raise DataIngestionError(
            f"Insufficient mutant lines: found {len(mutant_folders)} valid mutant "
            f"subfolders in {data_dir}, but at least 2 are required"
        )

    profiles: dict[str, MutantProfile] = {}
    all_term_names: set[str] = set()
    all_go_ids: set[str] = set()
    mutant_ids: list[str] = []

    for mutant_id, folder_path in mutant_folders:
        pos_file, neg_file = locate_report_files(folder_path, mutant_id)

        pos_records = parse_gsea_report(pos_file)
        neg_records = parse_gsea_report(neg_file)

        merged = merge_pos_neg(pos_records, neg_records)

        profile = MutantProfile(mutant_id=mutant_id, records=merged)
        profiles[mutant_id] = profile
        mutant_ids.append(mutant_id)

        for rec in merged.values():
            all_term_names.add(rec.term_name)
            all_go_ids.add(rec.go_id)

    # mutant_ids should already be sorted since discover_mutant_folders returns sorted
    mutant_ids.sort()

    return CohortData(
        mutant_ids=mutant_ids,
        profiles=profiles,
        all_term_names=all_term_names,
        all_go_ids=all_go_ids,
    )
