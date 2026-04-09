"""
src/fastatools/UniProtFastaManager.py

Utilities for downloading FASTA files from UniProt REST API and
filtering existing FASTA files by random subsampling or accession list.

UniProt REST API docs: https://www.uniprot.org/help/api_queries
"""

from __future__ import annotations

import io
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import requests

# -----------------------------------------------------------------------------
# UniProt REST API constants

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# Well-known NCBI taxonomy IDs
COMMON_SPECIES: dict[str, dict] = {
    "Human (Homo sapiens)": {"taxon_id": 9606, "short": "HUMAN"},
    "Mouse (Mus musculus)": {"taxon_id": 10090, "short": "MOUSE"},
    "Rat (Rattus norvegicus)": {"taxon_id": 10116, "short": "RAT"},
    "Yeast (Saccharomyces cerevisiae)": {"taxon_id": 559292, "short": "YEAST"},
    "E. coli (K-12)": {"taxon_id": 83333, "short": "ECOLI"},
    "Arabidopsis thaliana": {"taxon_id": 3702, "short": "ARATH"},
    "Zebrafish (Danio rerio)": {"taxon_id": 7955, "short": "DANRE"},
    "Fruit fly (Drosophila melanogaster)": {"taxon_id": 7227, "short": "DROME"},
    "C. elegans": {"taxon_id": 6239, "short": "CAEEL"},
    "Chicken (Gallus gallus)": {"taxon_id": 9031, "short": "CHICK"},
    "Pig (Sus scrofa)": {"taxon_id": 9823, "short": "PIG"},
    "Bovine (Bos taurus)": {"taxon_id": 9913, "short": "BOVIN"},
    "Rabbit (Oryctolagus cuniculus)": {"taxon_id": 9986, "short": "RABIT"},
    "Mycobacterium tuberculosis H37Rv": {"taxon_id": 83332, "short": "MYCTU"},
    "SARS-CoV-2": {"taxon_id": 2697049, "short": "SARS2"},
}

REVIEW_OPTIONS = {
    "Reviewed (Swiss-Prot)": "true",
    "Unreviewed (TrEMBL)": "false",
    "Both": None,
}

# -----------------------------------------------------------------------------
# Data structures


@dataclass
class DownloadResult:
    success: bool
    fasta_text: str = ""
    n_entries: int = 0
    error: str = ""
    filename: str = ""


@dataclass
class FilterResult:
    success: bool
    fasta_text: str = ""
    n_in: int = 0
    n_out: int = 0
    n_random_added: int = 0
    missing_accs: list[str] = field(default_factory=list)
    error: str = ""
    filename: str = ""


@dataclass
class AppendResult:
    success: bool
    fasta_text: str = ""
    n_base: int = 0
    n_appended: int = 0
    n_total: int = 0
    n_skipped_duplicates: int = 0
    error: str = ""


# -----------------------------------------------------------------------------
# Minimal pure-Python FASTA parser (avoids BioPython dependency in server env)


def _parse_fasta(text: str) -> list[tuple[str, str, str]]:
    """
    Parse FASTA text into a list of (header, accession, sequence) tuples.
    Accession is extracted from the UniProt-style header:
      >sp|P12345|GENE_HUMAN Description …
      >tr|A0A001|…
      >GENERIC_ACC Description …
    """
    records: list[tuple[str, str, str]] = []
    header = ""
    seq_parts: list[str] = []

    for line in text.splitlines():
        if line.startswith(">"):
            if header:
                records.append((header, _acc_from_header(header), "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())

    if header:
        records.append((header, _acc_from_header(header), "".join(seq_parts)))

    return records


def _acc_from_header(header: str) -> str:
    """
    Extract the primary accession from a FASTA header.
    Handles:
      sp|P12345|GENE_HUMAN …    → P12345
      tr|A0A001|…               → A0A001
      P12345 description         → P12345
      generic_id description     → generic_id (first token)
    """
    m = re.match(r"(?:sp|tr|ref)\|([A-Z0-9_.-]+)\|", header)
    if m:
        return m.group(1)
    # Try bare accession at start of header (word chars before whitespace)
    m = re.match(r"([^\s|]+)", header)
    return m.group(1) if m else header.split()[0]


def _records_to_fasta(records: list[tuple[str, str, str]], line_width: int = 60) -> str:
    """Serialise (header, acc, seq) tuples back to FASTA text."""
    parts: list[str] = []
    for header, _acc, seq in records:
        parts.append(f">{header}")
        for i in range(0, len(seq), line_width):
            parts.append(seq[i : i + line_width])
    return "\n".join(parts) + "\n"


# -----------------------------------------------------------------------------
# UniProt Download


def _build_uniprot_query(
    taxon_ids: list[int],
    reviewed: str | None,  # "true", "false", or None for both
    extra_query: str = "",
) -> str:
    """
    Build a UniProt query string.

    Examples:
        (taxonomy_id:9606) AND (reviewed:true)
        (taxonomy_id:9606 OR taxonomy_id:83333) AND (reviewed:true)
    """
    # Taxonomy clause
    if len(taxon_ids) == 1:
        tax_clause = f"(taxonomy_id:{taxon_ids[0]})"
    else:
        inner = " OR ".join(f"taxonomy_id:{t}" for t in taxon_ids)
        tax_clause = f"({inner})"

    parts = [tax_clause]

    if reviewed is not None:
        parts.append(f"(reviewed:{reviewed})")

    if extra_query.strip():
        parts.append(f"({extra_query.strip()})")

    return " AND ".join(parts)


def download_uniprot_fasta(
    species_names: list[str],
    review_option: str = "Reviewed (Swiss-Prot)",
    extra_query: str = "",
    include_isoforms: bool = False,
    chunk_size: int = 500,
    progress_cb=None,  # optional callable(fraction: float, msg: str)
    timeout: int = 60,
) -> DownloadResult:
    """
    Download a FASTA file from the UniProt REST API.

    Args:
        species_names:  Display names from COMMON_SPECIES.
        review_option:  One of REVIEW_OPTIONS keys.
        extra_query:    Additional UniProt query text (e.g. "gene:TP53").
        include_isoforms: Include canonical isoforms.
        chunk_size:     Proteins per streaming chunk (reduces memory).
        progress_cb:    Optional callback(fraction, message) for progress.
        timeout:        HTTP timeout in seconds.

    Returns:
        DownloadResult with fasta_text and n_entries filled in on success.
    """
    # -- Resolve species ----------------------------------------------------
    taxon_ids = []
    for name in species_names:
        info = COMMON_SPECIES.get(name)
        if info:
            taxon_ids.append(info["taxon_id"])
    if not taxon_ids:
        return DownloadResult(success=False, error="No valid species selected.")

    reviewed = REVIEW_OPTIONS.get(review_option)

    query = _build_uniprot_query(taxon_ids, reviewed, extra_query)

    # Build suggested filename
    short_tags = [
        COMMON_SPECIES[n]["short"] for n in species_names if n in COMMON_SPECIES
    ]
    rev_tag = {None: "all", "true": "reviewed", "false": "unreviewed"}[reviewed]
    filename = "_".join(short_tags) + f"_{rev_tag}.fasta"

    # -- Streaming download via /stream endpoint -----------------------------
    params = {
        "query": query,
        "format": "fasta",
        "compressed": "false",
    }
    if include_isoforms:
        params["includeIsoform"] = "true"

    if progress_cb:
        progress_cb(0.0, f"Querying UniProt: {query}")

    try:
        resp = requests.get(UNIPROT_STREAM, params=params, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return DownloadResult(success=False, error=f"UniProt HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        return DownloadResult(success=False, error=f"Network error: {e}")

    # Accumulate streamed response
    chunks: list[str] = []
    n_bytes = 0
    for raw_chunk in resp.iter_content(chunk_size=65536):
        if raw_chunk:
            chunks.append(raw_chunk.decode("utf-8", errors="replace"))
            n_bytes += len(raw_chunk)
            if progress_cb:
                progress_cb(None, f"Downloading… {n_bytes / 1024:.0f} KB received")

    fasta_text = "".join(chunks)
    n_entries = fasta_text.count("\n>") + (1 if fasta_text.startswith(">") else 0)

    if n_entries == 0:
        return DownloadResult(
            success=False,
            error=(
                "No sequences returned. The query may have no results, or "
                "UniProt returned an unexpected format.\n"
                f"Query used: {query}"
            ),
        )

    if progress_cb:
        progress_cb(1.0, f"Download complete — {n_entries:,} entries")

    return DownloadResult(
        success=True,
        fasta_text=fasta_text,
        n_entries=n_entries,
        filename=filename,
    )


# -----------------------------------------------------------------------------
# FASTA Filtering


def filter_fasta_random(
    fasta_text: str,
    n: int,
    seed: int | None = None,
    source_filename: str = "input.fasta",
) -> FilterResult:
    """
    Randomly subsample *n* sequences from *fasta_text*.

    Args:
        fasta_text:       Raw FASTA string.
        n:                Number of sequences to keep.
        seed:             Random seed for reproducibility.
        source_filename:  Used only to build the suggested output filename.
    """
    records = _parse_fasta(fasta_text)
    n_in = len(records)

    if n_in == 0:
        return FilterResult(success=False, error="Input FASTA contains no records.")

    if n > n_in:
        return FilterResult(
            success=False,
            error=f"Requested {n:,} sequences but input only has {n_in:,}.",
        )

    rng = random.Random(seed)
    selected = rng.sample(records, n)

    stem = Path(source_filename).stem
    filename = f"{stem}_random{n}.fasta"

    return FilterResult(
        success=True,
        fasta_text=_records_to_fasta(selected),
        n_in=n_in,
        n_out=len(selected),
        filename=filename,
    )


def filter_fasta_by_accession(
    fasta_text: str,
    accessions: list[str],
    source_filename: str = "input.fasta",
    extra_random_n: int = 0,
    seed: int | None = None,
) -> FilterResult:
    """
    Keep only sequences whose accession is in *accessions*.

    Args:
        fasta_text:      Raw FASTA string.
        accessions:      List of accession IDs to retain.
        source_filename: Used to build the suggested output filename.
        extra_random_n:  Number of additional random proteins to append from
                         the non-matching pool.
        seed:            Random seed for reproducible random augmentation.
    """
    records = _parse_fasta(fasta_text)
    n_in = len(records)

    if n_in == 0:
        return FilterResult(success=False, error="Input FASTA contains no records.")

    query_set = {a.strip() for a in accessions if a.strip()}
    if not query_set:
        return FilterResult(success=False, error="No accessions provided.")

    matched: list[tuple] = []
    unmatched: list[tuple] = []
    found_accs: set[str] = set()

    for rec in records:
        acc = rec[1]
        if acc in query_set:
            matched.append(rec)
            found_accs.add(acc)
        else:
            unmatched.append(rec)

    missing = sorted(query_set - found_accs)

    stem = Path(source_filename).stem
    n_acc = len(accessions)

    if not matched:
        return FilterResult(
            success=False,
            n_in=n_in,
            missing_accs=missing,
            error=f"None of the {n_acc} requested accessions were found in the FASTA.",
        )

    random_added: list[tuple[str, str, str]] = []
    if extra_random_n < 0:
        return FilterResult(
            success=False,
            n_in=n_in,
            missing_accs=missing,
            error="The number of extra random proteins must be 0 or greater.",
        )

    if extra_random_n:
        if extra_random_n > len(unmatched):
            return FilterResult(
                success=False,
                n_in=n_in,
                n_out=len(matched),
                missing_accs=missing,
                error=(
                    f"Requested {extra_random_n:,} extra random proteins, but only "
                    f"{len(unmatched):,} non-matching proteins are available."
                ),
            )
        rng = random.Random(seed)
        random_added = rng.sample(unmatched, extra_random_n)

    selected = matched + random_added
    filename = f"{stem}_filtered{len(matched)}"
    if random_added:
        filename += f"_plusrandom{len(random_added)}"
    filename += ".fasta"

    return FilterResult(
        success=True,
        fasta_text=_records_to_fasta(selected),
        n_in=n_in,
        n_out=len(selected),
        n_random_added=len(random_added),
        missing_accs=missing,
        filename=filename,
    )


def append_fasta_records(base_fasta_text: str, extra_fasta_text: str) -> AppendResult:
    """
    Append FASTA records from *extra_fasta_text* onto *base_fasta_text* while
    skipping exact-header duplicates. The output is reserialised so the final
    FASTA is properly formatted.
    """
    base_records = _parse_fasta(base_fasta_text)
    extra_records = _parse_fasta(extra_fasta_text)

    if not base_records:
        return AppendResult(success=False, error="Base FASTA contains no records.")
    if not extra_records:
        return AppendResult(success=False, error="Append FASTA contains no records.")

    seen_headers = {header for header, _acc, _seq in base_records}
    appended: list[tuple[str, str, str]] = []
    skipped = 0

    for record in extra_records:
        header = record[0]
        if header in seen_headers:
            skipped += 1
            continue
        seen_headers.add(header)
        appended.append(record)

    combined = base_records + appended
    return AppendResult(
        success=True,
        fasta_text=_records_to_fasta(combined),
        n_base=len(base_records),
        n_appended=len(appended),
        n_total=len(combined),
        n_skipped_duplicates=skipped,
    )


def count_fasta_entries(fasta_text: str) -> int:
    """Fast entry count without full parsing."""
    return sum(1 for line in fasta_text.splitlines() if line.startswith(">"))
