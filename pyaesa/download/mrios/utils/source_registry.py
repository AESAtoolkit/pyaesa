"""Canonical MRIO source registry shared across download and processing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MRIORegistryEntry:
    """Static contract for one supported MRIO source."""

    source_key: str
    family: str
    version_token: str
    system: str | None
    doi: str | None
    raw_root: str
    raw_full_dir_name: str
    shared_prereq_root: str
    region_code_column: str
    modeled_year_min: int
    modeled_year_max: int
    default_historical_cutoff: int
    display_label: str


_EXIOBASE_396_DOI = "10.5281/zenodo.15689391"
_EXIOBASE_3102_DOI = "10.5281/zenodo.20051562"

_REGISTRY = {
    "exiobase_396_ixi": MRIORegistryEntry(
        source_key="exiobase_396_ixi",
        family="exiobase",
        version_token="396",
        system="ixi",
        doi=_EXIOBASE_396_DOI,
        raw_root="exiobase_396",
        raw_full_dir_name="full_ixi",
        shared_prereq_root="exiobase_3",
        region_code_column="exio_code",
        modeled_year_min=1995,
        modeled_year_max=2022,
        default_historical_cutoff=2019,
        display_label="EXIOBASE 3.9.6 IXI",
    ),
    "exiobase_396_pxp": MRIORegistryEntry(
        source_key="exiobase_396_pxp",
        family="exiobase",
        version_token="396",
        system="pxp",
        doi=_EXIOBASE_396_DOI,
        raw_root="exiobase_396",
        raw_full_dir_name="full_pxp",
        shared_prereq_root="exiobase_3",
        region_code_column="exio_code",
        modeled_year_min=1995,
        modeled_year_max=2022,
        default_historical_cutoff=2019,
        display_label="EXIOBASE 3.9.6 PXP",
    ),
    "exiobase_3102_ixi": MRIORegistryEntry(
        source_key="exiobase_3102_ixi",
        family="exiobase",
        version_token="3102",
        system="ixi",
        doi=_EXIOBASE_3102_DOI,
        raw_root="exiobase_3102",
        raw_full_dir_name="full_ixi",
        shared_prereq_root="exiobase_3",
        region_code_column="exio_code",
        modeled_year_min=1995,
        modeled_year_max=2024,
        default_historical_cutoff=2022,
        display_label="EXIOBASE 3.10.2 IXI",
    ),
    "exiobase_3102_pxp": MRIORegistryEntry(
        source_key="exiobase_3102_pxp",
        family="exiobase",
        version_token="3102",
        system="pxp",
        doi=_EXIOBASE_3102_DOI,
        raw_root="exiobase_3102",
        raw_full_dir_name="full_pxp",
        shared_prereq_root="exiobase_3",
        region_code_column="exio_code",
        modeled_year_min=1995,
        modeled_year_max=2024,
        default_historical_cutoff=2022,
        display_label="EXIOBASE 3.10.2 PXP",
    ),
    "oecd_v2025": MRIORegistryEntry(
        source_key="oecd_v2025",
        family="oecd",
        version_token="v2025",
        system=None,
        doi=None,
        raw_root="oecd_v2025",
        raw_full_dir_name="full",
        shared_prereq_root="oecd_v2025",
        region_code_column="oecd_code",
        modeled_year_min=1995,
        modeled_year_max=2022,
        default_historical_cutoff=2022,
        display_label="OECD ICIO v2025",
    ),
}


def normalize_mrio_source_key(source_key: str) -> str:
    """Return the normalized MRIO source key and validate support."""
    key = str(source_key).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported MRIO source '{source_key}'. Supported sources: {sorted(_REGISTRY)}"
        )
    return key


def get_mrio_entry(source_key: str) -> MRIORegistryEntry:
    """Return one source registry entry."""
    return _REGISTRY[normalize_mrio_source_key(source_key)]


def list_mrio_source_keys() -> tuple[str, ...]:
    """Return all supported MRIO source keys."""
    return tuple(_REGISTRY)


def iter_mrio_entries() -> tuple[MRIORegistryEntry, ...]:
    """Return all registry entries in deterministic insertion order."""
    return tuple(_REGISTRY.values())


def is_exio_mrio_source(source_key: str) -> bool:
    """Return whether a MRIO source belongs to the EXIO family."""
    return get_mrio_entry(source_key).family == "exiobase"


def default_years_for_source(source_key: str) -> list[int]:
    """Return the default modeled year coverage for one MRIO source."""
    entry = get_mrio_entry(source_key)
    return list(range(entry.modeled_year_min, entry.modeled_year_max + 1))
