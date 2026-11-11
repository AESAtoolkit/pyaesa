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


_REGISTRY = {
    f"exiobase_{version}_{system}": MRIORegistryEntry(
        source_key=f"exiobase_{version}_{system}",
        family="exiobase",
        version_token=version,
        system=system,
        doi=doi,
        raw_root=f"exiobase_{version}",
        raw_full_dir_name=f"full_{system}",
        shared_prereq_root="exiobase_3",
        region_code_column="exio_code",
        modeled_year_min=1995,
        modeled_year_max=last_year,
        default_historical_cutoff=historical_cutoff,
        display_label=f"EXIOBASE {label} {system.upper()}",
    )
    for version, label, doi, last_year, historical_cutoff in (
        ("3112", "3.11.2", "10.5281/zenodo.23039074", 2024, 2023),
        ("3102", "3.10.2", "10.5281/zenodo.20051562", 2024, 2022),
        ("396", "3.9.6", "10.5281/zenodo.15689391", 2022, 2019),
    )
    for system in ("ixi", "pxp")
}
_REGISTRY["oecd_v2025"] = MRIORegistryEntry(
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
)


def normalize_mrio_source_key(source_key: str) -> str:
    """Return the normalized MRIO source key and validate support."""
    key = str(source_key).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported MRIO source '{source_key}'. Supported sources: {list(_REGISTRY)}"
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
