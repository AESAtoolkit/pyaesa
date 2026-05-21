"""LCIA uncertainty CoV asset loading."""

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from pyaesa.shared.lcia.paths import carbon_account_cov_path
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE


@dataclass(frozen=True)
class LCIACoVInputs:
    """Resolved LCIA CoV inputs used by uncertainty sources."""

    country_covs: dict[str, float]
    sector_covs: dict[str, float]
    world_cov: float
    sector_cov_mapping: dict[str, str]


def load_lcia_cov_inputs(
    *,
    sector_cov_mapping: dict[str, str],
    group_reg: bool = False,
    group_version: str | None = None,
    aggregate_region_covs: bool = False,
) -> LCIACoVInputs:
    """Load country, world, and sector CoV values for LCIA sampling."""
    country_covs, world_cov = _load_country_covs(
        group_reg=group_reg,
        group_version=group_version,
        aggregate_region_covs=aggregate_region_covs,
    )
    sector_covs = _load_sector_covs()
    return LCIACoVInputs(
        country_covs=country_covs,
        sector_covs=sector_covs,
        world_cov=world_cov,
        sector_cov_mapping=dict(sector_cov_mapping),
    )


def normalize_lcia_uncertainty_parameters(*, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized public LCIA uncertainty source parameters."""
    payload = dict(parameters)
    unknown = sorted(set(payload) - {"sector_cov_mapping"})
    if unknown:
        raise ValueError(f"Unsupported {LCIA_SOURCE} parameter keys: {unknown}.")
    return {
        "sector_cov_mapping": normalize_lcia_sector_cov_mapping(
            payload.get("sector_cov_mapping", {})
        )
    }


def normalize_lcia_sector_cov_mapping(value: object) -> dict[str, str]:
    """Return a validated output s_p label to sector CoV code mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{LCIA_SOURCE}.sector_cov_mapping must be a mapping.")
    mapping: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _non_empty_text(raw_key, field=f"{LCIA_SOURCE}.sector_cov_mapping key")
        mapping[key] = _non_empty_text(
            raw_value,
            field=f"{LCIA_SOURCE}.sector_cov_mapping[{key!r}]",
        )
    return mapping


def country_cov_values(*, covs: LCIACoVInputs, country_key: pd.Series) -> pd.Series:
    """Return validated country CoV values for LCIA uncertainty rows."""
    keys = country_key.astype(str)
    values = keys.map(covs.country_covs)
    missing = sorted(set(keys.loc[values.isna()].tolist()))
    if missing:
        raise ValueError(f"LCIA country CoV is not available for country labels: {missing}.")
    return values.astype("float64")


def sector_cov_keys(*, covs: LCIACoVInputs, sector_label: pd.Series) -> pd.Series:
    """Return validated sector CoV codes for output s_p labels."""
    labels = sector_label.astype(str)
    keys = labels.map(covs.sector_cov_mapping)
    missing = sorted(set(labels.loc[keys.isna()].tolist()))
    if missing:
        raise ValueError(
            f"{LCIA_SOURCE}.sector_cov_mapping is missing targeted sector labels: {missing}."
        )
    return keys.astype(str)


def sector_cov_values(*, covs: LCIACoVInputs, sector_key: pd.Series) -> pd.Series:
    """Return validated sector CoV values for LCIA uncertainty rows."""
    keys = sector_key.astype(str)
    values = keys.map(covs.sector_covs)
    missing = sorted(set(keys.loc[values.isna()].tolist()))
    if missing:
        raise ValueError(f"LCIA sector CoV is not available for sector codes: {missing}.")
    return values.astype("float64")


def _load_country_covs(
    *,
    group_reg: bool,
    group_version: str | None,
    aggregate_region_covs: bool,
) -> tuple[dict[str, float], float]:
    asset_name, label = _country_cov_asset(
        group_reg=group_reg,
        group_version=group_version,
        aggregate_region_covs=aggregate_region_covs,
    )
    path = carbon_account_cov_path(asset_name=asset_name)
    if (group_reg or aggregate_region_covs) and not path.exists():
        raise ValueError(f"{label} LCIA country CoV file is missing: {path}.")
    frame = pd.read_csv(path)
    missing_columns = sorted({"exio_code", "cov"} - set(frame.columns))
    if missing_columns:
        raise ValueError(f"LCIA country CoV file {path} is missing columns: {missing_columns}.")
    out: dict[str, float] = {}
    for raw_code, raw_cov in frame.loc[:, ["exio_code", "cov"]].itertuples(
        index=False,
        name=None,
    ):
        code = _non_empty_text(raw_code, field="exio_code")
        if code in out:
            raise ValueError(f"LCIA country CoV file {path} has duplicate label: {code}.")
        try:
            cov = float(str(raw_cov))
        except ValueError as exc:
            raise ValueError(f"LCIA country CoV file {path} has non numeric CoV: {code}.") from exc
        if pd.isna(cov):
            raise ValueError(f"LCIA country CoV file {path} has non numeric CoV: {code}.")
        out[code] = cov
    if "World" not in out:
        raise ValueError(f"LCIA country CoV file {path} must include 'World'.")
    return out, out["World"]


def _country_cov_asset(
    *,
    group_reg: bool,
    group_version: str | None,
    aggregate_region_covs: bool,
) -> tuple[str, str]:
    suffix = "_aggreg_indices" if aggregate_region_covs else ""
    if group_reg:
        version = _non_empty_text(group_version, field="group_version")
        return f"reg_cbca_covs_group_{version}{suffix}.csv", "Grouped region"
    if aggregate_region_covs:
        return "reg_cbca_covs_aggreg_indices.csv", "Aggregate region"
    return "reg_cbca_covs.csv", "Region"


def _load_sector_covs() -> dict[str, float]:
    path = carbon_account_cov_path(asset_name="sec_cbca_covs.csv")
    frame = pd.read_csv(path)
    return {
        str(raw_code).strip(): float(str(raw_cov))
        for raw_code, raw_cov in frame.loc[:, ["sector_code", " cov median "]].itertuples(
            index=False,
            name=None,
        )
    }


def _non_empty_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text or text in {"None", "nan", "<NA>"}:
        raise ValueError(f"{field} must be a non empty string.")
    return text
