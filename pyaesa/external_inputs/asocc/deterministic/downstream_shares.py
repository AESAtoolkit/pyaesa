"""Deterministic downstream loading of external aSoCC share tables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

from pyaesa.asocc.runtime.scope.context_rebuild import resolve_external_ssp_scenario_options_by_year
from pyaesa.asocc.runtime.paths.external import external_asocc_relative_dir

from pyaesa.external_inputs.asocc.schema.contracts import iter_external_method_selections
from pyaesa.external_inputs.asocc.deterministic.files import (
    describe_expected_external_deterministic_stems,
    expected_external_deterministic_stems,
    load_external_deterministic_rows,
)
from pyaesa.external_inputs.asocc.schema.file_specs import external_asocc_runtime_file_stem
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN


@dataclass(frozen=True)
class ExternalAsoccShare:
    """One resolved deterministic external aSoCC share table."""

    asocc_method_label: str
    level: str
    relative_dir: Path
    file_stem: str
    impacts: tuple[str, ...]
    frame_wide: pd.DataFrame


def _wide_frame(
    *,
    frame: pd.DataFrame,
    asocc_method_label: str,
    scenario: str | None,
) -> pd.DataFrame:
    out = frame.copy()
    out["l1_l2_method"] = str(asocc_method_label)
    out["year"] = out["year"].astype(int).astype(str)
    drop_columns = {"year", "value", "level"}
    id_columns = [column for column in out.columns if column not in drop_columns]
    duplicates = out.duplicated(subset=[*id_columns, "year"], keep=False)
    if bool(duplicates.any()):
        sample = out.loc[duplicates, [*id_columns, "year"]].head(5).to_dict("records")
        raise ValueError(
            "External aSoCC deterministic rows cannot be converted to one deterministic "
            "wide share table because duplicate identifier/year rows were found. "
            f"method='{asocc_method_label}', asocc_ssp_scenario='{scenario}', sample={sample}."
        )
    wide = out.pivot(index=id_columns, columns="year", values="value").reset_index()
    year_columns = sorted(
        [column for column in wide.columns if str(column).isdigit()],
        key=lambda value: int(str(value)),
    )
    other_columns = [column for column in wide.columns if column not in year_columns]
    return wide.loc[:, [*other_columns, *year_columns]].reset_index(drop=True)


def _scenario_slices(frame: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    """Return deterministic historical and SSP slices from normalized external rows."""
    if ASOCC_SSP_SCENARIO_COLUMN not in frame.columns:
        return [(None, frame.copy())]
    scenario_series = (
        pd.Series(frame.loc[:, ASOCC_SSP_SCENARIO_COLUMN], copy=False).astype("string").str.strip()
    )
    if bool(scenario_series.dropna().eq("").any()):
        raise ValueError(
            "Deterministic external aSoCC rows must not use blank 'asocc_ssp_scenario' values."
        )
    slices: list[tuple[str | None, pd.DataFrame]] = []
    historical = frame.loc[scenario_series.isna()].drop(columns=[ASOCC_SSP_SCENARIO_COLUMN]).copy()
    if not historical.empty:
        slices.append((None, historical))
    for scenario in sorted({str(value) for value in scenario_series.dropna().tolist()}):
        subset = (
            frame.loc[scenario_series.eq(str(scenario))]
            .drop(columns=[ASOCC_SSP_SCENARIO_COLUMN])
            .copy()
        )
        slices.append((str(scenario), subset))
    return slices


def _asocc_shares_for_frame(
    *,
    frame: pd.DataFrame,
    fu_code: str,
    asocc_method_label: str,
    level: str,
    lcia_method: str | None,
    file_method_token: str,
    l1_method: str | None,
) -> list[ExternalAsoccShare]:
    asocc_shares: list[ExternalAsoccShare] = []
    for scenario, scenario_subset in _scenario_slices(frame):
        impact_series = (
            pd.Series(dtype=object)
            if "impact" not in scenario_subset.columns
            else scenario_subset["impact"]
        )
        normalized_impacts = impact_series.astype(str).str.strip()
        impact_values = sorted(
            {
                str(value).strip()
                for value in impact_series.dropna().astype(str).tolist()
                if str(value).strip()
            }
        )
        if not impact_values:
            impact_values = [""]
        base_stem = external_asocc_runtime_file_stem(
            fu_code=fu_code,
            file_method_token=file_method_token,
            l1_method=l1_method,
            lcia_method=lcia_method,
            scenario=scenario,
        )
        base_stem = f"external__{base_stem}"
        for impact in impact_values:
            subset = scenario_subset.copy()
            if impact:
                subset = subset.loc[normalized_impacts.eq(impact)].copy()
            asocc_shares.append(
                ExternalAsoccShare(
                    asocc_method_label=asocc_method_label,
                    level=level,
                    relative_dir=external_asocc_relative_dir(level=level),
                    file_stem=base_stem,
                    impacts=tuple([impact] if impact else []),
                    frame_wide=_wide_frame(
                        frame=subset,
                        asocc_method_label=asocc_method_label,
                        scenario=scenario,
                    ),
                )
            )
    return asocc_shares


def load_external_asocc_shares(
    *,
    proj_base: Path,
    fu_code: str,
    external_method: dict[str, Any] | None,
    years: list[int],
    lcia_method: str | None,
    base_allocate_args: dict[str, Any],
    output_source_label: str,
) -> list[ExternalAsoccShare]:
    """Load resolved deterministic external aSoCC share tables for downstream aCC/ASR."""
    if external_method is None:
        return []
    requested_lcia_method = None if lcia_method is None else str(lcia_method)
    ssp_scenario_options_by_year = resolve_external_ssp_scenario_options_by_year(
        base_allocate_args=base_allocate_args,
        years=years,
        output_source_label=str(output_source_label),
    )
    asocc_shares: list[ExternalAsoccShare] = []
    for selection in iter_external_method_selections(
        external_method=external_method,
        fu_code=fu_code,
    ):
        frame = load_external_deterministic_rows(
            proj_base=proj_base,
            selection=selection,
            years=years,
            lcia_methods=None if requested_lcia_method is None else [requested_lcia_method],
            ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        )
        if frame is None:
            expected = expected_external_deterministic_stems(
                selection=selection,
                lcia_methods=None if requested_lcia_method is None else [requested_lcia_method],
                years=years,
                ssp_scenario_options_by_year=ssp_scenario_options_by_year,
            )
            expected_message = describe_expected_external_deterministic_stems(
                proj_base=proj_base,
                selection=selection,
                stems=expected,
            )
            raise ValueError(
                f"Missing deterministic external aSoCC file for '{selection.user_label}'. "
                f"{expected_message}."
            )
        asocc_shares.extend(
            _asocc_shares_for_frame(
                frame=frame,
                fu_code=selection.fu_code,
                asocc_method_label=selection.asocc_method_label,
                level=selection.level,
                lcia_method=requested_lcia_method,
                file_method_token=selection.file_method_token,
                l1_method=selection.l1_method,
            )
        )
    return asocc_shares
