"""Dynamic AR6 carrying capacity aCC branch execution."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.deterministic.request.contracts import cc_variable
from pyaesa.ar6_cc.deterministic.io.paths import get_cc_output_path, get_cc_scope_dir
from pyaesa.ar6_cc.deterministic.io.tables import read_cc_output
from pyaesa.ar6_cc.deterministic.io.tables import filter_to_denominator_cc_rows
from pyaesa.shared.acc_asr_common.deterministic.downstream.inputs import (
    LoadedAsoccShare,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.scenarios import (
    asocc_share_ssp_scenario_labels,
    share_transition_payload_for_output_stem,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.selection import (
    dynamic_compatible_share_frame,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.tabular_io import requested_year_columns
from pyaesa.shared.acc_asr_common.deterministic.status_labels import cc_branch_status_label
from pyaesa.process.ar6.utils.pipeline import runtime_helpers as ar6_runtime_helpers
from pyaesa.process.ar6.utils.pipeline import study_period as ar6_study_period
from pyaesa.acc.shared.runtime.dynamic_units import dynamic_acc_unit_factors
from pyaesa.shared.lcia.contracts import dynamic_cc_match
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.selectors.scenarios import normalize_ssp_token, normalize_ssp_tokens
from pyaesa.acc.shared.runtime.paths import build_acc_output_stem

from pyaesa.acc.deterministic.runtime.paths import (
    ACCDeterministicPathContext,
    acc_output_relative_dir,
    get_acc_output_dir,
)
from .tables import (
    materialize_acc_scope,
    resolve_acc_l1_l2_method,
    write_acc_output,
)


def _should_emit_dynamic_status(*, step: int, total_steps: int) -> bool:
    """Return whether one dynamic ACC loop step should refresh the status line."""
    return step == 1 or step == total_steps or step % 25 == 0


def _emit_dynamic_status(
    *,
    status: StatusSink,
    step: int,
    total_steps: int,
    display_name: str,
    category: str,
    ssp: str,
    cc_model: str,
    cc_label: str,
) -> None:
    """Emit one throttled dynamic ACC status line when the step is selected."""
    if not _should_emit_dynamic_status(step=step, total_steps=total_steps):
        return
    status.show(
        f"[deterministic_acc] {cc_label}: "
        f"{step}/{total_steps} "
        f"{display_name} {category}/{ssp}/{cc_model}"
    )


def _validate_dynamic_share_ssp_alignment(
    *,
    prepared_asocc_shares: list[tuple[LoadedAsoccShare, pd.DataFrame]],
    cc_table: pd.DataFrame,
    require_selected_ssp_shares: bool,
    cc_source: str,
    resolved_cc_path: Path,
    requested_years: list[int],
) -> None:
    """Fail when scoped deterministic shares do not match dynamic AR6 SSP scope."""
    cc_tokens = _resolved_dynamic_cc_ssp_tokens(cc_table=cc_table)
    share_tokens = sorted(
        {
            token
            for asocc_share, share_frame in prepared_asocc_shares
            for token in asocc_share_ssp_scenario_labels(
                asocc_share,
                frame_wide=share_frame,
            )
        }
    )
    if not share_tokens:
        if require_selected_ssp_shares and cc_tokens:
            share_names = sorted(
                {asocc_share.display_name for asocc_share, _ in prepared_asocc_shares}
            )
            raise ValueError(
                "Dynamic AR6 aCC requires SSP dependent aSoCC shares that match the "
                f"selected dynamic AR6 SSP scope {cc_tokens}, but none remained after "
                "dynamic SSP filtering. "
                f"cc_source='{cc_source}', requested years={requested_years}, "
                f"CC table={resolved_cc_path}, aSoCC shares={share_names}."
            )
        return
    if sorted(cc_tokens) != sorted(share_tokens):
        share_names = sorted({asocc_share.display_name for asocc_share, _ in prepared_asocc_shares})
        raise ValueError(
            "Dynamic AR6 aCC requires SSP dependent aSoCC shares to use "
            "the same SSP set as the selected dynamic AR6 scope. "
            f"aSoCC SSPs={share_tokens}, dynamic AR6 SSPs={cc_tokens}, "
            f"cc_source='{cc_source}', requested years={requested_years}, "
            f"CC table={resolved_cc_path}, aSoCC shares={share_names}."
        )


def _resolved_dynamic_cc_ssp_tokens(*, cc_table: pd.DataFrame) -> list[str]:
    """Return the canonical SSP tokens represented by one resolved dynamic CC table."""
    cc_series = cast(pd.Series, cc_table["ssp_scenario"])
    return normalize_ssp_tokens(cc_series.tolist())


def resolve_dynamic_cc_input(
    *,
    years,
    harmonization: bool,
    harmonization_method: str,
    category: list[str] | None,
    ssp_scenario: list[str] | None,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    subset_version: str | None,
    fmt: str,
) -> tuple[Path, pd.DataFrame]:
    """Resolve and load the canonical dynamic AR6 CC table for one aCC branch."""
    study_period = ar6_study_period.resolve_study_period(years)
    harm_method = ar6_runtime_helpers.validate_harmonization_method(
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    )
    scope_dir = get_cc_scope_dir(
        study_period=study_period,
        harmonization=harmonization,
        harmonization_method=harm_method,
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
        subset_version=subset_version,
        category=category,
        ssp_scenario=ssp_scenario,
    )
    cc_path = get_cc_output_path(cc_dir=scope_dir, output_format=fmt)
    cc_table = read_cc_output(output_file=cc_path, output_format=fmt)
    cc_table = filter_to_denominator_cc_rows(cc_table)
    return cc_path, _filter_dynamic_cc_subset(
        cc_table=cc_table,
        category=category,
        ssp_scenario=ssp_scenario,
    )


def dynamic_cc_coverage(*, cc_table: pd.DataFrame) -> dict[str, list[str]]:
    """Return deterministic dynamic CC coverage axes represented by one table."""
    return {
        "cc_category": _non_empty_unique_text_values(cc_table, "cc_category"),
        AR6_CC_SSP_SCENARIO_COLUMN: _resolved_dynamic_cc_ssp_tokens(cc_table=cc_table),
    }


def _filter_dynamic_share_ssp_scope(
    *,
    prepared_asocc_shares: list[tuple[LoadedAsoccShare, pd.DataFrame]],
    cc_table: pd.DataFrame,
) -> tuple[list[tuple[LoadedAsoccShare, pd.DataFrame]], bool]:
    """Return the dynamic-SSP scoped share subset for one dynamic ACC run."""
    cc_tokens = _resolved_dynamic_cc_ssp_tokens(cc_table=cc_table)
    filtered: list[tuple[LoadedAsoccShare, pd.DataFrame]] = []
    has_ssp_dependent_shares = False
    for asocc_share, share_frame in prepared_asocc_shares:
        share_tokens = asocc_share_ssp_scenario_labels(
            asocc_share,
            frame_wide=share_frame,
        )
        if share_tokens:
            has_ssp_dependent_shares = True
        if share_tokens and not share_tokens.issubset(set(cc_tokens)):
            continue
        filtered.append((asocc_share, share_frame))
    return filtered, has_ssp_dependent_shares


def process_dynamic_acc(
    *,
    path_context: ACCDeterministicPathContext,
    public_result_root_name: str | None,
    cc_source: str,
    asocc_shares: list[LoadedAsoccShare],
    fmt: str,
    lcia_method: str | None,
    years,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    share_transition_meta: dict[str, dict[str, object]],
    status: StatusSink,
    resolved_cc_path: Path,
    resolved_cc_table: pd.DataFrame,
) -> tuple[int, int, list[str], list[Path], list[Path], Path]:
    """Process dynamic AR6 aCC for all share files."""
    cc_table = resolved_cc_table

    dynamic_match = cast(dict[str, str], dynamic_cc_match(lcia_method=(lcia_method or cc_source)))
    expected_impact = str(dynamic_match["impact"]).strip()
    prepared_candidates = [
        (
            item,
            dynamic_compatible_share_frame(
                asocc_share=item,
                share_frame=item.frame_wide,
                lcia_method=lcia_method or cc_source,
                requested_years=[int(year) for year in years],
            ),
        )
        for item in asocc_shares
    ]
    prepared_asocc_shares = [
        (item, cast(pd.DataFrame, prepared_frame))
        for item, prepared_frame in prepared_candidates
        if prepared_frame is not None
    ]
    prepared_asocc_shares, has_ssp_dependent_shares = _filter_dynamic_share_ssp_scope(
        prepared_asocc_shares=prepared_asocc_shares,
        cc_table=cc_table,
    )
    _validate_dynamic_share_ssp_alignment(
        prepared_asocc_shares=prepared_asocc_shares,
        cc_table=cc_table,
        require_selected_ssp_shares=has_ssp_dependent_shares,
        cc_source=cc_source,
        resolved_cc_path=resolved_cc_path,
        requested_years=[int(year) for year in years],
    )

    out_dir = get_acc_output_dir(
        context=path_context,
        public_result_root_name=public_result_root_name,
    )
    n_share = 0
    output_dirs = [out_dir]
    total_status_steps = len(prepared_asocc_shares)
    status_step = 0
    cc_coverage = dynamic_cc_coverage(cc_table=cc_table)
    status_label = cc_branch_status_label(cc_source=lcia_method or cc_source, cc_type="dynamic_ar6")
    output_files: list[Path] = []

    for asocc_share, share_df in prepared_asocc_shares:
        share_year_cols = requested_year_columns(
            share_df,
            requested_years=[int(year) for year in years],
        )
        n_share += 1
        status_step += 1
        _emit_dynamic_status(
            status=status,
            step=status_step,
            total_steps=total_status_steps,
            display_name=asocc_share.display_name,
            category=", ".join(cc_coverage["cc_category"]),
            ssp=", ".join(cc_coverage[AR6_CC_SSP_SCENARIO_COLUMN]),
            cc_model=f"{len(cc_table)} CC rows",
            cc_label=status_label,
        )
        l1_l2_method = resolve_acc_l1_l2_method(
            frame=share_df,
            source_label=f"Dynamic aCC aSoCC share '{asocc_share.display_name}'",
        )
        acc_df = _build_dynamic_acc_frame(
            asocc_share=asocc_share,
            share_df=share_df,
            cc_table=cc_table,
            cc_source=lcia_method or cc_source,
            cc_path=resolved_cc_path,
            year_cols=share_year_cols,
            l1_l2_method=l1_l2_method,
            expected_impact=expected_impact,
            share_transition_meta=share_transition_meta,
        )
        rel_path = acc_output_relative_dir(upstream_relative_dir=asocc_share.relative_dir)
        branch_dir = out_dir / rel_path
        acc_path_stem = build_acc_output_stem(
            base_stem=asocc_share.file_stem,
            cc_source=lcia_method or cc_source,
            cc_type="dynamic_ar6",
        )
        output_path = (branch_dir / acc_path_stem).with_suffix(f".{fmt}")
        write_acc_output(
            acc_df,
            output_path,
            fmt,
        )
        output_files.append(output_path)

    n_written = len(output_files)

    variable = cc_variable(
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
    )
    return n_share, n_written, [variable], output_dirs, output_files, resolved_cc_path


def _build_dynamic_acc_frame(
    *,
    asocc_share: LoadedAsoccShare,
    share_df: pd.DataFrame,
    cc_table: pd.DataFrame,
    cc_source: str,
    cc_path: Path,
    year_cols: list[str],
    l1_l2_method: str,
    expected_impact: str,
    share_transition_meta: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Build one vectorized dynamic aCC table for one aSoCC share."""
    share_year_matrix = _numeric_year_matrix(
        share_df,
        year_cols=year_cols,
    )
    cc_year_matrix = _numeric_year_matrix(
        cc_table,
        year_cols=year_cols,
    )
    share_ssp = _share_row_ssp_array(share_df=share_df)
    share_rows, cc_rows = _dynamic_pair_indices(
        share_ssp=share_ssp,
        cc_ssp=_cc_row_ssp_array(cc_table),
        cc_count=len(cc_table),
    )
    impact_unit, conversion_factors = dynamic_acc_unit_factors(
        source_units=pd.Series(cc_table.iloc[cc_rows]["impact_unit"], copy=False),
        cc_source=cc_source,
        impact=expected_impact,
        source_path=cc_path,
    )
    values = (
        share_year_matrix[share_rows] * cc_year_matrix[cc_rows] * conversion_factors[:, np.newaxis]
    )
    out = share_df.drop(columns=year_cols).iloc[share_rows].reset_index(drop=True)
    for col_idx, year_col in enumerate(year_cols):
        out[year_col] = values[:, col_idx]
    out = materialize_acc_scope(
        out,
        l1_l2_method=l1_l2_method,
        impact=expected_impact,
        impact_unit=impact_unit,
        asocc_ssp_start_year=_asocc_share_ssp_start_year(
            asocc_share=asocc_share,
            share_transition_meta=share_transition_meta,
        ),
    )
    cc_meta = (
        cc_table.iloc[cc_rows]
        .loc[
            :,
            ["cc_model", "cc_scenario", "cc_category", "ssp_scenario", "cc_flow", "cc_variable"],
        ]
        .reset_index(drop=True)
    )
    return pd.concat(
        [
            cc_meta.rename(columns={"ssp_scenario": AR6_CC_SSP_SCENARIO_COLUMN}),
            out.reset_index(drop=True),
        ],
        axis=1,
    )


def _numeric_year_matrix(
    frame: pd.DataFrame,
    *,
    year_cols: list[str],
) -> np.ndarray:
    """Return one numeric matrix for the requested year columns."""
    resolved_columns = _resolve_year_columns(frame, year_cols=year_cols)
    numeric = cast(
        pd.DataFrame, frame.loc[:, resolved_columns].apply(pd.to_numeric, errors="raise")
    )
    return numeric.to_numpy(dtype=float, copy=True)


def _resolve_year_columns(
    frame: pd.DataFrame,
    *,
    year_cols: list[str],
) -> list[Any]:
    """Resolve exact year columns by their persisted string form."""
    columns_by_token = {str(column): column for column in frame.columns}
    return [columns_by_token[str(year_col)] for year_col in year_cols]


def _non_empty_unique_text_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted normalized text values from one AR6 CC identity column."""
    series = pd.Series(frame.loc[:, column], copy=False)
    values = {
        str(value).strip()
        for value in series.tolist()
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return sorted(values)


def _share_row_ssp_array(*, share_df: pd.DataFrame) -> np.ndarray:
    """Return row-owned aSoCC SSP labels, with ``None`` for SSP invariant rows."""
    if ASOCC_SSP_SCENARIO_COLUMN not in share_df.columns:
        return np.full(len(share_df), None, dtype=object)
    return np.array(
        [
            _optional_ssp_token(value)
            for value in pd.Series(
                share_df.loc[:, ASOCC_SSP_SCENARIO_COLUMN],
                copy=False,
            ).tolist()
        ],
        dtype=object,
    )


def _cc_row_ssp_array(cc_table: pd.DataFrame) -> np.ndarray:
    """Return canonical dynamic CC SSP labels row by row."""
    return np.array(
        [
            normalize_ssp_token(value, context="Dynamic CC SSP value")
            for value in pd.Series(cc_table.loc[:, "ssp_scenario"], copy=False).tolist()
        ],
        dtype=object,
    )


def _optional_ssp_token(value: object) -> str | None:
    """Return one optional canonical SSP label from a row value."""
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "nat", "<na>"}:
        return None
    return normalize_ssp_token(text, context="aSoCC SSP value")


def _dynamic_pair_indices(
    *,
    share_ssp: np.ndarray,
    cc_ssp: np.ndarray,
    cc_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained row pairs without allocating rejected Cartesian values."""
    invariant = pd.isna(pd.Series(share_ssp, copy=False)).to_numpy(dtype=bool)
    share_text = share_ssp.astype(str, copy=False)
    cc_text = cc_ssp.astype(str, copy=False)
    share_blocks: list[np.ndarray] = []
    cc_blocks: list[np.ndarray] = []
    invariant_rows = np.flatnonzero(invariant)
    if len(invariant_rows):
        share_blocks.append(np.repeat(invariant_rows, cc_count))
        cc_blocks.append(np.tile(np.arange(cc_count, dtype=np.int64), len(invariant_rows)))
    for token in np.unique(cc_text):
        share_rows = np.flatnonzero((~invariant) & (share_text == token))
        if not len(share_rows):
            continue
        cc_rows = np.flatnonzero(cc_text == token)
        share_blocks.append(np.repeat(share_rows, len(cc_rows)))
        cc_blocks.append(np.tile(cc_rows, len(share_rows)))
    return np.concatenate(share_blocks), np.concatenate(cc_blocks)


def _asocc_share_ssp_start_year(
    *,
    asocc_share: LoadedAsoccShare,
    share_transition_meta: dict[str, dict[str, object]],
) -> int | None:
    """Return the persisted aSoCC SSP transition year for one dynamic aSoCC share."""
    payload = share_transition_payload_for_output_stem(
        output_stem=asocc_share.file_stem,
        share_transition_meta=share_transition_meta,
    )
    value = payload.get("ssp_start_year")
    if value is None:
        return None
    return int(cast(int | float | str, value))


def _filter_dynamic_cc_subset(
    *,
    cc_table: pd.DataFrame,
    category: list[str] | None,
    ssp_scenario: list[str] | None,
) -> pd.DataFrame:
    """Filter one resolved dynamic CC table back to the requested public subset."""
    filtered = cc_table.copy()
    if category is not None:
        requested_categories = {str(value).strip() for value in category if str(value).strip()}
        category_series = cast(pd.Series, filtered["cc_category"]).astype(str).str.strip()
        filtered = filtered.loc[category_series.isin(sorted(requested_categories))].copy()
    if ssp_scenario is not None:
        requested_ssps = set(normalize_ssp_tokens(ssp_scenario))
        ssp_source = cast(pd.Series, filtered["ssp_scenario"])
        ssp_series = cast(
            pd.Series,
            pd.Series(
                [normalize_ssp_tokens([value])[0] for value in ssp_source.tolist()],
                index=filtered.index,
                copy=False,
            ),
        )
        filtered = filtered.loc[ssp_series.isin(sorted(requested_ssps))].copy()
    return filtered
