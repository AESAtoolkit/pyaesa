"""Validation runner for allocation share checks on deterministic aSoCC outputs."""

from pathlib import Path
from typing import NamedTuple, TypedDict

import pandas as pd

from pyaesa import deterministic_asocc
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.asocc.runtime.selection.normalize import normalize_l1_reg_mode
from pyaesa.workspace_initialisation.workspace import get_default_repo_root

from .utils.io_helpers import scalar_float
from .utils.l1_validators import L1ValidationContext, validate_l1_outputs
from .utils.l2_contexts import L2RunContext
from .utils.l2_star_b_validators import validate_l2_star_b_outputs
from .utils.l2_validators import validate_l2_outputs
from .utils.report_helpers import (
    combined_columns,
    normalize_report_frame,
    per_fu_columns,
    select_existing_columns,
)
from .utils.workflow_helpers import (
    resolve_lcia_methods,
    resolve_validation_output_paths,
    validation_project_name,
)

YEAR: int = 2019
REFERENCE_YEARS: int | list[int] | None = 1995
PROJECT_NAME = "allocation_validation"
SOURCES = "exiobase_3102_ixi"
L1_FUS = ("L1.a", "L1.b")
L2_FUS = ("L2.a.a", "L2.a.b", "L2.a.c", "L2.b.a", "L2.b.b", "L2.c.a", "L2.c.b")
EXIO_LCIA_METHODS = "gwp100_lcia"
L2_BUCKETS = ("l2_in_l1", "l2_vs_global")
ATOL = 1e-6
OUTPUT_FORMAT = "pickle"

GROUP_REG: bool | None = None
GROUP_SEC: bool | None = None
AGG_VERSION: str | None = None
L1_REG_AGGREG: str | list[str] | None = "pre"

METHOD_PLAN: str = "default"
L1_METHODS: list[str] | None = None
ONE_STEP_METHODS: list[str] | None = None
TWO_STEP_METHODS: list[str] | None = None
L1_L2_PAIRS: list[str] | None = None
REFRESH_ALLOCATE_OUTPUTS: bool = False
REFRESH_VALIDATION_REPORTS: bool = False


def _as_list(value: object) -> list[object]:
    """Wrap scalar config values into a one item list for uniform iteration."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_single_year(value: object) -> int:
    """Return one validated year; reject list/tuple multi year config."""
    if isinstance(value, (list, tuple, set)):
        raise ValueError(f"YEAR must be a single int, got: {value!r}")
    if isinstance(value, bool):
        raise TypeError(f"YEAR must be an int year, got bool: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("YEAR must not be empty.")
        return int(text)
    raise TypeError(f"YEAR must be int/float/str, got: {type(value).__name__}")


def _fu_report_path(
    *,
    reports_dir: Path,
    source: str,
    fu_code: str,
    year_tag: str,
) -> Path:
    """Return per FU report path in the validation output directory."""
    fu_stem = fu_code.replace(".", "_")
    return reports_dir / f"sum_to_one_report_{source}_{fu_stem}_{year_tag}.csv"


def _print_validation_summary(
    *,
    report_df: pd.DataFrame,
) -> None:
    """Print per FU pass rate summary from the combined validation DataFrame."""
    passed_bool = (
        report_df["passed"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )
    summary = (
        pd.DataFrame(
            {
                "source": report_df["source"].astype(str),
                "fu_code": report_df["fu_code"].astype(str),
                "passed_bool": passed_bool.astype(bool),
            }
        )
        .groupby(["source", "fu_code"], dropna=False)["passed_bool"]
        .agg(rows="size", passed_rows="sum")
        .reset_index()
    )
    summary["passed_pct"] = (summary["passed_rows"] / summary["rows"]) * 100.0
    print("Validation pass summary by FU:")
    for _, item in summary.iterrows():
        passed_pct = scalar_float(item["passed_pct"])
        passed_rows = int(scalar_float(item["passed_rows"]))
        total_rows = int(scalar_float(item["rows"]))
        print(
            f"- {item['source']} / {item['fu_code']}: "
            f"{passed_pct:.1f}% "
            f"({passed_rows}/{total_rows})"
        )
    if "validation_note" not in report_df.columns:
        return
    notes = report_df["validation_note"].fillna("").astype(str).str.strip()
    flagged = report_df.loc[notes != ""].copy()
    if flagged.empty:
        return
    print("\nValidation notes (coverage diagnostics):")
    cols = [
        "source",
        "fu_code",
        "method",
        "impact",
        "reference_year",
        "validation_note",
    ]
    unique_notes = flagged[cols].drop_duplicates()
    preview = unique_notes.head(10)
    for _, item in preview.iterrows():
        print(
            f"- {item['source']} / {item['fu_code']} / {item['method']} "
            f"(impact={item['impact']}, reference_year={item['reference_year']}): "
            f"{item['validation_note']}"
        )
    remaining = len(unique_notes) - len(preview)
    if remaining > 0:
        print(f"- ... {remaining} additional note row(s) in CSV report.")


def _ensure_repo_root() -> Path:
    """Ensure package repo root is configured and return it."""
    try:
        return get_default_repo_root()
    except RuntimeError as exc:
        raise RuntimeError(
            "Workspace repository root not configured. Run `set_workspace(<top_path>)` "
            "before running allocation validation."
        ) from exc


def _resolved_matrix_version() -> str | None:
    """Return processed data matrix version exactly from user AGG_VERSION."""
    if AGG_VERSION is None:
        return None
    version = str(AGG_VERSION).strip()
    return version or None


def _is_l2_star_b_fu(fu_code: str) -> bool:
    """Return whether FU is an L2*b functional unit."""
    code = str(fu_code).strip()
    return code.startswith("L2.") and code.endswith(".b")


_CONFIG_KEYS = (
    "YEAR",
    "REFERENCE_YEARS",
    "PROJECT_NAME",
    "SOURCES",
    "L1_FUS",
    "L2_FUS",
    "EXIO_LCIA_METHODS",
    "L2_BUCKETS",
    "ATOL",
    "OUTPUT_FORMAT",
    "GROUP_REG",
    "GROUP_SEC",
    "AGG_VERSION",
    "L1_REG_AGGREG",
    "METHOD_PLAN",
    "L1_METHODS",
    "ONE_STEP_METHODS",
    "TWO_STEP_METHODS",
    "L1_L2_PAIRS",
    "REFRESH_ALLOCATE_OUTPUTS",
    "REFRESH_VALIDATION_REPORTS",
)


class ValidationRunInputs(NamedTuple):
    """Resolved runtime inputs for one validation execution."""

    year: int
    reports_project_root: Path
    sources: list[str]
    l1_fus: list[str]
    l2_fus: list[str]
    l2_star_b_fus: set[str]
    buckets: list[str]
    include_intermediate_outputs: bool
    exio_lcia_methods: str | list[str]
    l1_modes: list[str]
    per_fu_reports_dir: Path
    year_tag: str
    combined_path: Path
    matrix_version: str | None


class ValidationAllocateArgs(TypedDict, total=False):
    """Typed deterministic_asocc request used by the validation runner."""

    project_name: str
    source: str
    years: list[int]
    fu_code: str
    agg_reg: bool
    agg_sec: bool
    agg_version: str
    l1_reg_aggreg: str
    group_indices: bool
    method_plan: str
    l1_methods: list[str] | None
    one_step_methods: list[str] | None
    two_step_methods: list[str] | None
    l1_l2_pairs: list[str] | None
    reference_years: int | list[int] | None
    output_format: str
    intermediate_outputs: bool
    figures: bool
    lcia_method: str | list[str]


def _normalized_l1_modes(value: str | list[str] | None) -> list[str]:
    """Normalize validation L1 aggregation config to explicit mode values."""
    if isinstance(value, list):
        modes: list[str] = []
        for item in value:
            modes.append(normalize_l1_reg_mode(item))
        deduped = list(dict.fromkeys(modes))
        if not deduped:
            raise ValueError("L1_REG_AGGREG list must contain at least one mode.")
        return deduped
    return [normalize_l1_reg_mode(value)]


def _build_run_inputs(*, year: int) -> ValidationRunInputs:
    """Resolve and normalize effective runtime configuration."""
    reports_project_root = outputs_project_root(project_name=PROJECT_NAME)
    sources = [str(v) for v in _as_list(SOURCES)]
    l1_fus = [str(v) for v in _as_list(L1_FUS)]
    l2_fus = [str(v) for v in _as_list(L2_FUS)]
    buckets = [str(v) for v in _as_list(L2_BUCKETS)]
    include_intermediate_outputs = any(
        bucket == "utility_propagation_contrib" for bucket in buckets
    )
    matrix_version = _resolved_matrix_version()
    per_fu_reports_dir = reports_project_root / "validation"
    per_fu_reports_dir.mkdir(parents=True, exist_ok=True)
    return ValidationRunInputs(
        year=year,
        reports_project_root=reports_project_root,
        sources=sources,
        l1_fus=l1_fus,
        l2_fus=l2_fus,
        l2_star_b_fus={fu for fu in l2_fus if _is_l2_star_b_fu(fu)},
        buckets=buckets,
        include_intermediate_outputs=include_intermediate_outputs,
        exio_lcia_methods=resolve_lcia_methods(EXIO_LCIA_METHODS),
        l1_modes=_normalized_l1_modes(L1_REG_AGGREG),
        per_fu_reports_dir=per_fu_reports_dir,
        year_tag=str(year),
        combined_path=per_fu_reports_dir / f"sum_to_one_report_{year}.csv",
        matrix_version=matrix_version,
    )


def _target_project_name(*, source: str, fu_code: str, l1_mode: str) -> str:
    """Return canonical validation project name for one deterministic target."""
    return validation_project_name(
        base_project_name=PROJECT_NAME,
        source=source,
        fu_code=fu_code,
        l1_reg_aggreg=l1_mode,
    )


def _allocate_args_for_fu(
    *,
    inputs: ValidationRunInputs,
    source: str,
    fu_code: str,
    l1_mode: str,
) -> ValidationAllocateArgs:
    """Return one current deterministic_asocc request for a validation target."""
    request: ValidationAllocateArgs = {
        "project_name": _target_project_name(source=source, fu_code=fu_code, l1_mode=l1_mode),
        "source": source,
        "years": [inputs.year],
        "fu_code": fu_code,
        "agg_reg": bool(GROUP_REG),
        "agg_sec": bool(GROUP_SEC),
        "agg_version": "" if AGG_VERSION is None else str(AGG_VERSION),
        "l1_reg_aggreg": l1_mode,
        "group_indices": False,
        "method_plan": METHOD_PLAN,
        "l1_methods": L1_METHODS,
        "one_step_methods": ONE_STEP_METHODS,
        "two_step_methods": TWO_STEP_METHODS,
        "l1_l2_pairs": L1_L2_PAIRS,
        "reference_years": REFERENCE_YEARS,
        "output_format": OUTPUT_FORMAT,
        "intermediate_outputs": bool(inputs.include_intermediate_outputs),
        "figures": False,
    }
    if str(source).startswith("exiobase_"):
        request["lcia_method"] = inputs.exio_lcia_methods
    return request


def _ensure_fu_outputs(
    *,
    inputs: ValidationRunInputs,
    source: str,
    fu_code: str,
) -> None:
    """Ensure deterministic_asocc outputs exist for one FU."""
    for l1_mode in inputs.l1_modes:
        base_allocate_args = _allocate_args_for_fu(
            inputs=inputs,
            source=source,
            fu_code=fu_code,
            l1_mode=l1_mode,
        )
        target_label = f"{source} / {fu_code} / l1_reg_aggreg={l1_mode}"
        if REFRESH_ALLOCATE_OUTPUTS:
            reason = "Refreshing deterministic_asocc outputs for validation"
        else:
            reason = "Ensuring deterministic_asocc outputs for validation"
        print(f"{reason}: {target_label}")
        _run_allocate_for_request(
            base_allocate_args=base_allocate_args,
            refresh=REFRESH_ALLOCATE_OUTPUTS,
        )


def _validate_fu_report(
    *,
    inputs: ValidationRunInputs,
    source: str,
    fu_code: str,
) -> pd.DataFrame:
    """Build one per FU validation DataFrame."""
    fu_rows: list[dict[str, object]] = []
    for l1_mode in inputs.l1_modes:
        base_allocate_args = _allocate_args_for_fu(
            inputs=inputs,
            source=source,
            fu_code=fu_code,
            l1_mode=l1_mode,
        )
        output_paths = resolve_validation_output_paths(
            base_allocate_args=base_allocate_args,
            buckets=inputs.buckets,
        )
        if fu_code.startswith("L1."):
            fu_rows.extend(
                validate_l1_outputs(
                    L1ValidationContext(
                        share_dir=output_paths.l1_share_dir,
                        source=source,
                        fu_code=fu_code,
                        year=inputs.year,
                        l1_mode=l1_mode,
                        output_format=OUTPUT_FORMAT,
                        atol=ATOL,
                    )
                )
            )
            continue

        l2_context = L2RunContext(
            l2_root=output_paths.l2_share_root,
            validation_project_name_root=PROJECT_NAME,
            source=source,
            fu_code=fu_code,
            year=inputs.year,
            l1_mode=l1_mode,
            buckets=inputs.buckets,
            output_format=OUTPUT_FORMAT,
            atol=ATOL,
            matrix_version=inputs.matrix_version,
            agg_reg=GROUP_REG,
            group_indices=False,
        )
        if fu_code in inputs.l2_star_b_fus:
            fu_rows.extend(validate_l2_star_b_outputs(l2_context))
        else:
            fu_rows.extend(validate_l2_outputs(l2_context))
    return select_existing_columns(normalize_report_frame(pd.DataFrame(fu_rows)), per_fu_columns())


def run_allocation_methods_with_config(
    config: dict[str, object] | None = None,
    **overrides: object,
) -> None:
    """Apply config overrides, then run the validation workflow."""
    updates: dict[str, object] = {}
    if config:
        updates.update(config)
    updates.update(overrides)
    unknown = sorted(set(updates) - set(_CONFIG_KEYS))
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    globals().update(updates)
    run_allocation_methods_sum_to_one_rules()


def run_allocation_methods_sum_to_one_rules() -> None:
    """Run allocation on current deterministic outputs and verify validation rules."""
    year = _as_single_year(YEAR)
    _ensure_repo_root()
    inputs = _build_run_inputs(year=year)

    print(
        "Effective validation config: "
        f"sources={list(inputs.sources)}, year={year}, "
        f"L1_FUS={list(inputs.l1_fus)}, L2_FUS={list(inputs.l2_fus)}, "
        f"L2_STAR_B_FUS={sorted(inputs.l2_star_b_fus)}, buckets={list(inputs.buckets)}, "
        f"ATOL={ATOL}, AGG_VERSION={inputs.matrix_version}, "
        f"REFRESH_ALLOCATE_OUTPUTS={REFRESH_ALLOCATE_OUTPUTS}, "
        f"REFRESH_VALIDATION_REPORTS={REFRESH_VALIDATION_REPORTS}",
        end="\n\n",
    )

    combined_parts: list[pd.DataFrame] = []
    wrote_any_fu_report = False
    for source in inputs.sources:
        for fu_code in [*inputs.l1_fus, *inputs.l2_fus]:
            fu_report_path = _fu_report_path(
                reports_dir=inputs.per_fu_reports_dir,
                source=source,
                fu_code=fu_code,
                year_tag=inputs.year_tag,
            )

            if not REFRESH_VALIDATION_REPORTS and fu_report_path.exists():
                print(f"Using existing validation report: {fu_report_path}")
                combined_parts.append(pd.read_csv(fu_report_path))
                continue

            _ensure_fu_outputs(inputs=inputs, source=source, fu_code=fu_code)
            per_fu_df = _validate_fu_report(inputs=inputs, source=source, fu_code=fu_code)
            fu_report_path.parent.mkdir(parents=True, exist_ok=True)
            per_fu_df.to_csv(fu_report_path, index=False)
            print(f"Wrote FU validation report: {fu_report_path}", end="\n\n", flush=True)
            combined_parts.append(per_fu_df)
            wrote_any_fu_report = True

    report_df = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    report_path = inputs.combined_path
    if not report_df.empty:
        combined = select_existing_columns(report_df.copy(), combined_columns())
        if REFRESH_VALIDATION_REPORTS or wrote_any_fu_report or not report_path.exists():
            combined.to_csv(report_path, index=False)
            print(f"Combined validation report written to: {report_path}")
        else:
            print(f"Using existing combined validation report: {report_path}")
            combined = pd.read_csv(report_path)
        report_df = combined

    if report_df.empty:
        raise RuntimeError(
            f"Error: Validation produced an empty combined report at {report_path!s}."
        )

    _print_validation_summary(report_df=report_df)


def _run_allocate_for_request(
    *,
    base_allocate_args: ValidationAllocateArgs,
    refresh: bool,
) -> None:
    """Run deterministic_asocc for one validation target."""
    deterministic_asocc(
        **base_allocate_args,
        refresh=refresh,
    )


if __name__ == "__main__":
    run_allocation_methods_sum_to_one_rules()
