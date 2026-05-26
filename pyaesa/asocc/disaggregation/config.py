"""Argument validation for disaggregate_asocc."""

from typing import Any, cast

from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.selectors.time_selectors import normalize_time_selector_mapping

from pyaesa.external_inputs.asocc.schema.contracts import (
    normalize_external_method_selector,
)
from .models import (
    DisaggregationConfigModel,
    DisaggregationSpec,
    ParsedArgs,
    RunSelector,
)
from ..entrypoints.argument_contracts import (
    normalize_allocate_output_format,
)
from pyaesa.asocc.methods.registry.model.types import normalize_fu_code
from pyaesa.asocc.runtime.request.defaults import DISAGGREGATION_BASE_ALLOCATE_DEFAULTS

_REQUIRED_CONFIG_KEYS = {
    "target_agg_run",
    "ref_agg_run",
    "ref_disagg_run",
    "disaggregation_specs",
    "new_disagg_version_name",
}
_ALLOWED_CONFIG_KEYS = set(_REQUIRED_CONFIG_KEYS)
_ALLOWED_BASE_ALLOCATE_ARGS = {
    "project_name",
    "years",
    "fu_code",
    "r_p",
    "r_c",
    "r_f",
    "group_indices",
    "method_plan",
    "l1_methods",
    "one_step_methods",
    "two_step_methods",
    "l1_l2_pairs",
    "l1_reg_aggreg",
    "ssp_scenario",
    "projection_mode",
    "reg_window",
    "l2_reuse_years",
}
_FORBIDDEN_BASE_ALLOCATE_ARGS = {
    "source",
    "agg_reg",
    "agg_sec",
    "agg_version",
    "s_p",
    "lcia_method",
    "reference_years",
    "output_format",
    "intermediate_outputs",
    "refresh",
}
_BASE_ALLOCATE_REQUIRED_KEYS = {"project_name", "fu_code"}
_BASE_ALLOCATE_ARG_DEFAULTS = dict(DISAGGREGATION_BASE_ALLOCATE_DEFAULTS)
_ALLOWED_SELECTOR_KEYS = {"source", "agg_reg", "agg_sec", "agg_version", "s_p"}
_ALLOWED_SPEC_KEYS = {"agg_sector_label", "disagg_sector_label"}
_ALLOWED_SOURCES = {"oecd_v2025", "exiobase_396_ixi", "exiobase_3102_ixi"}


def _require_bool(value: Any, *, name: str) -> bool:
    """Require strict bool runtime argument."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{name}' must be a boolean. Use True or False.")


def normalize_non_empty_str(value: Any, *, name: str) -> str:
    """Normalize one required non-empty string argument."""
    if not isinstance(value, str):
        raise ValueError(
            f"'{name}' must be provided as a text value; found {type(value).__name__}."
        )
    text = value.strip()
    if not text:
        raise ValueError(f"'{name}' cannot be empty.")
    return text


def _normalize_non_empty_str_list(value: Any, *, name: str) -> list[str]:
    """Normalize one selector requiring a non-empty list[str]."""
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a non-empty list of strings.")
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError(
                f"'{name}' entries must be provided as text values; found {type(entry).__name__}."
            )
        text = entry.strip()
        if not text:
            raise ValueError(f"'{name}' cannot contain empty values.")
        out.append(text)
    if not out:
        raise ValueError(f"'{name}' must contain at least one value.")
    return out


def _parse_optional_bool(value: Any, *, name: str) -> bool:
    """Parse optional bool input with strict typing."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{name}' must be a boolean when provided. Use True or False.")


def parse_selector(name: str, raw: Any) -> RunSelector:
    """Parse one prerequisite deterministic selector payload."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"'disaggregation_config.{name}' must be a dictionary describing one "
            "prerequisite run selector."
        )
    unknown = sorted(set(raw) - _ALLOWED_SELECTOR_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown selector key(s) in 'disaggregation_config.{name}': {unknown}. "
            "Supported keys are 'source', 'agg_reg', 'agg_sec', 'agg_version', and 's_p'."
        )
    source = normalize_non_empty_str(raw.get("source"), name=f"{name}.source").lower()
    if source not in _ALLOWED_SOURCES:
        raise ValueError(
            f"Unsupported source in 'disaggregation_config.{name}.source': '{source}'. "
            f"Use one of {sorted(_ALLOWED_SOURCES)}."
        )
    agg_reg = _parse_optional_bool(raw.get("agg_reg"), name=f"{name}.agg_reg")
    agg_sec = _parse_optional_bool(raw.get("agg_sec"), name=f"{name}.agg_sec")
    agg_version_raw = raw.get("agg_version")
    agg_version = (
        normalize_non_empty_str(agg_version_raw, name=f"{name}.agg_version")
        if (agg_reg or agg_sec)
        else None
    )
    if not (agg_reg or agg_sec) and agg_version_raw not in {None, ""}:
        raise ValueError(
            f"'disaggregation_config.{name}.agg_version' must be omitted when "
            f"'disaggregation_config.{name}.agg_reg' and "
            f"'disaggregation_config.{name}.agg_sec' are both False."
        )
    s_p = _normalize_non_empty_str_list(raw.get("s_p"), name=f"{name}.s_p")
    return RunSelector(
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        s_p=s_p,
    )


def parse_specs(raw_specs: Any) -> list[DisaggregationSpec]:
    """Parse disaggregation specs and validate aggregated-to-disaggregate cardinality."""
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(
            "'disaggregation_config.disaggregation_specs' must be a non-empty list of "
            "{'agg_sector_label', 'disagg_sector_label'} mappings."
        )
    specs: list[DisaggregationSpec] = []
    disaggregate_to_aggregated: dict[str, str] = {}
    for idx, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(
                f"'disaggregation_config.disaggregation_specs[{idx}]' must be a dictionary."
            )
        unknown = sorted(set(raw) - _ALLOWED_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"'disaggregation_config.disaggregation_specs[{idx}]' contains unknown "
                f"key(s): {unknown}. Provide only 'agg_sector_label' and "
                "'disagg_sector_label'."
            )
        aggregated = normalize_non_empty_str(
            raw.get("agg_sector_label"),
            name=f"disaggregation_specs[{idx}].agg_sector_label",
        )
        disaggregate = normalize_non_empty_str(
            raw.get("disagg_sector_label"),
            name=f"disaggregation_specs[{idx}].disagg_sector_label",
        )
        existing = disaggregate_to_aggregated.get(disaggregate)
        if existing is not None and existing != aggregated:
            raise ValueError(
                "One disaggregate sector can map to exactly one aggregated sector. "
                f"Disaggregate sector '{disaggregate}' is mapped to both '{existing}' and "
                f"'{aggregated}'."
            )
        disaggregate_to_aggregated[disaggregate] = aggregated
        specs.append(
            DisaggregationSpec(
                agg_sector_label=aggregated,
                disagg_sector_label=disaggregate,
            )
        )
    return specs


def validate_specs_against_selectors(
    *,
    target_agg_run: RunSelector,
    ref_agg_run: RunSelector,
    ref_disagg_run: RunSelector,
    specs: list[DisaggregationSpec],
) -> None:
    """Validate selector sector lists against the disaggregation specs."""
    aggregated_labels = {spec.agg_sector_label for spec in specs}
    disaggregate_labels = {spec.disagg_sector_label for spec in specs}
    if set(target_agg_run.s_p) != aggregated_labels:
        raise ValueError(
            "'disaggregation_config.target_agg_run.s_p' must exactly match the aggregated "
            "labels declared in 'disaggregation_config.disaggregation_specs'."
        )
    if set(ref_agg_run.s_p) != aggregated_labels:
        raise ValueError(
            "'disaggregation_config.ref_agg_run.s_p' must exactly match the aggregated "
            "labels declared in 'disaggregation_config.disaggregation_specs'."
        )
    if set(ref_disagg_run.s_p) != disaggregate_labels:
        raise ValueError(
            "'disaggregation_config.ref_disagg_run.s_p' must exactly match the disaggregate "
            "labels declared in 'disaggregation_config.disaggregation_specs'."
        )
    if ref_agg_run.source != ref_disagg_run.source:
        raise ValueError(
            "'disaggregation_config.ref_agg_run.source' and "
            "'disaggregation_config.ref_disagg_run.source' must be identical because the "
            "reference aggregated and disaggregate runs must come from the same MRIO domain."
        )


def _parse_base_allocate_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate the public disaggregate_asocc base_asocc_args block."""
    if not isinstance(raw, dict):
        raise ValueError(
            "'base_asocc_args' must be a dictionary identifying the deterministic "
            "aSoCC scope to reuse for disaggregation."
        )
    unknown = sorted(set(raw) - _ALLOWED_BASE_ALLOCATE_ARGS - _FORBIDDEN_BASE_ALLOCATE_ARGS)
    if unknown:
        raise ValueError(
            f"'base_asocc_args' contains unknown key(s): {unknown}. "
            "Provide only the documented deterministic aSoCC keys in this envelope."
        )
    forbidden = sorted(set(raw) & _FORBIDDEN_BASE_ALLOCATE_ARGS)
    if forbidden:
        if "lcia_method" in forbidden:
            raise ValueError(
                "disaggregate_asocc is restricted to non LCIA scope. "
                "Remove 'base_asocc_args.lcia_method'."
            )
        if "reference_years" in forbidden:
            raise ValueError(
                "disaggregate_asocc does not accept 'base_asocc_args.reference_years'. "
                "Remove it from the disaggregation request."
            )
        raise ValueError(
            "'base_asocc_args' contains forbidden key(s): "
            f"{forbidden}. Use 'disaggregation_config' selectors and top-level runtime "
            "arguments for these controls."
        )
    if "project_name" not in raw:
        raise ValueError(
            "'base_asocc_args.project_name' is required because it identifies the "
            "deterministic aSoCC project scope to disaggregate."
        )
    if "fu_code" not in raw:
        raise ValueError(
            "'base_asocc_args.fu_code' is required because disaggregation is resolved "
            "for one L2 functional-unit branch."
        )
    fu_norm = normalize_fu_code(raw["fu_code"])
    if not fu_norm.startswith("L2."):
        raise ValueError(
            "disaggregate_asocc requires an L2 functional unit (for example 'L2.c.b')."
        )
    out = dict(_BASE_ALLOCATE_ARG_DEFAULTS)
    out.update(raw)
    out["fu_code"] = fu_norm
    return cast(dict[str, Any], normalize_time_selector_mapping(out))


def parse_disaggregate_args(
    *,
    disaggregation_config: dict[str, Any],
    base_allocate_args: dict[str, Any],
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    figure_external_method: dict[str, list[str]] | None,
    refresh: bool,
) -> ParsedArgs:
    """Validate public disaggregate_asocc arguments."""
    if not isinstance(disaggregation_config, dict):
        raise ValueError(
            "'disaggregation_config' must be a dictionary describing the "
            "aggregated and disaggregate deterministic prerequisite runs."
        )
    missing = sorted(_REQUIRED_CONFIG_KEYS - set(disaggregation_config))
    if missing:
        raise ValueError(
            f"'disaggregation_config' is missing required key(s): {missing}. "
            "Provide 'target_agg_run', 'ref_agg_run', 'ref_disagg_run', "
            "'disaggregation_specs', and 'new_disagg_version_name'."
        )
    unknown = sorted(set(disaggregation_config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"'disaggregation_config' contains unknown key(s): {unknown}. "
            "Use only the documented disaggregation envelope keys."
        )
    target_agg_run = parse_selector("target_agg_run", disaggregation_config["target_agg_run"])
    ref_agg_run = parse_selector("ref_agg_run", disaggregation_config["ref_agg_run"])
    ref_disagg_run = parse_selector("ref_disagg_run", disaggregation_config["ref_disagg_run"])
    specs = parse_specs(disaggregation_config["disaggregation_specs"])
    validate_specs_against_selectors(
        target_agg_run=target_agg_run,
        ref_agg_run=ref_agg_run,
        ref_disagg_run=ref_disagg_run,
        specs=specs,
    )
    new_label = normalize_non_empty_str(
        disaggregation_config["new_disagg_version_name"],
        name="disaggregation_config.new_disagg_version_name",
    )
    output_format_norm = normalize_allocate_output_format(output_format)
    figures_norm = _require_bool(figures, name="figures")
    figure_options_norm = normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
    )
    figure_format_norm = normalize_figure_format(figure_format)
    if figure_external_method is not None and not figures_norm:
        raise ValueError("figure_external_method is only valid when figures=True.")
    parsed = DisaggregationConfigModel(
        target_agg_run=target_agg_run,
        ref_agg_run=ref_agg_run,
        ref_disagg_run=ref_disagg_run,
        disaggregation_specs=specs,
        new_disagg_version_name=new_label,
    )
    parsed_base_allocate_args = _parse_base_allocate_args(base_allocate_args)
    return ParsedArgs(
        disaggregation=parsed,
        base_allocate_args=parsed_base_allocate_args,
        output_format=output_format_norm,
        figures=figures_norm,
        figure_options=figure_options_norm,
        figure_format=figure_format_norm,
        figure_external_method=normalize_external_method_selector(
            figure_external_method,
            fu_code=str(parsed_base_allocate_args["fu_code"]),
            argument_name="figure_external_method",
        ),
        refresh=_require_bool(refresh, name="refresh"),
    )
