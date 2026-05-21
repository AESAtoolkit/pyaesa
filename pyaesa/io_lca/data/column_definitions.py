"""Shared text column definitions for IO-LCA upstream outputs."""

from pathlib import Path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import join_user_text_lines

_SELECTOR_DESCRIPTIONS = {
    "r_f": "Final demand region selected for the functional unit.",
    "r_c": "Consumer region selected for the functional unit.",
    "r_p": "Producer region selected for the functional unit.",
    "r_u": "Upstream region selector used by the functional unit contract.",
    "r_y": "Region selector applied to final-demand component F_Y.",
    "s_p": "Producer sector selected for the functional unit.",
    "s_u": "Upstream sector selector used by the functional unit contract.",
}


def _is_year_column(name: str) -> bool:
    """Return whether one label is a year column."""
    text = str(name).strip()
    if not text:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    if not numeric.is_integer():
        return False
    year = int(numeric)
    return 1800 <= year <= 2600


def _selector_description(column: str) -> str:
    """Return descriptive text for one selector column."""
    return _SELECTOR_DESCRIPTIONS.get(
        column,
        "Selector column inherited from the FU contract and current run filters.",
    )


def _origin_column_lines(*, columns: list[str]) -> list[str]:
    """Return deterministic definition lines for upstream origin columns."""
    lines: list[str] = [
        "Applies to:",
        "- origins__<method>.* (absolute values by year columns)",
        "- origins_ratio__<method>.* (share values by year columns)",
        "",
    ]
    seen = set()
    for raw in columns:
        column = str(raw)
        if column in seen:
            continue
        seen.add(column)
        if column == "impact":
            definition = "Impact category code."
        elif column == "origin_r_p":
            definition = "Producer region of the origin node."
        elif column == "origin_s_p":
            definition = (
                "Producer sector of the origin node. "
                "For direct final-demand origins, this is `F_Y`."
            )
        elif column == "impact_unit":
            definition = "Impact unit associated with the impact category."
        elif _is_year_column(column):
            definition = (
                "Year column. In absolute tables: absolute contribution value. "
                "In ratio tables: contribution share in [0, 1]."
            )
        else:
            definition = _selector_description(column)
        lines.extend([column, f"- {definition}", ""])
    return lines


def _stage_column_lines(*, columns: list[str]) -> list[str]:
    """Return deterministic definition lines for upstream stage columns."""
    lines: list[str] = []
    seen = set()
    for raw in columns:
        column = str(raw)
        if column in seen:
            continue
        seen.add(column)
        if column == "year":
            definition = "Studied year."
        elif column == "stage":
            definition = (
                "Supply-chain stage label (`n`, `n-1`, ...). "
                "The `direct_final_demand_FY` row is the separate F_Y component."
            )
        elif column == "stage_r_p":
            definition = "Producer region of the node at this stage."
        elif column == "stage_s_p":
            definition = "Producer sector of the node at this stage (blank for F_Y rows)."
        elif column == "linked_from_stage":
            definition = (
                "Parent stage from which this node is linked. "
                "Blank for root rows (`n` and `direct_final_demand_FY`)."
            )
        elif column == "linked_from_r_p":
            definition = "Producer region of the parent node."
        elif column == "linked_from_s_p":
            definition = "Producer sector of the parent node."
        elif column == "impact":
            definition = "Impact category code."
        elif column == "direct_at_stage":
            definition = (
                "Direct impact emitted at this stage node for this impact category. "
                "Represents Scope 1 (direct emissions at the node)."
            )
        elif column == "embedded_from_deeper_stages":
            definition = (
                "Impact embedded in this node and caused by deeper upstream stages. "
                "Represents Scope 2 and 3 upstream (supply-chain impacts)."
            )
        elif column == "stage_total":
            definition = (
                "Total for this node (`direct_at_stage + embedded_from_deeper_stages`). "
                "For CBA_FD/CBA_TD: Scope 1, 2, 3 upstream. For PBA: Scope 1 only."
            )
        elif column == "impact_unit":
            definition = "Impact unit associated with the impact category."
        else:
            definition = _selector_description(column)
        lines.extend([column, f"- {definition}", ""])
    return lines


def render_origin_columns_defs(*, columns: list[str]) -> str:
    """Render upstream origin column definitions as deterministic text."""
    lines = _origin_column_lines(columns=columns)
    while lines and lines[-1] == "":
        lines.pop()
    return join_user_text_lines(lines, trailing_newline=True)


def render_stage_columns_defs(*, columns: list[str]) -> str:
    """Render upstream stage column definitions as deterministic text."""
    lines = _stage_column_lines(columns=columns)
    while lines and lines[-1] == "":
        lines.pop()
    return join_user_text_lines(lines, trailing_newline=True)


def write_origin_columns_defs(*, path: Path, columns: list[str]) -> Path:
    """Write shared upstream origin text column definitions."""
    path = ensure_file_parent(path)
    path.write_text(render_origin_columns_defs(columns=columns), encoding="utf-8")
    return path


def write_stage_columns_defs(*, path: Path, columns: list[str]) -> Path:
    """Write shared upstream stage text column definitions."""
    path = ensure_file_parent(path)
    path.write_text(render_stage_columns_defs(columns=columns), encoding="utf-8")
    return path
