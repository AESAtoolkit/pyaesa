"""Shared LCIA metadata used to label shared figure outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.shared.lcia.contracts import load_bundled_static_cc_rows
from pyaesa.shared.lcia.path_tokens import infer_lcia_method_from_path
from pyaesa.shared.figures.scope_values import visible_scope_values
from pyaesa.shared.tabular.scalars import display_scalar, is_display_missing

GWP100_LCIA_METHOD = "gwp100_lcia"
PB_LCIA_IMPACT_PANEL_ORDER = (
    "SOD",
    "AAL",
    "OA",
    "N",
    "P GLO",
    "FWU",
    "LSC",
    "BI FD",
    "aCO2",
    "EI",
)


@dataclass(frozen=True)
class LCIAMetadata:
    """Normalized LCIA metadata for figure rendering."""

    family: str
    schema_kind: str
    impacts: tuple[str, ...]
    labels: dict[str, str]
    units: dict[str, str]
    min_cc: dict[str, float]
    max_cc: dict[str, float]
    ratios: dict[str, float]

    @property
    def n_impacts(self) -> int:
        """Return the number of impact codes represented in this metadata."""
        return len(self.impacts)


def load_lcia_metadata(cc_source: str) -> LCIAMetadata:
    """Load LCIA metadata for one bundled static carrying capacity source."""
    _cc_csv_path, schema_kind, rows = load_bundled_static_cc_rows(lcia_method=cc_source)
    impacts: list[str] = []
    labels: dict[str, str] = {}
    units: dict[str, str] = {}
    min_cc: dict[str, float] = {}
    max_cc: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for row in rows:
        impact = row.impact
        alias = str(row.impact_full_name_normalized).strip()
        label = format_impact_label(schema_kind=str(schema_kind), row=row)
        impacts.append(impact)
        for key in dict.fromkeys(key for key in (impact, alias) if key):
            labels[key] = label
            units[key] = row.impact_unit
            min_cc[key] = row.min_cc
            max_cc[key] = row.max_cc
            ratios[key] = row.max_cc / row.min_cc if row.min_cc != 0 else float("inf")
    return LCIAMetadata(
        family=str(cc_source).strip(),
        schema_kind=str(schema_kind),
        impacts=tuple(impacts),
        labels=labels,
        units=units,
        min_cc=min_cc,
        max_cc=max_cc,
        ratios=ratios,
    )


def ordered_impact_panels(*, lcia_method: str, impacts: list[str]) -> list[str]:
    """Return an LCIA method specific subplot order when one is defined."""
    method = str(lcia_method).strip()
    if method != "pb_lcia":
        metadata = load_lcia_metadata(method)
        requested = {str(impact).strip() for impact in impacts}
        ordered = [impact for impact in metadata.impacts if impact in requested]
        ordered.extend(sorted(impact for impact in requested if impact not in set(ordered)))
        return ordered
    requested = {str(impact).strip() for impact in impacts}
    ordered = [impact for impact in PB_LCIA_IMPACT_PANEL_ORDER if impact in requested]
    ordered.extend(sorted(impact for impact in requested if impact not in set(ordered)))
    return ordered


def format_impact_label(*, schema_kind: str, row: object) -> str:
    """Return one canonical visible impact label from bundled CC metadata."""
    impact = str(getattr(row, "impact")).strip()
    if str(schema_kind).strip() == "planetary boundary":
        planetary_boundary = getattr(row, "planetary_boundary")
        control_variable = getattr(row, "control_variable")
        boundary = "" if planetary_boundary is None else str(planetary_boundary).strip()
        control = "" if control_variable is None else str(control_variable).strip()
        root = f"{boundary}: {control}" if control else boundary
        if not root:
            raise ValueError(
                "Bundled planetary-boundary metadata is missing the boundary display text "
                f"for impact '{impact}'."
            )
        return f"{root} ({impact})"
    label_root = str(getattr(row, "impact_full_name_normalized")).strip()
    return f"{label_root} ({impact})"


def resolve_impact_title(*, lcia_method: str, impact: str) -> str:
    """Return the exact figure title label for one LCIA impact."""
    metadata = load_lcia_metadata(str(lcia_method).strip())
    return _impact_title_from_metadata(
        metadata=metadata,
        lcia_method=lcia_method,
        impact_text=str(impact).strip(),
    )


def _impact_title_from_metadata(
    *,
    metadata: LCIAMetadata,
    lcia_method: str,
    impact_text: str,
) -> str:
    """Return one impact title from already loaded LCIA metadata."""
    impact_text = str(impact_text).strip()
    if impact_text not in metadata.labels:
        if metadata.schema_kind == "standard" and metadata.n_impacts == 1:
            return str(metadata.labels[metadata.impacts[0]]).strip()
        raise ValueError(
            "Bundled LCIA metadata is missing the requested impact label for figure rendering. "
            f"lcia_method='{lcia_method}', impact='{impact_text}'."
        )
    return str(metadata.labels[impact_text]).strip()


def ensure_frame_lcia_method_metadata(
    frame: pd.DataFrame,
    *,
    lcia_method_column: str = "lcia_method",
    impact_column: str = "impact",
) -> pd.DataFrame:
    """Return a frame with explicit LCIA method metadata when impact rows are present."""
    if impact_column not in frame.columns:
        return frame
    source_path = frame.attrs.get("source_path")
    inferred = None
    if isinstance(source_path, str) and source_path.strip():
        inferred = infer_lcia_method_from_path(Path(source_path))
    if inferred is not None:
        out = frame.copy()
        out[lcia_method_column] = str(inferred)
        out.attrs.update(frame.attrs)
        return out
    if lcia_method_column in frame.columns:
        return frame
    impact_values = [
        str(value).strip()
        for value in frame[impact_column].tolist()
        if not is_display_missing(value)
    ]
    if not impact_values:
        return frame
    raise ValueError(
        "Impact scoped figure input is missing an explicit 'lcia_method' column and the "
        "persisted filename does not encode an LCIA method token."
    )


def resolve_frame_impact_title(
    frame: pd.DataFrame,
    *,
    lcia_method_column: str = "lcia_method",
    impact_column: str = "impact",
) -> str | None:
    """Return one strict resolved impact title for a frame slice."""
    if frame.empty or impact_column not in frame.columns:
        return None
    frame = ensure_frame_lcia_method_metadata(
        frame,
        lcia_method_column=lcia_method_column,
        impact_column=impact_column,
    )
    if lcia_method_column not in frame.columns:
        return None
    labels: set[str] = set()
    metadata_by_method: dict[str, LCIAMetadata] = {}
    for _index, row in frame.iterrows():
        impact = row.get(impact_column)
        if is_display_missing(impact):
            continue
        lcia_method = row.get(lcia_method_column)
        if is_display_missing(lcia_method):
            raise ValueError(
                "Figure rendering received impact-scoped rows with a missing 'lcia_method'."
            )
        lcia_method_text = cast(str, display_scalar(lcia_method))
        impact_text = cast(str, display_scalar(impact))
        method = lcia_method_text.strip()
        metadata = metadata_by_method.get(method)
        if metadata is None:
            metadata = load_lcia_metadata(method)
            metadata_by_method[method] = metadata
        labels.add(
            _impact_title_from_metadata(
                metadata=metadata,
                lcia_method=method,
                impact_text=impact_text.strip(),
            )
        )
    if not labels:
        return None
    if len(labels) != 1:
        raise ValueError(
            "Figure rendering found multiple visible impact labels inside one impact-scoped "
            f"slice: {sorted(labels)}."
        )
    return sorted(labels)[0]


def lcia_title_parts(
    frame: pd.DataFrame,
    *,
    include_impact: bool,
    lcia_method_column: str = "lcia_method",
    impact_column: str = "impact",
) -> list[str]:
    """Return compact LCIA title parts for figure scope titles."""
    methods = visible_scope_values(frame, lcia_method_column)
    if not methods:
        return []
    method = methods[0]
    impact_title = _single_frame_impact_title(
        frame,
        lcia_method_column=lcia_method_column,
        impact_column=impact_column,
    )
    if method == GWP100_LCIA_METHOD and impact_title is not None:
        return [impact_title]
    parts = [method]
    if include_impact and impact_title is not None:
        parts.append(impact_title)
    return parts


def _single_frame_impact_title(
    frame: pd.DataFrame,
    *,
    lcia_method_column: str,
    impact_column: str,
) -> str | None:
    if len(visible_scope_values(frame, impact_column)) != 1:
        return None
    return resolve_frame_impact_title(
        frame,
        lcia_method_column=lcia_method_column,
        impact_column=impact_column,
    )


def resolve_frame_impact_unit(
    frame: pd.DataFrame,
    *,
    lcia_method_column: str = "lcia_method",
    impact_column: str = "impact",
    impact_unit_column: str = "impact_unit",
) -> str | None:
    """Return one strict resolved impact unit for a frame slice."""
    if frame.empty:
        return None
    if impact_unit_column in frame.columns:
        explicit_units = sorted(
            {
                str(value).strip()
                for value in frame[impact_unit_column].tolist()
                if not is_display_missing(value) and str(value).strip()
            }
        )
        if len(explicit_units) > 1:
            raise ValueError(
                "Figure rendering found multiple impact unit labels inside one impact-scoped "
                f"slice: {explicit_units}."
            )
        if explicit_units:
            return explicit_units[0]
    if impact_column not in frame.columns:
        return None
    frame = ensure_frame_lcia_method_metadata(
        frame,
        lcia_method_column=lcia_method_column,
        impact_column=impact_column,
    )
    if lcia_method_column not in frame.columns:
        return None
    units: set[str] = set()
    metadata_by_method: dict[str, LCIAMetadata] = {}
    for _index, row in frame.iterrows():
        impact = row.get(impact_column)
        if is_display_missing(impact):
            continue
        lcia_method = row.get(lcia_method_column)
        if is_display_missing(lcia_method):
            raise ValueError(
                "Figure rendering received impact-scoped rows with a missing 'lcia_method'."
            )
        lcia_method_text = cast(str, display_scalar(lcia_method))
        impact_text = cast(str, display_scalar(impact))
        method = str(lcia_method_text).strip()
        impact_key = impact_text.strip()
        metadata = metadata_by_method.get(method)
        if metadata is None:
            metadata = load_lcia_metadata(method)
            metadata_by_method[method] = metadata
        if impact_key not in metadata.units:
            raise ValueError(
                "Bundled LCIA metadata is missing the requested impact unit for figure rendering. "
                f"lcia_method='{method}', impact='{impact_key}'."
            )
        unit = str(metadata.units[impact_key]).strip()
        units.add(unit)
    if not units:
        return None
    if len(units) != 1:
        raise ValueError(
            "Figure rendering found multiple resolved impact units inside one impact-scoped "
            f"slice: {sorted(units)}."
        )
    return sorted(units)[0]
