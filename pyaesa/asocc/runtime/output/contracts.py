"""Typed output contracts for allocation runtime artifacts."""

from dataclasses import dataclass

import pandas as pd

from pyaesa.shared.selectors.scenarios import ssp_partition_token

_FILE_TOKEN_DELIMITER = "__"


@dataclass(frozen=True)
class IdentifierSchema:
    """Canonical identifier schema for one output artifact."""

    columns: tuple[str, ...]
    year_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputRoute:
    """Deterministic route metadata for one output artifact."""

    level: str
    bucket: str | None
    source: str | None
    grouped_mode: bool
    variant_tag: str | None
    ssp_scenario: str | None
    lcia_method: str | None
    projection_subfolder: str | None = None


@dataclass(frozen=True)
class OutputArtifact:
    """One canonical wide form output artifact.

    This is the validated payload passed to the writer for persistence.
    Path routing and method identity are resolved before this stage.
    """

    schema: IdentifierSchema
    data_wide: pd.DataFrame


@dataclass(frozen=True)
class OutputSpec:
    """Canonical wide frame output descriptor before persistence.

    Compared with :class:`OutputArtifact`, this contract intentionally excludes
    the DataFrame so setup logic can reason about paths/schemas without data.
    """

    l1_l2_method: str
    l2_method: str | None
    l1_method: str | None
    file_stem: str
    route: OutputRoute
    scenario_dependent: bool
    identifier_columns: tuple[str, ...]
    terminal_suffix: str | None = None

    @property
    def persisted_stem(self) -> str:
        """Return the persisted file stem."""
        return self.file_stem

    @property
    def stem_with_owned_tokens(self) -> str:
        """Return the canonical stem with scenario and terminal suffix tokens."""
        tokens = [self.persisted_stem]
        if self.route.ssp_scenario and self.scenario_dependent:
            tokens.append(
                scenario_file_token(
                    self.route.ssp_scenario,
                    context=f"Output file '{self.persisted_stem}'",
                )
            )
        suffix = _optional_file_token(self.terminal_suffix)
        if suffix is not None:
            tokens.append(suffix)
        return join_file_owned_tokens(*tokens)

    @property
    def file_name(self) -> str:
        """Return deterministic CSV filename."""
        return f"{self.stem_with_owned_tokens}.csv"

    def file_name_for_format(self, output_format: str) -> str:
        """Return deterministic filename for the configured output format."""
        suffix = {
            "csv": ".csv",
            "pickle": ".pickle",
            "parquet": ".parquet",
        }[output_format]
        return f"{self.stem_with_owned_tokens}{suffix}"


def join_file_owned_tokens(*tokens: str | None) -> str:
    """Return one canonical deterministic filename stem from owned tokens."""
    cleaned_tokens = [
        cleaned for token in tokens if (cleaned := _optional_file_token(token)) is not None
    ]
    if not cleaned_tokens:
        raise ValueError("Deterministic output filenames require at least one non-empty token.")
    return _FILE_TOKEN_DELIMITER.join(cleaned_tokens)


def scenario_file_token(value: object, *, context: str) -> str:
    """Return one canonical lowercase SSP filename token."""
    return ssp_partition_token(value, context=context)


def _optional_file_token(token: str | None) -> str | None:
    """Return one normalized file-owned token or None when blank."""
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    return text.strip("_")


def identifier_columns_from_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    """Extract canonical identifier columns from a frame index."""
    idx = frame.index
    if isinstance(idx, pd.MultiIndex):
        names = list(idx.names)
    else:
        names = [idx.name]
    if any(name is None for name in names):
        raise ValueError(
            f"Output frame index must have named identifier levels. Got index names={names}"
        )
    return tuple(str(name) for name in names)


def contract_year_columns(context) -> tuple[str, ...]:
    """Return the deterministic year column contract for persisted outputs.

    Persisted output years define one stable public wide schema for the branch.
    """
    years = getattr(context, "persisted_years", None)
    if isinstance(years, list) and years:
        return tuple(str(int(y)) for y in years)
    return tuple(str(int(y)) for y in context.resolved_years)


def persisted_method_columns_for_output_spec(spec: OutputSpec) -> tuple[str, ...]:
    """Return the canonical persisted method block for one output spec."""
    has_l1_l2 = bool(str(spec.l1_l2_method).strip())
    has_l2 = spec.l2_method is not None and bool(str(spec.l2_method).strip())
    has_l1 = spec.l1_method is not None and bool(str(spec.l1_method).strip())
    if not has_l2 and not has_l1_l2:
        return ("l1_method",) if has_l1 else ()
    columns: list[str] = ["l1_l2_method"]
    if has_l1:
        columns.append("l1_method")
    if has_l2:
        columns.append("l2_method")
    return tuple(columns)
