"""LCIA impact inventory validation for external aSoCC Monte Carlo files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pyaesa.shared.lcia.contracts import bundled_cc_expected_impacts

from pyaesa.external_inputs.asocc.schema.file_specs import validate_lcia_axis_columns

_PC = cast(Any, pc)


@dataclass
class ImpactInventory:
    """Selected LCIA impact inventory for one external file."""

    path: Path
    lcia_method: str | None
    row_count: int = 0
    found: set[str] = field(default_factory=set)

    def validate_arrow_schema(self, table: pa.Table) -> None:
        """Validate the file impact column contract for one Arrow chunk."""
        validate_lcia_axis_columns(
            columns=table.column_names,
            path=self.path,
            lcia_method=self.lcia_method,
        )

    def observe_arrow(self, table: pa.Table) -> None:
        """Collect impact labels from one selected Arrow chunk."""
        self.validate_arrow_schema(table)
        self.row_count += table.num_rows
        if self.lcia_method is None:
            return
        self._extend(unique_nonempty_strings(table["impact"]))

    def observe_frame(self, frame: pd.DataFrame) -> None:
        """Collect impact labels from one selected non CSV frame."""
        self.validate_frame_schema(frame)
        if frame.empty:
            return
        self.row_count += len(frame)
        if self.lcia_method is None:
            return
        values = pd.Series(frame.loc[:, "impact"], copy=False).dropna().astype(str).str.strip()
        self._extend(values.unique())

    def validate_frame_schema(self, frame: pd.DataFrame) -> None:
        """Validate the file impact column contract for one pandas chunk."""
        validate_lcia_axis_columns(
            columns=frame.columns,
            path=self.path,
            lcia_method=self.lcia_method,
        )

    def validate(self) -> None:
        """Validate selected impact labels against bundled carrying capacity impacts."""
        if self.row_count == 0 or self.lcia_method is None:
            return
        cc_csv_path, expected = bundled_cc_expected_impacts(lcia_method=self.lcia_method)
        found = sorted(self.found)
        if found != expected:
            raise ValueError(
                "External aSoCC LCIA impacts must match the carrying capacity CSV "
                "exactly. "
                f"External file: '{self.path}'. Validation CSV: '{cc_csv_path}'. "
                f"Expected impacts: {expected}. Found: {found}."
            )

    def _extend(self, values) -> None:
        self.found.update(str(value) for value in values.tolist())


def unique_nonempty_strings(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Return unique non empty string values from an Arrow array."""
    text = _PC.utf8_trim_whitespace(pc.cast(values, pa.string()))
    present = _PC.and_(_PC.invert(_PC.is_null(text)), _PC.not_equal(text, ""))
    selected = text.filter(present)
    if len(selected) == 0:
        return np.empty(0, dtype=object)
    return np.asarray(_PC.unique(selected).to_pylist(), dtype=object)


def string_has_nonempty(values: pa.Array | pa.ChunkedArray) -> bool:
    """Return whether any Arrow value is a non empty string."""
    text = _PC.utf8_trim_whitespace(pc.cast(values, pa.string()))
    present = _PC.and_(_PC.invert(_PC.is_null(text)), _PC.not_equal(text, ""))
    return bool(_PC.any(present).as_py())
