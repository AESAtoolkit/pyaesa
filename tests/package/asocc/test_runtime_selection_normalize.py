import pytest
from typing import Any, cast

from pyaesa.asocc.runtime.selection import normalize as mod


def test_normalize_l1_reg_mode_defaults_and_variants() -> None:
    assert mod.normalize_l1_reg_mode(None) == "post"
    assert mod.normalize_l1_reg_mode(" pre ") == "pre"
    with pytest.raises(ValueError):
        mod.normalize_l1_reg_mode("both")


def test_normalize_l1_reg_mode_required() -> None:
    assert mod.normalize_l1_reg_mode_required(" POST ") == "post"
    with pytest.raises(ValueError):
        mod.normalize_l1_reg_mode_required(None)
    with pytest.raises(ValueError):
        mod.normalize_l1_reg_mode_required("bad")


def test_normalize_output_mode() -> None:
    assert mod.normalize_output_mode(False) is False
    assert mod.normalize_output_mode(True) is True
    with pytest.raises(ValueError):
        mod.normalize_output_mode(cast(Any, "both"))
    with pytest.raises(ValueError):
        mod.normalize_output_mode(cast(Any, "invalid"))


def test_resolve_level() -> None:
    assert mod.resolve_level(fu_norm="L1.a") == "l1"
    assert mod.resolve_level(fu_norm="L2.c.b") == "l2"


def test_normalize_plan() -> None:
    assert mod.normalize_plan(" default ") == "default"
    assert mod.normalize_plan("one_step_pairs") == "one_step_pairs"
    with pytest.raises(ValueError):
        mod.normalize_plan("invalid")
