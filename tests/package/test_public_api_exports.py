import pytest

import pyaesa
from pyaesa.asocc.uncertainty_asocc import uncertainty_asocc


def test_package_level_uncertainty_asocc_export() -> None:
    assert pyaesa.__getattr__("uncertainty_asocc") is uncertainty_asocc
    assert "uncertainty_asocc" in pyaesa.__dir__()
    assert "uncertainty_asocc" in pyaesa.__all__


def test_package_level_unknown_export_reports_missing_attribute() -> None:
    with pytest.raises(AttributeError, match="not_public"):
        pyaesa.__getattr__("not_public")
