import pytest

import pyaesa


def test_package_dir_lists_public_exports() -> None:
    exported = dir(pyaesa)

    assert "set_workspace" in exported
    assert "deterministic_asocc" in exported


def test_package_missing_attribute_raises_public_api_error() -> None:
    with pytest.raises(AttributeError, match="not_a_public_api"):
        getattr(pyaesa, "not_a_public_api")
