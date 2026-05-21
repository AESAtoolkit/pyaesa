import pytest

from pyaesa.asocc.data import region_group_mapping as mod


def test_load_region_group_mapping_success(allocation_dummy_repo) -> None:
    mapping = mod.load_region_group_mapping(
        source_key="exiobase_396_ixi",
        group_version="demo_reg",
    )
    assert mapping == {"FR": "EU", "US": "NAM"}


def test_load_region_group_mapping_wraps_read_errors() -> None:
    with pytest.raises(ValueError):
        mod.load_region_group_mapping(
            source_key="oecd_v2025",
            group_version="missing_reg",
        )
