import importlib

import pytest

mod = importlib.import_module("pyaesa.asocc.orchestration.setup.loading.loading")


def test_validate_sector_filter_labels_real_paths(allocation_dummy_repo) -> None:
    mod._validate_sector_filter_labels(
        source=mod.ISO3_SOURCE_KEY,
        group_version=None,
        filters={"s_p": ["bad"]},
    )
    mod._validate_sector_filter_labels(
        source="exiobase_396_ixi",
        group_version=None,
        filters={"s_p": None},
    )
    mod._validate_sector_filter_labels(
        source="exiobase_396_ixi",
        group_version=None,
        filters={"s_p": [" ", ""]},
    )
    mod._validate_sector_filter_labels(
        source="exiobase_396_ixi",
        group_version=None,
        filters={"s_p": ["D"]},
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version="elec",
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
    )

    with pytest.raises(ValueError) as exc:
        mod._validate_sector_filter_labels(
            source="exiobase_396_ixi",
            group_version="elec",
            filters={"s_p": ["INVALID_SECTOR"]},
        )
    assert "matrix_version='elec'" in str(exc.value)
    assert "metadata.json" in str(exc.value)
