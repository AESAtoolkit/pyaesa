import pandas as pd
import pytest

from pyaesa.asocc.data.enacting_metric_units import (
    lcia_unit_series_for_method,
    load_enacting_metric_units_from_metadata,
    parse_enacting_metric_units_from_year_entry,
)


def _year_entry_with_units() -> dict:
    return {
        "core": ["A", "L"],
        "extensions": {},
        "enacting_metrics": {
            "units": {
                "mrio_default_monetary": "M EUR",
                "mrio_by_metric": {
                    "fd_rf": "M EUR",
                    "gva_rp": "M EUR",
                },
                "lcia_by_method": {
                    "gwp100_lcia": {
                        "Climate change": "kg CO2-eq",
                        "Other climate": "kg CO2-eq",
                    }
                },
            }
        },
    }


def test_parse_enacting_metric_units_from_year_entry() -> None:
    default_unit, mrio_units, lcia_units = parse_enacting_metric_units_from_year_entry(
        year_entry=_year_entry_with_units(),
        year=2019,
    )
    assert default_unit == "M EUR"
    assert mrio_units["fd_rf"] == "M EUR"
    assert set(lcia_units.keys()) == {"gwp100_lcia"}
    assert isinstance(lcia_units["gwp100_lcia"], pd.Series)
    assert lcia_units["gwp100_lcia"].loc["Climate change"] == "kg CO2-eq"


def test_lcia_unit_series_for_method() -> None:
    unit_series = lcia_unit_series_for_method(
        year_entry=_year_entry_with_units(),
        year=2019,
        lcia_method="gwp100_lcia",
    )
    assert unit_series.loc["Other climate"] == "kg CO2-eq"

    with pytest.raises(ValueError, match="pb_lcia"):
        lcia_unit_series_for_method(
            year_entry=_year_entry_with_units(),
            year=2019,
            lcia_method="pb_lcia",
        )


def test_load_enacting_metric_units_from_metadata(allocation_dummy_repo) -> None:
    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2018, 2019],
    )
    payload = allocation_dummy_repo._read_mrio_metadata(source="oecd_v2025", matrix_version=None)
    payload["years"] = {
        "2018": _year_entry_with_units(),
        "2019": _year_entry_with_units(),
    }
    allocation_dummy_repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version=None,
        payload=payload,
    )

    default_unit, mrio_units, lcia_units = load_enacting_metric_units_from_metadata(
        source="oecd_v2025",
        matrix_version=None,
        years=[2018, 2019],
    )
    assert default_unit == "M EUR"
    assert mrio_units["gva_rp"] == "M EUR"
    assert "gwp100_lcia" in lcia_units


def test_load_enacting_metric_units_from_metadata_rejects_inconsistencies(
    allocation_dummy_repo,
) -> None:
    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="inconsistent",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2018, 2019],
    )
    bad_2019 = _year_entry_with_units()
    bad_2019["enacting_metrics"]["units"]["mrio_default_monetary"] = "k EUR"
    bad_2019["enacting_metrics"]["units"]["mrio_by_metric"] = {"fd_rf": "k EUR", "gva_rp": "k EUR"}
    inconsistent_payload = allocation_dummy_repo._read_mrio_metadata(
        source="oecd_v2025",
        matrix_version="inconsistent",
    )
    inconsistent_payload["years"] = {
        "2018": _year_entry_with_units(),
        "2019": bad_2019,
    }
    allocation_dummy_repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version="inconsistent",
        payload=inconsistent_payload,
    )

    with pytest.raises(ValueError):
        load_enacting_metric_units_from_metadata(
            source="oecd_v2025",
            matrix_version="inconsistent",
            years=[2018, 2019],
        )

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="missing_year",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2018],
    )
    missing_year_payload = allocation_dummy_repo._read_mrio_metadata(
        source="oecd_v2025",
        matrix_version="missing_year",
    )
    missing_year_payload["years"] = {"2018": _year_entry_with_units()}
    allocation_dummy_repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version="missing_year",
        payload=missing_year_payload,
    )
    with pytest.raises(ValueError, match="2019"):
        load_enacting_metric_units_from_metadata(
            source="oecd_v2025",
            matrix_version="missing_year",
            years=[2018, 2019],
        )

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="metric_inconsistent",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2018, 2019],
    )
    metric_2019 = _year_entry_with_units()
    metric_2019["enacting_metrics"]["units"]["mrio_by_metric"]["fd_rf"] = "k EUR"
    metric_payload = allocation_dummy_repo._read_mrio_metadata(
        source="oecd_v2025",
        matrix_version="metric_inconsistent",
    )
    metric_payload["years"] = {
        "2018": _year_entry_with_units(),
        "2019": metric_2019,
    }
    allocation_dummy_repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version="metric_inconsistent",
        payload=metric_payload,
    )
    with pytest.raises(ValueError):
        load_enacting_metric_units_from_metadata(
            source="oecd_v2025",
            matrix_version="metric_inconsistent",
            years=[2018, 2019],
        )

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="lcia_inconsistent",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2018, 2019],
    )
    lcia_2019 = _year_entry_with_units()
    lcia_2019["enacting_metrics"]["units"]["lcia_by_method"]["gwp100_lcia"] = {
        "Climate change": "t CO2-eq"
    }
    lcia_payload = allocation_dummy_repo._read_mrio_metadata(
        source="oecd_v2025",
        matrix_version="lcia_inconsistent",
    )
    lcia_payload["years"] = {
        "2018": _year_entry_with_units(),
        "2019": lcia_2019,
    }
    allocation_dummy_repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version="lcia_inconsistent",
        payload=lcia_payload,
    )
    with pytest.raises(ValueError):
        load_enacting_metric_units_from_metadata(
            source="oecd_v2025",
            matrix_version="lcia_inconsistent",
            years=[2018, 2019],
        )

    with pytest.raises(ValueError, match="years is empty"):
        load_enacting_metric_units_from_metadata(
            source="oecd_v2025",
            matrix_version=None,
            years=[],
        )
