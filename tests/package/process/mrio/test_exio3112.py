"""Public EXIOBASE 3.11.2 processing with real tiny archives and bundled methods."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaesa import process_mrio
from pyaesa.process.mrios.utils.io.metadata import _read_metadata, _write_metadata
from pyaesa.process.mrios.utils.io.paths import (
    _get_characterization_matrix_path,
    _get_year_saved_dir,
)
from pyaesa.process.mrios.utils.raw_corrections.runtime import load_raw_corrected_value_rows
from tests.package.helpers.data_processing_dummy import (
    write_characterization_matrix,
    write_exio_archive_files,
)


@pytest.mark.parametrize("source", ("exiobase_3112_ixi", "exiobase_3112_pxp"))
def test_process_exio3112_bundled_methods_and_absent_2024_land(
    project_repo: Path, source: str
) -> None:
    """Bundled methods characterize available accounts and report missing ones."""
    methods = ("pb_lcia", "ef_3.1", "ef_3.1_14cat", "gwp100_ef31", "gwp100_ipcc")
    write_exio_archive_files(
        project_repo,
        source=source,
        years=[2022, 2023, 2024],
        lcia_methods=methods,
        missing_extensions_by_year={2022: ("deforestation",), 2024: ("land", "deforestation")},
        include_correction_products=True,
    )
    report = process_mrio(
        source,
        years=[2022, 2023, 2024],
        lcia_method=methods,
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert report is not None
    assert report.processed == [2022, 2023, 2024]
    assert report.errors == {}
    assert report.raw_corrected_value_row_count == sum(
        len(load_raw_corrected_value_rows(source=source, year=year)) for year in (2022, 2023)
    )
    assert report.lcia_missing_by_year == {
        2022: {method: ["deforestation"] for method in methods[1:]},
        2024: {
            "pb_lcia": ["land"],
            "ef_3.1": ["deforestation", "land"],
            "ef_3.1_14cat": ["deforestation", "land"],
            "gwp100_ef31": ["deforestation"],
            "gwp100_ipcc": ["deforestation"],
        },
    }
    metadata = _read_metadata(source, matrix_version=None)
    assert metadata["source"] == source
    for year in (2022, 2023):
        assert metadata["years"][str(year)]["raw_corrected_values"]["row_count"] == len(
            load_raw_corrected_value_rows(source=source, year=year)
        )
    assert metadata["years"]["2024"].get("raw_corrected_values") is None
    assert metadata["years"]["2022"]["lcia_status"] == {
        "pb_lcia": {"available": True},
        **{method: {"available": True, "missing": ["deforestation"]} for method in methods[1:]},
    }
    assert (
        metadata["years"]["2022"]["characterization_inputs"]
        == metadata["years"]["2023"]["characterization_inputs"]
    )
    assert metadata["years"]["2023"]["lcia_status"] == {
        method: {"available": True} for method in methods
    }
    assert metadata["years"]["2024"]["lcia_status"] == {
        "pb_lcia": {"available": False, "missing": ["land"]},
        "ef_3.1": {"available": False, "missing": ["deforestation", "land"]},
        "ef_3.1_14cat": {"available": False, "missing": ["deforestation", "land"]},
        "gwp100_ef31": {"available": True, "missing": ["deforestation"]},
        "gwp100_ipcc": {"available": True, "missing": ["deforestation"]},
    }
    for year, available in ((2022, methods), (2023, methods), (2024, methods[-2:])):
        saved = _get_year_saved_dir(source, year)
        assert metadata["years"][str(year)]["enacting_metrics"]["lcia_methods"] == list(available)
        for method in available:
            matrix = pd.read_csv(
                _get_characterization_matrix_path(source_key=source, lcia_method=method)
            )
            if year != 2023:
                matrix = matrix.loc[matrix["extension"] != "deforestation"]
            factors = pd.Series(matrix.groupby("impact")["factor"].sum()).sort_index()
            extension = saved / "preclip" / "extensions" / method
            for table in ("F", "F_Y"):
                projected = pd.read_pickle(extension / f"{table}.pickle").sort_index()
                assert projected.index.tolist() == factors.index.tolist()
                expected = np.repeat(
                    factors.to_numpy(dtype=float)[:, None], projected.shape[1], axis=1
                )
                if table == "F":
                    corrections = load_raw_corrected_value_rows(source=source, year=year)
                    for _product, group in corrections.groupby(["region", "sector"]):
                        flows = pd.Series(1.0, index=matrix.index)
                        for ext, stressor, corrected in group[
                            ["extension", "stressor", "corrected_value"]
                        ].itertuples(index=False, name=None):
                            flows.loc[matrix.extension.eq(ext) & matrix.stressor.eq(stressor)] = (
                                corrected
                            )
                        corrected_impacts = pd.Series(
                            matrix.assign(factor=matrix.factor * flows)
                            .groupby("impact")["factor"]
                            .sum()
                        ).reindex(factors.index)
                        column = projected.columns.get_loc(
                            (str(group.region.iloc[0]), str(group.sector.iloc[0]))
                        )
                        expected[:, column] = corrected_impacts.to_numpy()
                np.testing.assert_allclose(
                    projected.to_numpy(dtype=float),
                    expected,
                    rtol=1e-14,
                    atol=0,
                )
    assert "computed without this extension" in " ".join(str(report).split())
    reused = process_mrio(
        source,
        years=[2022, 2023, 2024],
        lcia_method=methods,
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert reused is not None
    assert reused.skipped_already_saved == [2022, 2023, 2024]
    assert reused.lcia_missing_by_year == {
        2022: {method: ["deforestation"] for method in methods[1:]},
        2024: {method: ["deforestation"] for method in methods[-2:]},
    }


@pytest.mark.parametrize(
    "extension,other_extension", (("deforestation", None), ("Deforestation", "air emissions"))
)
def test_absent_deforestation_preserves_custom_impact_axis(
    project_repo: Path, extension: str, other_extension: str | None
) -> None:
    source = "exiobase_3112_ixi"
    method = "custom_deforestation"
    matrix_path = write_characterization_matrix(
        project_repo, source_key=source, method_name=method, extension_name=extension
    )
    matrix = pd.read_csv(matrix_path)
    if other_extension:
        other = matrix.assign(extension=other_extension, impact="aaa_other", impact_parent="other")
        other["impact_full_name"] = "other"
        matrix = pd.concat([matrix, other], ignore_index=True)
        matrix.to_csv(matrix_path, index=False)
    write_exio_archive_files(
        project_repo,
        source=source,
        years=[2022, 2023],
        lcia_methods=(method,),
        missing_extensions_by_year={2022: (extension,)},
        include_correction_products=True,
    )
    report = process_mrio(
        source,
        years=[2022, 2023],
        lcia_method=method,
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert report is not None
    assert report.lcia_missing_by_year == {2022: {method: [extension]}}
    assert "computed without this extension" in " ".join(str(report).split())
    for year in (2022, 2023):
        saved = _get_year_saved_dir(source, year) / "preclip" / "extensions" / method
        expected = matrix.assign(
            factor=matrix["factor"].where((matrix["extension"] != extension) | (year == 2023), 0)
        )
        factors = pd.Series(expected.groupby("impact")["factor"].sum()).sort_index()
        production = pd.read_pickle(saved / "F.pickle")
        demand = pd.read_pickle(saved / "F_Y.pickle")
        assert demand.index.tolist() == production.index.tolist()
        for actual in (production, demand):
            actual = actual.sort_index()
            assert actual.index.tolist() == factors.index.tolist()
            np.testing.assert_allclose(
                actual.to_numpy(), np.repeat(factors.to_numpy()[:, None], actual.shape[1], 1)
            )
    metadata = _read_metadata(source, matrix_version=None)
    entry = metadata["years"]["2022"]
    entry["lcia_status"][method]["available"] = False
    entry["enacting_metrics"]["lcia_methods"] = []
    entry["extensions"] = {}
    _write_metadata(source, metadata, matrix_version=None)
    retry = process_mrio(
        source,
        years=[2022, 2023],
        lcia_method=method,
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert retry is not None
    assert retry.processed == [2022]
    assert retry.skipped_already_saved == [2023]
    assert _read_metadata(source, matrix_version=None)["years"]["2022"]["lcia_status"] == {
        method: {"available": True, "missing": [extension]}
    }


@pytest.mark.parametrize("field,value", (("stressor", "unknown"), ("stressor_unit", "m2")))
def test_present_deforestation_keeps_stressor_and_unit_validation(
    project_repo: Path, field: str, value: str
) -> None:
    source = "exiobase_3112_ixi"
    method = "custom_deforestation"
    path = write_characterization_matrix(
        project_repo, source_key=source, method_name=method, extension_name="deforestation"
    )
    write_exio_archive_files(
        project_repo,
        source=source,
        years=[2023],
        lcia_methods=(method,),
        include_correction_products=True,
    )
    matrix = pd.read_csv(path)
    matrix[field] = value
    matrix.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="Characterization validation failed"):
        process_mrio(source, years=[2023], lcia_method=method)
