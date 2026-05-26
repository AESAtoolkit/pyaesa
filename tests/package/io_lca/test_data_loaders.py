from pathlib import Path

import pandas as pd
import pytest

from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec, resolve_fu_spec
from pyaesa.io_lca.data.loaders import (
    _align_matrix_axis_names_to_driver,
    _load_fy_matrix,
    impact_unit_text_map,
    load_domain_metadata,
    load_io_lca_method_table,
    load_main_payload,
    load_upstream_payload,
)
from pyaesa.process.mrios.utils.io.paths import _get_year_saved_dir, _get_year_saved_path


def _write_processed_year_payload(
    *,
    saved_dir: Path,
    lcia_method: str,
    metric_key: str,
    metric_frame: pd.DataFrame,
    fy_frame: pd.DataFrame | None = None,
) -> None:
    products = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("DE", "B")],
        names=["r_p", "s_p"],
    )
    impacts = pd.Index(["AAL", "BI FD GHG"], name="impact")

    (saved_dir / "enacting_metrics" / "level_1" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "enacting_metrics" / "level_2" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "extensions" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "utility_propag_uncasext").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        index=products,
        columns=products,
    ).to_pickle(saved_dir / "A.pickle")
    pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=products,
        columns=products,
    ).to_pickle(saved_dir / "L.pickle")
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=impacts,
        columns=products,
    ).to_pickle(saved_dir / "extensions" / lcia_method / "S.pickle")
    pd.DataFrame(
        [[10.0], [20.0]],
        index=products,
        columns=pd.Index(["RC"], name="r_c"),
    ).to_pickle(saved_dir / "utility_propag_uncasext" / "x_to_rc.pickle")

    metric_level = "level_1" if metric_key in {"e_cba_fd_reg", "e_pba_reg"} else "level_2"
    metric_frame.to_pickle(
        saved_dir / "enacting_metrics" / metric_level / lcia_method / f"{metric_key}.pickle"
    )

    if fy_frame is not None:
        fy_frame.to_pickle(saved_dir / "enacting_metrics" / "level_1" / lcia_method / "F_Y.pickle")


def test_impact_unit_text_map_normalizes_index_values() -> None:
    units = pd.Series({"AAL": "kg", 1: 2})

    assert impact_unit_text_map(unit_by_impact=units) == {"AAL": "kg", "1": "2"}


def test_load_main_payload_covers_available_and_missing_contract_paths(
    io_lca_dummy_repo,
) -> None:
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    spec = resolve_fu_spec(fu_code="L1.a")

    available_payload, unavailable_reason = load_main_payload(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata=metadata,
        metadata_path=metadata_path,
        year=2019,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )
    assert unavailable_reason is None
    assert available_payload is not None
    assert available_payload.metric.columns.tolist() == ["FR", "DE"]
    assert available_payload.unit_by_impact.to_dict() == {"AAL": "kg", "BI FD": "kg"}

    skipped_payload, skipped_reason = load_main_payload(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata=metadata,
        metadata_path=metadata_path,
        year=2020,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )
    assert skipped_payload is None
    assert skipped_reason == "extension missing"

    missing_dir_candidates = [
        (io_lca_dummy_repo.source, 2019, "missing_aggregate"),
        (io_lca_dummy_repo.source, 2019, "missing_agg_alt"),
    ]
    missing_source, missing_year, missing_agg_version = next(
        candidate
        for candidate in missing_dir_candidates
        if not _get_year_saved_path(
            candidate[0],
            candidate[1],
            matrix_version=candidate[2],
        ).exists()
    )
    with pytest.raises(ValueError):
        load_main_payload(
            source=missing_source,
            agg_version=missing_agg_version,
            agg_reg=False,
            agg_sec=False,
            metadata=metadata,
            metadata_path=metadata_path,
            year=missing_year,
            lcia_method=io_lca_dummy_repo.lcia_method,
            fu_spec=spec,
        )

    missing_metric_year = 2022
    missing_metric_saved_dir = _get_year_saved_dir(
        io_lca_dummy_repo.source,
        missing_metric_year,
        matrix_version=None,
    )
    missing_metric_saved_dir.mkdir(parents=True, exist_ok=True)
    missing_metric_metadata = dict(metadata)
    missing_metric_years = dict(metadata["years"])
    missing_metric_years[str(missing_metric_year)] = dict(metadata["years"]["2019"])
    missing_metric_metadata["years"] = missing_metric_years
    with pytest.raises(ValueError):
        load_main_payload(
            source=io_lca_dummy_repo.source,
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
            metadata=missing_metric_metadata,
            metadata_path=metadata_path,
            year=missing_metric_year,
            lcia_method=io_lca_dummy_repo.lcia_method,
            fu_spec=spec,
        )


def test_load_main_payload_rejects_missing_year_entry(io_lca_dummy_repo) -> None:
    _metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    spec = resolve_fu_spec(fu_code="L1.a")

    with pytest.raises(ValueError):
        load_main_payload(
            source=io_lca_dummy_repo.source,
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
            metadata={"years": {}},
            metadata_path=metadata_path,
            year=2019,
            lcia_method=io_lca_dummy_repo.lcia_method,
            fu_spec=spec,
        )


def test_load_main_payload_covers_l2_metric_branch(io_lca_dummy_repo) -> None:
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    year = 2024
    saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, year, matrix_version=None)
    saved_dir.mkdir(parents=True, exist_ok=True)

    metric_frame = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("DE", "B")],
            names=["r_p", "s_p"],
        ),
    )
    _write_processed_year_payload(
        saved_dir=saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        metric_key="e_cba_fd_rp_sp",
        metric_frame=metric_frame,
    )

    year_entry = dict(metadata["years"]["2019"])
    year_entry["lcia_status"] = {
        io_lca_dummy_repo.lcia_method: {"available": True},
    }
    metadata_for_year = dict(metadata)
    metadata_for_year["years"] = dict(metadata["years"])
    metadata_for_year["years"][str(year)] = year_entry

    spec = IOLCAFUSpec(
        fu_code="manual_l2",
        level="L2",
        family="fd",
        lcia_matrix_key="e_cba_fd_rp_sp",
        selector_axes=("r_p", "s_p"),
        upstream_driver="x_to_rc",
        upstream_supported=True,
        fy_relevant=False,
    )
    payload, unavailable_reason = load_main_payload(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata=metadata_for_year,
        metadata_path=metadata_path,
        year=year,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )
    assert unavailable_reason is None
    assert payload is not None
    assert payload.metric.columns.names == ["r_p", "s_p"]
    assert payload.metric.loc["AAL", ("FR", "A")] == pytest.approx(1.0)


def test_align_matrix_axis_names_to_driver_leaves_unmatched_axes_unchanged() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["row_a", "row_b"], name="row"),
        columns=pd.Index(["col_a", "col_b"], name="col"),
    )
    driver_index = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("DE", "B")],
        names=["r_p", "s_p"],
    )

    aligned = _align_matrix_axis_names_to_driver(frame=frame, driver_index=driver_index)

    assert aligned.index.name == "row"
    assert aligned.columns.name == "col"

    plain_driver = pd.Index(["FR", "DE"], name="r_p")
    plain_aligned = _align_matrix_axis_names_to_driver(frame=frame, driver_index=plain_driver)
    assert plain_aligned.index.name == "row"
    assert plain_aligned.columns.name == "col"


def test_load_upstream_payload_covers_x_to_rc_and_multiindex_fy_aggregation(
    io_lca_dummy_repo,
) -> None:
    year = 2025
    saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, year, matrix_version=None)
    saved_dir.mkdir(parents=True, exist_ok=True)

    metric_index = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("DE", "B")],
        names=["r_p", "s_p"],
    )
    metric_columns = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("DE", "B")],
        names=["r_p", "s_p"],
    )
    fy_columns = pd.MultiIndex.from_tuples(
        [("FR", "left"), ("FR", "right"), ("DE", "left"), ("DE", "right")],
        names=["region", "component"],
    )
    (saved_dir / "extensions" / io_lca_dummy_repo.lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "enacting_metrics" / "level_1" / io_lca_dummy_repo.lcia_method).mkdir(
        parents=True, exist_ok=True
    )
    (saved_dir / "utility_propag_uncasext").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        index=metric_index,
        columns=metric_columns,
    ).to_pickle(saved_dir / "A.pickle")
    pd.DataFrame(
        [[4.0, 5.0], [6.0, 7.0]],
        index=metric_index,
        columns=metric_columns,
    ).to_pickle(saved_dir / "L.pickle")
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("DE", "B")],
            names=["r_p", "s_p"],
        ),
    ).to_pickle(saved_dir / "extensions" / io_lca_dummy_repo.lcia_method / "S.pickle")
    pd.DataFrame(
        [[10.0], [20.0]],
        index=metric_index,
        columns=pd.Index(["RC"], name="r_c"),
    ).to_pickle(saved_dir / "utility_propag_uncasext" / "x_to_rc.pickle")
    pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=fy_columns,
    ).to_pickle(
        saved_dir / "enacting_metrics" / "level_1" / io_lca_dummy_repo.lcia_method / "F_Y.pickle"
    )

    spec = resolve_fu_spec(fu_code="L1.b")
    payload = load_upstream_payload(
        source=io_lca_dummy_repo.source,
        saved_dir=saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )

    assert payload.driver_matrix.columns.name == "r_c"
    assert payload.a_matrix.index.names == ["r_p", "s_p"]
    assert payload.a_matrix.columns.names == ["r_p", "s_p"]
    assert payload.s_matrix.index.tolist() == ["AAL", "BI FD"]
    assert payload.fy_matrix is not None
    assert payload.fy_matrix.columns.name == "region"
    assert payload.fy_matrix.loc["AAL", "FR"] == pytest.approx(3.0)
    assert payload.fy_matrix.loc["BI FD", "DE"] == pytest.approx(15.0)


def test_load_upstream_payload_covers_y_fd_driver_without_fy_relevance(
    io_lca_dummy_repo,
) -> None:
    year = 2027
    saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, year, matrix_version=None)
    saved_dir.mkdir(parents=True, exist_ok=True)

    metric_index = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("DE", "B")],
        names=["r_p", "s_p"],
    )
    metric_columns = pd.Index(["F1"], name="r_f")
    (saved_dir / "extensions" / io_lca_dummy_repo.lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "enacting_metrics" / "level_1" / io_lca_dummy_repo.lcia_method).mkdir(
        parents=True, exist_ok=True
    )
    (saved_dir / "enacting_metrics" / "level_2").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        index=metric_index,
        columns=metric_index,
    ).to_pickle(saved_dir / "A.pickle")
    pd.DataFrame(
        [[4.0, 5.0], [6.0, 7.0]],
        index=metric_index,
        columns=metric_index,
    ).to_pickle(saved_dir / "L.pickle")
    pd.DataFrame(
        [[1.0], [2.0]],
        index=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("DE", "B")],
            names=["r_p", "s_p"],
        ),
        columns=metric_columns,
    ).to_pickle(saved_dir / "enacting_metrics" / "level_2" / "fd_rp_sp_rf.pickle")
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("DE", "B")],
            names=["r_p", "s_p"],
        ),
    ).to_pickle(saved_dir / "extensions" / io_lca_dummy_repo.lcia_method / "S.pickle")

    spec = IOLCAFUSpec(
        fu_code="manual_y_fd",
        level="L2",
        family="fd",
        lcia_matrix_key="e_cba_fd_rp_sp_rf",
        selector_axes=("r_p", "s_p", "r_f"),
        upstream_driver="y_fd",
        upstream_supported=True,
        fy_relevant=False,
    )
    payload = load_upstream_payload(
        source=io_lca_dummy_repo.source,
        saved_dir=saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )

    assert payload.driver_matrix.columns.name == "r_f"
    assert payload.fy_matrix is None


def test_load_fy_matrix_rejects_unresolvable_multiindex_columns(io_lca_dummy_repo) -> None:
    saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, 2026, matrix_version=None)
    saved_dir.mkdir(parents=True, exist_ok=True)
    fy_path = (
        saved_dir / "enacting_metrics" / "level_1" / io_lca_dummy_repo.lcia_method / "F_Y.pickle"
    )
    fy_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR", "left"), ("DE", "right")],
            names=["foo", "bar"],
        ),
    ).to_pickle(fy_path)

    with pytest.raises(ValueError):
        _load_fy_matrix(
            saved_dir=saved_dir,
            lcia_method=io_lca_dummy_repo.lcia_method,
            selected_axis_name=None,
        )


def test_load_fy_matrix_covers_missing_file_and_single_level_multiindex_columns(
    io_lca_dummy_repo,
) -> None:
    missing_saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, 2028, matrix_version=None)
    missing_saved_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        _load_fy_matrix(
            saved_dir=missing_saved_dir,
            lcia_method=io_lca_dummy_repo.lcia_method,
            selected_axis_name=None,
        )

    single_level_saved_dir = _get_year_saved_dir(
        io_lca_dummy_repo.source,
        2029,
        matrix_version=None,
    )
    single_level_saved_dir.mkdir(parents=True, exist_ok=True)
    fy_path = (
        single_level_saved_dir
        / "enacting_metrics"
        / "level_1"
        / io_lca_dummy_repo.lcia_method
        / "F_Y.pickle"
    )
    fy_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR",), ("DE",)],
            names=["foo"],
        ),
    ).to_pickle(fy_path)

    fy_matrix = _load_fy_matrix(
        saved_dir=single_level_saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        selected_axis_name=None,
    )
    assert fy_matrix.columns.name == "region"
    assert fy_matrix.loc["AAL", "FR"] == pytest.approx(1.0)
    assert fy_matrix.loc["BI FD GHG", "DE"] == pytest.approx(4.0)

    plain_saved_dir = _get_year_saved_dir(io_lca_dummy_repo.source, 2030, matrix_version=None)
    plain_saved_dir.mkdir(parents=True, exist_ok=True)
    plain_fy_path = (
        plain_saved_dir
        / "enacting_metrics"
        / "level_1"
        / io_lca_dummy_repo.lcia_method
        / "F_Y.pickle"
    )
    plain_fy_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[5.0, 6.0], [7.0, 8.0]],
        index=pd.Index(["AAL", "BI FD GHG"], name="impact"),
        columns=pd.Index(["FR", "DE"], name="region"),
    ).to_pickle(plain_fy_path)

    plain_fy_matrix = _load_fy_matrix(
        saved_dir=plain_saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        selected_axis_name=None,
    )
    assert plain_fy_matrix.columns.name == "region"
    assert plain_fy_matrix.loc["AAL", "FR"] == pytest.approx(5.0)
    assert plain_fy_matrix.loc["BI FD GHG", "DE"] == pytest.approx(8.0)


def test_load_io_lca_method_table_covers_supported_extensions_and_errors(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})

    csv_path = tmp_path / "table.csv"
    frame.to_csv(csv_path, index=False)
    assert load_io_lca_method_table(path=csv_path).equals(frame)

    pickle_path = tmp_path / "table.pickle"
    frame.to_pickle(pickle_path)
    assert load_io_lca_method_table(path=pickle_path).equals(frame)

    parquet_path = tmp_path / "table.parquet"
    frame.to_parquet(parquet_path, index=False)
    assert load_io_lca_method_table(path=parquet_path).equals(frame)
