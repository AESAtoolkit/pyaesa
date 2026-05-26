from pathlib import Path

import pandas as pd
import pytest

from pyaesa.asocc.data import load_mrio as mod
from pyaesa.asocc.data.paths import _get_mrio_year_dir


def test_axis_name_and_pickle_cover_validation_paths(tmp_path: Path) -> None:
    assert mod._axis_names(pd.MultiIndex.from_tuples([("FR", "D")], names=["r_p", "s_p"])) == [
        "r_p",
        "s_p",
    ]
    assert mod._axis_names(pd.Index(["FR"], name="r_f")) == ["r_f"]

    frame = pd.DataFrame(
        [[1.0]],
        index=pd.Index(["FR"], name="r_f"),
        columns=pd.Index(["US"], name="r_c"),
    )
    assert mod._require_names(
        frame,
        name="x_to_rc",
        index_names=["r_f"],
        column_names=["r_c"],
    ).equals(frame)

    with pytest.raises(ValueError):
        mod._require_names(
            ["FR"],
            name="demo",
            index_names=["r_f"],
        )

    with pytest.raises(ValueError):
        mod._require_names(
            pd.Series([1.0], index=pd.Index(["FR"], name="bad")),
            name="demo",
            index_names=["r_f"],
        )

    with pytest.raises(ValueError):
        mod._require_names(
            pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            name="demo",
            column_names=["r_c"],
        )

    with pytest.raises(ValueError):
        mod._require_names(
            pd.DataFrame(
                [[1.0]],
                index=pd.Index(["FR"], name="r_f"),
                columns=pd.Index(["US"], name="bad"),
            ),
            name="demo",
            column_names=["r_c"],
        )

    pickle_path = tmp_path / "demo.pickle"
    frame.to_pickle(pickle_path)
    loaded = mod._load_pickle_required(pickle_path, "demo")
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.equals(frame)

    with pytest.raises(FileNotFoundError):
        mod._load_pickle_required(tmp_path / "missing.pickle", "missing")


def test_metric_loaders_cover_success_unknown_metrics_and_invalid_stored_types(
    allocation_dummy_repo_factory,
    tmp_path: Path,
) -> None:
    allocation_dummy_repo_factory(name="load_mrio_success")
    saved_dir = _get_mrio_year_dir(
        source="oecd_v2025",
        year=2005,
        agg_version=None,
    )

    l1_payload = {
        "fd_rf": mod._load_enacting_metric_l1_metric(saved_dir, "fd_rf"),
        "gva_rp": mod._load_enacting_metric_l1_metric(saved_dir, "gva_rp"),
    }
    assert set(l1_payload) == {"fd_rf", "gva_rp"}
    assert isinstance(l1_payload["fd_rf"], pd.Series)

    l2_payload = {
        "fd_rp_sp_rf": mod._load_enacting_metric_l2_metric(saved_dir, "fd_rp_sp_rf"),
        "fd_rp_sp": mod._load_enacting_metric_l2_metric(saved_dir, "fd_rp_sp"),
        "fd_rf_sp": mod._load_enacting_metric_l2_metric(saved_dir, "fd_rf_sp"),
        "gva_rp_sp": mod._load_enacting_metric_l2_metric(saved_dir, "gva_rp_sp"),
    }
    assert set(l2_payload) == {"fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp", "gva_rp_sp"}
    assert isinstance(l2_payload["fd_rp_sp_rf"], pd.DataFrame)
    assert isinstance(l2_payload["fd_rp_sp"], pd.Series)

    utility_payload = {
        "x_to_rc": mod._load_utility_metric(saved_dir, "x_to_rc"),
        "kappa": mod._load_utility_metric(saved_dir, "kappa"),
        "omega_reg": mod._load_utility_metric(saved_dir, "omega_reg"),
    }
    assert set(utility_payload) == {"x_to_rc", "kappa", "omega_reg"}
    assert isinstance(utility_payload["omega_reg"], pd.DataFrame)

    assert mod._years_from_metadata(source="oecd_v2025", agg_version=None) == [2005, 2006]

    invalid_saved_dir = tmp_path / "invalid_saved_dir"
    (invalid_saved_dir / "enacting_metrics" / "level_1").mkdir(parents=True, exist_ok=True)
    (invalid_saved_dir / "enacting_metrics" / "level_2").mkdir(parents=True, exist_ok=True)
    (invalid_saved_dir / "utility_propag_uncasext").mkdir(parents=True, exist_ok=True)
    (invalid_saved_dir / "enacting_metrics" / "level_1" / "gwp100_lcia").mkdir(
        parents=True,
        exist_ok=True,
    )
    (invalid_saved_dir / "enacting_metrics" / "level_2" / "gwp100_lcia").mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [[1.0]],
        index=pd.Index(["FR"], name="r_f"),
        columns=pd.Index(["demo"], name="unused"),
    ).to_pickle(invalid_saved_dir / "enacting_metrics" / "level_1" / "fd_rf.pickle")
    with pytest.raises(ValueError):
        mod._load_enacting_metric_l1_metric(invalid_saved_dir, "fd_rf")

    pd.DataFrame(
        [[1.0]],
        index=pd.MultiIndex.from_tuples([("FR", "D")], names=["r_p", "s_p"]),
        columns=pd.Index(["FR"], name="r_f"),
    ).to_pickle(invalid_saved_dir / "enacting_metrics" / "level_2" / "fd_rp_sp.pickle")
    with pytest.raises(ValueError):
        mod._load_enacting_metric_l2_metric(invalid_saved_dir, "fd_rp_sp")

    pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("FR", "D")], names=["r_p", "s_p"]),
    ).to_pickle(invalid_saved_dir / "enacting_metrics" / "level_2" / "fd_rp_sp_rf.pickle")
    with pytest.raises(ValueError):
        mod._load_enacting_metric_l2_metric(invalid_saved_dir, "fd_rp_sp_rf")

    pd.Series([1.0], index=pd.Index(["climate_child"], name="impact")).to_pickle(
        invalid_saved_dir / "enacting_metrics" / "level_1" / "gwp100_lcia" / "e_cba_fd_reg.pickle"
    )
    with pytest.raises(ValueError):
        mod._load_lcia_l1_metric(invalid_saved_dir, "gwp100_lcia", "e_cba_fd_reg")

    pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["climate_child"], name="impact"),
        columns=pd.MultiIndex.from_tuples([("FR", "D"), ("US", "X")], names=["bad", "s_p"]),
    ).to_pickle(
        invalid_saved_dir / "enacting_metrics" / "level_2" / "gwp100_lcia" / "e_pba_rp_sp.pickle"
    )
    with pytest.raises(ValueError):
        mod._load_lcia_l2_metric(invalid_saved_dir, "gwp100_lcia", "e_pba_rp_sp")


def test_metric_to_series_covers_series_frame_and_multiindex_paths() -> None:
    direct_series = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [("climate_child", "FR"), ("climate_child", "US")],
            names=["impact", "r_f"],
        ),
    )
    assert mod._metric_to_series("e_cba_fd_reg", direct_series).equals(direct_series)

    single_level_frame = pd.DataFrame(
        [[3.0, 4.0]],
        index=pd.Index(["climate_child"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )
    stacked_single_level = mod._metric_to_series("e_cba_fd_reg", single_level_frame)
    assert stacked_single_level.index.names == ["impact", "r_f"]
    assert stacked_single_level.loc[("climate_child", "FR")] == pytest.approx(3.0)

    multi_level_frame = pd.DataFrame(
        [[5.0, 6.0, 7.0, 8.0]],
        index=pd.Index(["climate_child"], name="impact"),
        columns=pd.MultiIndex.from_tuples(
            [("FR", "D"), ("FR", "X"), ("US", "D"), ("US", "X")],
            names=["r_p", "s_p"],
        ),
    )
    stacked_multi_level = mod._metric_to_series("e_pba_rp_sp", multi_level_frame)
    assert stacked_multi_level.index.names == ["impact", "r_p", "s_p"]
    assert stacked_multi_level.loc[("climate_child", "FR", "D")] == pytest.approx(5.0)

    unknown_metric = mod._metric_to_series(
        "unknown_metric",
        pd.Series([9.0], index=pd.Index(["FR"], name="region")),
    )
    assert unknown_metric.index.names == ["region"]

    with pytest.raises(ValueError):
        mod._metric_to_series(
            "e_pba_rp_sp",
            pd.DataFrame(
                [[5.0, 6.0]],
                index=pd.Index(["climate_child"], name="impact"),
                columns=pd.MultiIndex.from_tuples(
                    [("FR", "D"), ("US", "X")],
                    names=["bad", "s_p"],
                ),
            ),
        )
