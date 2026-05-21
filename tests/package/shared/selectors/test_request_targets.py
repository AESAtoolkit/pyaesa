from pathlib import Path

from pyaesa.shared.selectors import request_targets as targets_mod
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args


def test_request_target_contracts_cover_asocc_merge_and_io_lca_selection(
    project_repo: Path,
) -> None:
    del project_repo

    normalized = normalize_base_allocate_args(
        {
            "project_name": "demo",
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "lcia_method": "gwp100_lcia",
            "years": [2005],
        }
    )

    merged_selector = targets_mod.build_asocc_target_selector(
        base_asocc_args=normalized,
        external_method={"one_step_methods": ["AR(E^{CBA_FD})"]},
    )
    assert merged_selector["years"] == [2005]
    assert merged_selector["methods"] == ["AR(E^{CBA_FD})", "UT(FD)"]

    native_selector = targets_mod.build_asocc_target_selector(base_asocc_args=normalized)
    assert native_selector["methods"] == ["UT(FD)"]

    pair_selector = targets_mod.build_asocc_target_selector(
        base_asocc_args=normalize_base_allocate_args(
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "method_plan": "pairs",
                "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FD)"],
                "lcia_method": "gwp100_lcia",
                "years": [2005],
            }
        )
    )
    assert pair_selector["lcia_method"] == ["gwp100_lcia"]
    assert pair_selector["methods"] == ["AR(E^{CBA_FD})_UT(FD)"]

    io_selector = targets_mod.build_io_lca_target_selector(
        base_io_lca_args={"years": [2030, 2031], "lcia_method": ["pb_lcia"]}
    )
    assert io_selector == {"years": [2030, 2031], "methods": ["pb_lcia"]}
    assert targets_mod.build_io_lca_target_selector(base_io_lca_args={}) == {}
