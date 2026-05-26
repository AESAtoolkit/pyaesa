import pandas as pd
import pytest

from pyaesa.io_lca.compute.main_results import (
    _apply_selector_slices,
    _resolve_unit_map_for_impacts,
    _stack_to_long,
    build_main_results_rows,
)
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec, resolve_fu_spec
from pyaesa.io_lca.data.loaders import (
    YearMethodMainPayload,
    load_domain_metadata,
    load_main_payload,
)


def _load_main_payload_for_tests(io_lca_dummy_repo) -> tuple[YearMethodMainPayload, IOLCAFUSpec]:
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    spec = resolve_fu_spec(fu_code="L1.a")
    payload, unavailable_reason = load_main_payload(
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
    assert payload is not None
    return payload, spec


def test_main_result_contracts_cover_stack_slices_and_unit_resolution(
    io_lca_dummy_repo,
) -> None:
    stacked = _stack_to_long(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index(["AAL"], name="impact"),
            columns=pd.MultiIndex.from_tuples(
                [("FR", "detail_a"), ("DE", "detail_b")],
                names=["r_f", "detail"],
            ),
        )
    )
    assert stacked.columns.tolist() == ["impact", "r_f", "detail", "lca_value"]
    assert stacked["lca_value"].tolist() == [1.0, 2.0]

    payload, spec = _load_main_payload_for_tests(io_lca_dummy_repo)
    unchanged = _apply_selector_slices(
        frame=payload.metric,
        spec=spec,
        filters={"r_f": None},
    )
    assert unchanged.equals(payload.metric)

    sliced = _apply_selector_slices(
        frame=payload.metric,
        spec=spec,
        filters={"r_f": ["FR"]},
    )
    assert sliced.columns.tolist() == ["FR"]

    resolved_units = _resolve_unit_map_for_impacts(
        impacts=pd.Series(["AAL", "BI FD GHG"], dtype=str),
        unit_by_impact=payload.unit_by_impact,
    )
    assert resolved_units["AAL"] == "kg"
    assert resolved_units["BI FD GHG"] == "kg"

    with pytest.raises(ValueError):
        _resolve_unit_map_for_impacts(
            impacts=pd.Series(["AAL", "unknown_impact"], dtype=str),
            unit_by_impact=payload.unit_by_impact,
        )


def test_build_main_results_rows_covers_empty_and_success(
    io_lca_dummy_repo,
) -> None:
    payload, spec = _load_main_payload_for_tests(io_lca_dummy_repo)

    rows = build_main_results_rows(
        payload=payload,
        spec=spec,
        filters={"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": None},
    )
    assert rows.columns.tolist() == [
        "lcia_method",
        "year",
        "impact",
        "r_f",
        "lca_value",
        "impact_unit",
    ]
    assert rows["r_f"].tolist() == ["FR", "FR"]
    assert rows["impact"].tolist() == ["AAL", "BI FD"]

    zero_payload = YearMethodMainPayload(
        year=payload.year,
        lcia_method=payload.lcia_method,
        saved_dir=payload.saved_dir,
        year_entry=payload.year_entry,
        metric=pd.DataFrame(
            [[0.0, 0.0], [1.0, 2.0]],
            index=pd.Index(["AAL", "BI FD"], name="impact"),
            columns=pd.Index(["FR", "DE"], name="r_f"),
        ),
        unit_by_impact=payload.unit_by_impact,
    )
    with pytest.raises(ValueError):
        build_main_results_rows(
            payload=zero_payload,
            spec=spec,
            filters={"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": None},
        )

    multi_index_zero_payload = YearMethodMainPayload(
        year=payload.year,
        lcia_method=payload.lcia_method,
        saved_dir=payload.saved_dir,
        year_entry=payload.year_entry,
        metric=pd.DataFrame(
            [[0.0, 0.0], [1.0, 2.0]],
            index=pd.MultiIndex.from_tuples(
                [("AAL", "FR"), ("BI FD", "FR")],
                names=["impact", "r_p"],
            ),
            columns=pd.Index(["FR", "DE"], name="r_f"),
        ),
        unit_by_impact=payload.unit_by_impact,
    )
    with pytest.raises(ValueError):
        build_main_results_rows(
            payload=multi_index_zero_payload,
            spec=spec,
            filters={"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": None},
        )

    empty_payload = YearMethodMainPayload(
        year=payload.year,
        lcia_method=payload.lcia_method,
        saved_dir=payload.saved_dir,
        year_entry=payload.year_entry,
        metric=pd.DataFrame(
            index=pd.Index([], name="impact"),
            columns=pd.Index([], name="r_f"),
        ),
        unit_by_impact=payload.unit_by_impact,
    )
    empty_rows = build_main_results_rows(
        payload=empty_payload,
        spec=spec,
        filters={"r_f": None},
    )
    assert empty_rows.empty
    assert empty_rows.columns.tolist() == [
        "lcia_method",
        "year",
        "impact",
        "r_f",
        "lca_value",
        "impact_unit",
    ]
