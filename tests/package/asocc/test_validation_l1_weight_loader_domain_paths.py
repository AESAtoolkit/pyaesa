from pathlib import Path

import pandas as pd

from pyaesa.asocc.runtime.scope.branch_resolution import (
    asocc_l1_dir,
    build_asocc_deterministic_path_scope,
    outputs_project_root,
)
from tests.allocation_equation_validation.validation_function.utils.l2_two_step_helpers import (
    _L1_WEIGHTS_CACHE,
    L1RegionWeightRequest,
    load_l1_region_weights,
)
from tests.allocation_equation_validation.validation_function.utils.workflow_helpers import (
    validation_project_name,
)


def _write_l1_csv(
    *,
    request: L1RegionWeightRequest,
    fu_code: str,
    axis_col: str,
) -> None:
    project_name = validation_project_name(
        base_project_name=request.validation_project_name_root,
        source=request.source,
        fu_code=fu_code,
        l1_reg_aggreg=request.l1_mode,
    )
    path_scope = build_asocc_deterministic_path_scope(
        proj_base=outputs_project_root(project_name=project_name),
        source_label=request.source,
        group_version=request.group_version,
    )
    out_dir = asocc_l1_dir(scope=path_scope, lcia_sub=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "l1_method": [request.l1_method, request.l1_method],
            axis_col: ["A", "B"],
            str(request.year): [0.4, 0.6],
        }
    )
    df.to_csv(out_dir / f"l1_{request.l1_method}.csv", index=False)


def test_load_l1_region_weights_reads_original_classification_domain(project_repo: Path) -> None:
    del project_repo
    request = L1RegionWeightRequest(
        validation_project_name_root="validation_demo",
        source="exiobase_396_ixi",
        group_version=None,
        group_reg=False,
        aggreg_indices=False,
        l1_mode="pre",
        output_format="csv",
        l1_method="EG(Pop)",
        year=2019,
        impact=None,
        reference_year=None,
    )
    _write_l1_csv(
        request=request,
        fu_code="L1.b",
        axis_col="r_p",
    )
    _L1_WEIGHTS_CACHE.clear()
    weights = load_l1_region_weights(request)
    assert weights is not None
    assert float(weights.sum(min_count=1)) == 1.0
    assert list(weights.index) == ["A", "B"]


def test_load_l1_region_weights_reads_grouped_domain(project_repo: Path) -> None:
    del project_repo
    request = L1RegionWeightRequest(
        validation_project_name_root="validation_demo",
        source="exiobase_396_ixi",
        group_version="elec",
        group_reg=False,
        aggreg_indices=False,
        l1_mode="pre",
        output_format="csv",
        l1_method="EG(Pop)",
        year=2019,
        impact=None,
        reference_year=None,
    )
    _write_l1_csv(
        request=request,
        fu_code="L1.b",
        axis_col="r_p",
    )
    _L1_WEIGHTS_CACHE.clear()
    weights = load_l1_region_weights(request)
    assert weights is not None
    assert float(weights.sum(min_count=1)) == 1.0


def test_load_l1_region_weights_ignores_domainless_layout_for_mrio(
    project_repo: Path,
) -> None:
    del project_repo
    request = L1RegionWeightRequest(
        validation_project_name_root="validation_demo",
        source="exiobase_396_ixi",
        group_version=None,
        group_reg=False,
        aggreg_indices=False,
        l1_mode="pre",
        output_format="csv",
        l1_method="EG(Pop)",
        year=2019,
        impact=None,
        reference_year=None,
    )
    _L1_WEIGHTS_CACHE.clear()
    weights = load_l1_region_weights(request)
    assert weights is None


def test_load_l1_region_weights_iso3_uses_no_domain_layout(project_repo: Path) -> None:
    del project_repo
    request = L1RegionWeightRequest(
        validation_project_name_root="validation_demo",
        source="iso3",
        group_version=None,
        group_reg=False,
        aggreg_indices=False,
        l1_mode="pre",
        output_format="csv",
        l1_method="EG(Pop)",
        year=2019,
        impact=None,
        reference_year=None,
    )
    _write_l1_csv(
        request=request,
        fu_code="L1.a",
        axis_col="r_f",
    )
    _L1_WEIGHTS_CACHE.clear()
    weights = load_l1_region_weights(request)
    assert weights is not None
    assert float(weights.sum(min_count=1)) == 1.0
