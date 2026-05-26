"""Targeted performance/equivalence workload helpers for deterministic aSoCC."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from pyaesa import deterministic_asocc

from .perf_equivalence import compare_output_files


@dataclass(frozen=True)
class WorkloadSpec:
    """One deterministic_asocc workload used for perf/equivalence checks."""

    name: str
    kwargs: dict[str, Any]


def _collect_output_files(*, root: Path) -> list[Path]:
    """Return sorted persisted output tables under one output root."""
    if not root.exists():
        return []
    allowed = {".csv", ".pickle", ".parquet"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    return sorted(files, key=lambda p: str(p.relative_to(root)).lower())


def run_workload(
    *,
    spec: WorkloadSpec,
    project_name: str,
) -> float:
    """Run one workload and return elapsed wall time in seconds."""
    kwargs = dict(spec.kwargs)
    kwargs["project_name"] = project_name
    started = perf_counter()
    deterministic_asocc(**kwargs)
    return perf_counter() - started


def compare_project_outputs(
    *,
    baseline_root: Path,
    candidate_root: Path,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> None:
    """Compare all persisted tables between baseline and candidate output trees."""
    baseline_files = _collect_output_files(root=baseline_root)
    candidate_files = _collect_output_files(root=candidate_root)
    baseline_rel = {str(path.relative_to(baseline_root)) for path in baseline_files}
    candidate_rel = {str(path.relative_to(candidate_root)) for path in candidate_files}
    if baseline_rel != candidate_rel:
        missing = sorted(baseline_rel - candidate_rel)
        extra = sorted(candidate_rel - baseline_rel)
        raise AssertionError(
            "Output file sets differ between baseline and candidate. "
            f"missing={missing[:20]} extra={extra[:20]}"
        )
    for rel_path in sorted(baseline_rel):
        compare_output_files(
            baseline_path=baseline_root / rel_path,
            candidate_path=candidate_root / rel_path,
            atol=atol,
            rtol=rtol,
        )


def default_targeted_workloads() -> list[WorkloadSpec]:
    """Return curated workloads covering key heavy and representative paths."""
    return [
        WorkloadSpec(
            name="l2cb_gwp_single_ssp",
            kwargs={
                "source": "exiobase_396_ixi",
                "agg_sec": True,
                "agg_reg": True,
                "agg_version": "eu_energy",
                "years": list(range(1995, 2051)),
                "lcia_method": "gwp100_lcia",
                "fu_code": "L2.c.b",
                "r_c": "EU",
                "s_p": "Energy",
                "l1_reg_aggreg": "pre",
                "ssp_scenario": "SSP2",
            },
        ),
        WorkloadSpec(
            name="l2cb_pb_gwp_multi_ssp",
            kwargs={
                "source": "exiobase_396_ixi",
                "agg_sec": True,
                "agg_version": "elec",
                "years": list(range(1995, 2061)),
                "lcia_method": ["pb_lcia", "gwp100_lcia"],
                "fu_code": "L2.c.b",
                "r_c": "FR",
                "s_p": "Electricity",
                "ssp_scenario": ["SSP1", "SSP2", "SSP5"],
            },
        ),
        WorkloadSpec(
            name="l1_reference_heavy",
            kwargs={
                "source": "exiobase_396_ixi",
                "agg_reg": True,
                "agg_sec": True,
                "agg_version": "eu_energy",
                "years": list(range(2000, 2031)),
                "lcia_method": ["gwp100_lcia"],
                "fu_code": "L1.a",
                "reference_years": [2000, 2010, 2020],
                "l1_methods": ["AR(E)", "PR-HR(Ecap,cum)"],
                "ssp_scenario": "SSP2",
            },
        ),
    ]
