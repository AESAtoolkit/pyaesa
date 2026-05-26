"""Sobol public parameter normalization and method constants."""

from dataclasses import dataclass
from typing import Any

SOBOL_TARGETS: tuple[str, ...] = ("S1", "ST")
DEFAULT_SOBOL_BASE_SAMPLES = 128
DEFAULT_SOBOL_MAX_BASE_SAMPLES = 1048576
DEFAULT_SOBOL_RTOL = 0.05
DEFAULT_SOBOL_ABS_TOL = 0.01
DEFAULT_SOBOL_SCALE_FLOOR = 0.05
DEFAULT_SOBOL_CONFIDENCE_LEVEL = 0.95
DEFAULT_SOBOL_CONFIDENCE_RESAMPLES = 100


@dataclass(frozen=True)
class SobolPlan:
    """Normalized optional Sobol uncertainty plan."""

    enabled: bool
    mode: str
    n_base_samples: int
    max_base_samples: int
    rtol: float
    sobol_years: tuple[int, ...] | None = None

    @property
    def abs_tol(self) -> float:
        """Absolute confidence half width tolerance for Sobol convergence."""
        return DEFAULT_SOBOL_ABS_TOL

    @property
    def scale_floor(self) -> float:
        """Minimum relative tolerance scale for near zero Sobol indices."""
        return DEFAULT_SOBOL_SCALE_FLOOR

    @property
    def confidence_level(self) -> float:
        """Confidence level used for Sobol bootstrap half widths."""
        return DEFAULT_SOBOL_CONFIDENCE_LEVEL

    @property
    def confidence_resamples(self) -> int:
        """Bootstrap resample count used for Sobol confidence half widths."""
        return DEFAULT_SOBOL_CONFIDENCE_RESAMPLES


def sobol_plan_payload(*, plan: SobolPlan) -> dict[str, object]:
    """Return the exact normalized Sobol request and method identity."""
    payload: dict[str, object] = {
        "mode": plan.mode,
        "n_base_samples": int(plan.n_base_samples),
        "max_base_samples": int(plan.max_base_samples),
        "rtol": float(plan.rtol),
        "abs_tol": float(plan.abs_tol),
        "scale_floor": float(plan.scale_floor),
        "convergence_targets": list(SOBOL_TARGETS),
        "confidence_level": float(plan.confidence_level),
        "confidence_resamples": int(plan.confidence_resamples),
    }
    if plan.sobol_years is not None:
        payload["sobol_years"] = list(plan.sobol_years)
    return payload


def selected_sobol_output_years(
    *,
    plan: SobolPlan,
    available_years: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the studied output years selected for yearly Sobol targets."""
    studied = studied_output_years(available_years)
    if plan.sobol_years is not None:
        requested = set(plan.sobol_years)
        return tuple(year for year in studied if year in requested)
    return tuple(dict.fromkeys((studied[0], studied[-1])))


def studied_output_years(years: int | list[int] | tuple[int, ...] | range) -> tuple[int, ...]:
    """Return sorted unique studied output years from a public year selector."""
    if isinstance(years, int):
        return (int(years),)
    return tuple(sorted({int(year) for year in years}))


def normalize_sobol_plan(
    *,
    sobol_parameters: dict[str, Any] | None,
    available_years: list[int] | tuple[int, ...] | range | None = None,
) -> SobolPlan:
    """Normalize optional public Sobol parameters."""
    if sobol_parameters is None:
        return SobolPlan(
            enabled=False,
            mode="fixed",
            n_base_samples=0,
            max_base_samples=0,
            rtol=DEFAULT_SOBOL_RTOL,
        )
    params = dict(sobol_parameters)
    active = params.pop("active", True)
    if not isinstance(active, bool):
        raise ValueError("sobol_parameters.active must be a boolean.")
    if not active:
        unknown = sorted(set(params) - {"fixed", "convergence", "sobol_years"})
        if unknown:
            raise ValueError(f"Unsupported Sobol parameter(s): {unknown}.")
        return SobolPlan(
            enabled=False,
            mode="convergence",
            n_base_samples=0,
            max_base_samples=0,
            rtol=DEFAULT_SOBOL_RTOL,
        )
    sobol_years = _optional_years(params.pop("sobol_years", None))
    _validate_sobol_year_membership(
        sobol_years=sobol_years,
        available_years=available_years,
    )
    fixed_params, convergence_params = _normalize_sobol_mode_blocks(params)
    active_modes = tuple(
        mode
        for mode, block in (("fixed", fixed_params), ("convergence", convergence_params))
        if bool(block["active"])
    )
    if len(active_modes) != 1:
        raise ValueError("Exactly one Sobol mode block must be active.")
    mode = active_modes[0]
    fixed_n = _positive_power_of_two(fixed_params["n_base_samples"], field="fixed.n_base_samples")
    max_base_samples = _positive_power_of_two(
        convergence_params["max_base_samples"],
        field="convergence.max_base_samples",
    )
    convergence_n = min(DEFAULT_SOBOL_BASE_SAMPLES, max_base_samples)
    n_base_samples = convergence_n if mode == "convergence" else fixed_n
    rtol = _positive_float(convergence_params["rtol"], field="convergence.rtol")
    if mode == "fixed" and max_base_samples < n_base_samples:
        raise ValueError(
            "sobol_parameters.convergence.max_base_samples must be greater than or equal to "
            "the active n_base_samples."
        )
    if params:
        raise ValueError(f"Unsupported Sobol parameter(s): {sorted(params)}.")
    return SobolPlan(
        enabled=True,
        mode=mode,
        n_base_samples=n_base_samples,
        max_base_samples=max_base_samples,
        rtol=rtol,
        sobol_years=sobol_years,
    )


def _normalize_sobol_mode_blocks(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = _normalize_sobol_mode_block(
        params.pop(
            "fixed",
            {
                "active": False,
                "n_base_samples": DEFAULT_SOBOL_BASE_SAMPLES,
            },
        ),
        mode="fixed",
        defaults={
            "active": False,
            "n_base_samples": DEFAULT_SOBOL_BASE_SAMPLES,
        },
    )
    convergence = _normalize_sobol_mode_block(
        params.pop(
            "convergence",
            {
                "active": True,
                "max_base_samples": DEFAULT_SOBOL_MAX_BASE_SAMPLES,
                "rtol": DEFAULT_SOBOL_RTOL,
            },
        ),
        mode="convergence",
        defaults={
            "active": True,
            "max_base_samples": DEFAULT_SOBOL_MAX_BASE_SAMPLES,
            "rtol": DEFAULT_SOBOL_RTOL,
        },
    )
    return fixed, convergence


def _normalize_sobol_mode_block(
    value: object,
    *,
    mode: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"sobol_parameters.{mode} must be a dictionary.")
    block = {**defaults, **value}
    unsupported = sorted(set(block) - set(defaults))
    if unsupported:
        raise ValueError(f"Unsupported Sobol parameter(s): {unsupported}.")
    if not isinstance(block["active"], bool):
        raise ValueError(f"sobol_parameters.{mode}.active must be a boolean.")
    return block


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"sobol_parameters.{field} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"sobol_parameters.{field} must be a positive integer.")
    return parsed


def _positive_power_of_two(value: Any, *, field: str) -> int:
    parsed = _positive_int(value, field=field)
    if parsed & (parsed - 1):
        raise ValueError(f"sobol_parameters.{field} must be a power of two.")
    return parsed


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"sobol_parameters.{field} must be positive.")
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"sobol_parameters.{field} must be positive.")
    return parsed


def _optional_years(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError("sobol_parameters.sobol_years must be a non-empty sequence of years.")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(
            "sobol_parameters.sobol_years must be a non-empty sequence of years."
        ) from exc
    if not raw:
        raise ValueError("sobol_parameters.sobol_years must be a non-empty sequence of years.")
    years: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError("sobol_parameters.sobol_years must contain integer years.")
        year = int(item)
        if year <= 0:
            raise ValueError("sobol_parameters.sobol_years must contain positive integer years.")
        if year not in years:
            years.append(year)
    return tuple(years)


def _validate_sobol_year_membership(
    *,
    sobol_years: tuple[int, ...] | None,
    available_years: list[int] | tuple[int, ...] | range | None,
) -> None:
    """Require requested Sobol years to belong to the studied output years."""
    if sobol_years is None or available_years is None:
        return
    studied = sorted({int(year) for year in available_years})
    unsupported = sorted(set(sobol_years) - set(studied))
    if unsupported:
        raise ValueError(
            "sobol_parameters.sobol_years must be selected from the studied years. "
            f"Unsupported year(s): {unsupported}. Studied years: {studied}."
        )
