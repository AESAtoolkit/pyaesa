"""Normalization for method-selection runtime inputs."""

_L1_REG_MODES = {"pre", "post"}


def normalize_l1_reg_mode_required(value: str | None) -> str:
    """Normalize one required L1 aggregation mode input."""
    if not isinstance(value, str):
        raise ValueError("l1_reg_aggreg must be a string with value 'pre' or 'post'.")
    mode = value.strip().lower()
    if mode not in _L1_REG_MODES:
        raise ValueError("l1_reg_aggreg must be one of: 'pre', 'post'.")
    return mode


def normalize_l1_reg_mode(value: str | None) -> str:
    """Normalize L1 aggregation mode input to one explicit branch."""
    if value is None:
        return "post"
    return normalize_l1_reg_mode_required(value)


def normalize_output_mode(value: bool) -> bool:
    """Normalize the public aggregation selector to one execution branch."""
    if isinstance(value, bool):
        return value
    raise ValueError("aggreg_indices must be a boolean.")


def resolve_level(*, fu_norm: str) -> str:
    """Resolve functional unit level from normalized FU code."""
    if fu_norm.startswith("L1."):
        return "l1"
    return "l2"


def normalize_plan(method_plan: str) -> str:
    """Validate and normalize method plan name."""
    plan = str(method_plan).strip().lower()
    valid_plans = {"default", "one_step", "two_steps", "pairs", "one_step_pairs"}
    if plan not in valid_plans:
        raise ValueError(
            "method_plan must be one of: "
            "'default', 'one_step', 'two_steps', 'pairs', 'one_step_pairs'."
        )
    return plan
