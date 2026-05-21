"""Functional unit selector axis contracts shared by staged inputs."""


def expected_fu_selector_columns(*, fu_code: str) -> tuple[str, ...]:
    """Return public selector columns required by one functional unit."""
    fu = str(fu_code).strip()
    if fu == "L1.a":
        return ("r_f",)
    if fu == "L1.b":
        return ("r_p",)
    if fu.startswith("L2.a."):
        return ("r_p", "s_p")
    if fu == "L2.b.a":
        return ("r_p", "s_p", "r_f")
    if fu == "L2.b.b":
        return ("r_p", "s_p", "r_c")
    if fu == "L2.c.a":
        return ("s_p", "r_f")
    if fu == "L2.c.b":
        return ("s_p", "r_c")
    raise ValueError(f"Unsupported functional unit code '{fu_code}'.")
