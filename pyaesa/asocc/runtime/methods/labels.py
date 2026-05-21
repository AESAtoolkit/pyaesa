"""Raw aSoCC method-label parsing shared across the aSoCC family."""


def l1_l2_method_label(*, l1_method: str, l2_method: str) -> str:
    """Return the canonical public two step method label."""
    return f"{str(l1_method).strip()}_{str(l2_method).strip()}"


def parse_raw_asocc_method_label(
    raw_asocc_method_label: str,
) -> tuple[str, str | None, str]:
    """Parse one raw aSoCC method label into tree classification tokens.

    Args:
        raw_asocc_method_label: Raw aSoCC method label such as ``"EG(Pop)"``,
            ``"PR-HR(Ecap,cum)"``, or ``"AR(E^{CBA_FD})"``.

    Returns:
        Tuple ``(sharing_principle, subprinciple, enacting_metric)``.

    Raises:
        ValueError: If ``raw_asocc_method_label`` does not use the canonical scientific
            label structure.
    """
    text = str(raw_asocc_method_label).strip()
    if "(" not in text or not text.endswith(")"):
        raise ValueError(
            "Unsupported aSoCC method label. Expected forms such as 'UT(FD)', "
            "'PR-HR(Ecap,cum)', or 'AR(E^{CBA_FD})'. "
            f"Received '{raw_asocc_method_label}'."
        )
    prefix, suffix = text.split("(", 1)
    prefix = prefix.strip()
    inner = suffix[:-1].strip().replace("^{", "_").replace("}", "")
    if not prefix or not inner:
        raise ValueError(
            "Unsupported aSoCC method label: both the sharing principle before '(' "
            "and the enacting metric inside parentheses are required. "
            f"Received '{raw_asocc_method_label}'."
        )
    if "-" in prefix:
        sharing_principle, subprinciple = prefix.split("-", 1)
        sharing_principle = sharing_principle.strip()
        subprinciple = subprinciple.strip() or None
    else:
        sharing_principle, subprinciple = prefix, None
    if not sharing_principle:
        raise ValueError(
            "Unsupported aSoCC method label: the sharing principle portion is empty. "
            f"Received '{raw_asocc_method_label}'."
        )
    return sharing_principle, subprinciple, inner
