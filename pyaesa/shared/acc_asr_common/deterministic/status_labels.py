"""Live status labels shared by deterministic aCC and ASR branches."""


def cc_branch_status_label(*, cc_source: str, cc_type: str) -> str:
    """Return the compact carrying capacity label for live branch messages."""
    if str(cc_type).strip() == "static":
        return str(cc_source).strip()
    return "GWP100_dynamic"
