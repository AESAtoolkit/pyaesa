"""Workbook and artefact naming ownership for processed AR6 outputs."""


def processed_workbook_name(*, harmonization: bool) -> str:
    """Return the main workbook file name for the selected processing mode."""
    if harmonization:
        return "harmonized_ar6_public.xlsx"
    return "filtered_original_ar6_public.xlsx"


def harmonization_log_workbook_name() -> str:
    """Return the harmonization log workbook file name."""
    return "harmonized_ar6_public_log.xlsx"


def final_pathways_sheet_name(*, harmonization: bool) -> str:
    """Return the final pathways worksheet name for the selected mode."""
    if harmonization:
        return "HARMONIZED_AR6"
    return "RETAINED_AR6_FILTERED"


def budget_stats_sheet_name(*, harmonization: bool) -> str:
    """Return the budget statistics worksheet name for the selected mode."""
    if harmonization:
        return "BUDGET_STATS_HARMONIZED"
    return "BUDGET_STATS_RETAINED"
