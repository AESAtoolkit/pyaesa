"""Configuration constants for AR6 dynamic climate change collection."""

RAW_KYOTO_WITH_AFOLU = "Emissions|Kyoto Gases"
RAW_CO2_WITH_AFOLU = "Emissions|CO2"
RAW_CO2_AFOLU = "Emissions|CO2|AFOLU"
RAW_CH4_AFOLU = "Emissions|CH4|AFOLU"
RAW_N2O_AFOLU = "Emissions|N2O|AFOLU"
RAW_CO2_AFOLU_LAND = "Emissions|CO2|AFOLU|Land"
RAW_CO2_ENERGY = "Emissions|CO2|Energy"
RAW_CO2_INDUSTRIAL_PROCESSES = "Emissions|CO2|Industrial Processes"
RAW_CO2_OTHER = "Emissions|CO2|Other"
RAW_CO2_WASTE = "Emissions|CO2|Waste"
RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES = "Emissions|CO2|Energy and Industrial Processes"

RAW_SEQUESTRATION_LAND_USE = "Carbon Sequestration|Land Use"
RAW_SEQUESTRATION_COMPONENTS = (
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|Direct Air Capture",
    "Carbon Sequestration|Enhanced Weathering",
    "Carbon Sequestration|Feedstocks",
    RAW_SEQUESTRATION_LAND_USE,
    "Carbon Sequestration|Other",
)
RAW_SEQUESTRATION_SUBTOTAL_COMPONENTS = tuple(
    component
    for component in RAW_SEQUESTRATION_COMPONENTS
    if component != "Carbon Sequestration|CCS"
)
RAW_VARIABLES_RELEVANT = (
    RAW_KYOTO_WITH_AFOLU,
    RAW_CO2_WITH_AFOLU,
    RAW_CO2_AFOLU,
    RAW_CH4_AFOLU,
    RAW_N2O_AFOLU,
    RAW_CO2_AFOLU_LAND,
    RAW_CO2_ENERGY,
    RAW_CO2_INDUSTRIAL_PROCESSES,
    RAW_CO2_OTHER,
    RAW_CO2_WASTE,
    RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES,
    *RAW_SEQUESTRATION_COMPONENTS,
)

NET_KYOTO_WITH_AFOLU = "Emissions(net)|Kyoto Gases"
NET_KYOTO_WO_AFOLU = "Emissions(net)|Kyoto Gases|WO AFOLU"
NET_CO2_WITH_AFOLU = "Emissions(net)|CO2"
NET_CO2_WO_AFOLU = "Emissions(net)|CO2|WO AFOLU"

GROSS_KYOTO_WITH_AFOLU = "Emissions(gross)|Kyoto Gases"
GROSS_KYOTO_WO_AFOLU = "Emissions(gross)|Kyoto Gases|WO AFOLU"
GROSS_CO2_WITH_AFOLU = "Emissions(gross)|CO2"
GROSS_CO2_WO_AFOLU = "Emissions(gross)|CO2|WO AFOLU"

GROSS_ALT_KYOTO_WITH_AFOLU = "Emissions(gross_alt)|Kyoto Gases"
GROSS_ALT_KYOTO_WO_AFOLU = "Emissions(gross_alt)|Kyoto Gases|WO AFOLU"
GROSS_ALT_CO2_WITH_AFOLU = "Emissions(gross_alt)|CO2"
GROSS_ALT_CO2_WO_AFOLU = "Emissions(gross_alt)|CO2|WO AFOLU"

SEQUESTRATION_TOTAL = "Carbon Sequestration|Total"
SEQUESTRATION_SUBTOTAL = "Carbon Sequestration|Subtotal_seq"

PROCESSED_OUTPUT_VARIABLES = (
    NET_KYOTO_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_CO2_WITH_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    GROSS_KYOTO_WITH_AFOLU,
    GROSS_CO2_WO_AFOLU,
    GROSS_CO2_WITH_AFOLU,
    GROSS_ALT_KYOTO_WO_AFOLU,
    GROSS_ALT_KYOTO_WITH_AFOLU,
    GROSS_ALT_CO2_WO_AFOLU,
    GROSS_ALT_CO2_WITH_AFOLU,
)

DEFAULT_DATABASE = "ar6-public"
DEFAULT_REGION = "World"

DEFAULT_VARIABLES_RELEVANT = list(RAW_VARIABLES_RELEVANT)

DEFAULT_META_COLUMNS = [
    "Ssp_family",
    "Vetting_historical",
    "Vetting_future",
    "Time horizon",
    "Category",
    "Category_name",
    "Category_subset",
    "Cumulative net CO2 (2020-2100, Gt CO2) (Harm-Infilled)",
    "Cumulative net CO2 (2020 to netzero, Gt CO2) (Harm-Infilled)",
    "Cumulative net-negative CO2 (post net-zero, Gt CO2) (Harm-Infilled)",
    "Peak Emissions|CO2",
    "Peak Emissions|GHGs",
    "Exceedance Probability 1.5C (FaIRv1.6.2)",
    "Exceedance Probability 1.5C (MAGICCv7.5.3)",
    "Exceedance Probability 2.0C (FaIRv1.6.2)",
    "Exceedance Probability 2.0C (MAGICCv7.5.3)",
    "CO2 emissions 2030 Gt CO2/yr",
    "CO2 emissions 2050 Gt CO2/yr",
    "CO2 emissions 2100 Gt CO2/yr",
    "GHG emissions 2030 Gt CO2-equiv/yr (Harmonized-Infilled)",
    "GHG emissions 2050 Gt CO2-equiv/yr (Harmonized-Infilled)",
    "GHG emissions 2100 Gt CO2-equiv/yr (Harmonized-Infilled)",
    "Median warming in 2100 (FaIRv1.6.2)",
    "Median warming in 2100 (MAGICCv7.5.3)",
    "Policy_category",
    "Policy_category_name",
    "Literature Reference (if applicable)",
]

RECOMMENDED_AR6_CATEGORIES = ["C1", "C2", "C3", "C4"]
VALID_AR6_CATEGORIES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
DEFAULT_CATEGORIES = list(RECOMMENDED_AR6_CATEGORIES)
DEFAULT_SSPS = [1, 2, 3, 4, 5]
DEFAULT_VARIABLES_OUTPUT = list(PROCESSED_OUTPUT_VARIABLES)


def normalize_ar6_categories(category: str | list[str] | None) -> list[str]:
    """Normalize and validate AR6 category selectors."""
    if category is None:
        return list(RECOMMENDED_AR6_CATEGORIES)
    if isinstance(category, str):
        values = [category]
    elif isinstance(category, list):
        values = category
    else:
        raise ValueError("category must be a non empty AR6 category string or list.")
    if not values or any(not isinstance(item, str) for item in values):
        raise ValueError("category must be a non empty AR6 category string or list.")
    categories = [item.strip().upper() for item in values]
    if not categories or any(not item for item in categories):
        raise ValueError("category must be a non empty AR6 category string or list.")
    invalid = sorted(set(categories) - set(VALID_AR6_CATEGORIES))
    if invalid:
        valid = ", ".join(VALID_AR6_CATEGORIES)
        received = ", ".join(invalid)
        raise ValueError(
            f"category must contain only AR6 categories {valid}. Received: {received}."
        )
    return sorted(dict.fromkeys(categories))
