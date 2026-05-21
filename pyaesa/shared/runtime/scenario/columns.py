"""Canonical scenario and aSoCC time route column names."""

ASOCC_SSP_SCENARIO_COLUMN = "asocc_ssp_scenario"
ASOCC_TIME_ROUTE_PUBLIC_COLUMN = "asocc_time_route"
ASOCC_TIME_ROUTE_HISTORICAL = "historical"
ASOCC_TIME_ROUTE_HISTORICAL_REUSE = "historical_reuse"
ASOCC_TIME_ROUTE_REGRESSION = "regression_proj"
ASOCC_PROSPECTIVE_TIME_ROUTE_VALUES = frozenset(
    {ASOCC_TIME_ROUTE_HISTORICAL_REUSE, ASOCC_TIME_ROUTE_REGRESSION}
)
AR6_CC_SSP_SCENARIO_COLUMN = "ar6_cc_ssp_scenario"
EXT_LCA_SSP_SCENARIO_COLUMN = "lca_ssp_scenario"
LCA_SSP_START_YEAR_COLUMN = "lca_ssp_start_year"
