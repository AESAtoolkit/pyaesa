"""Figure explanation text for processed AR6 climate figures."""

from pathlib import Path

from pyaesa.shared.runtime.text import join_user_text_lines


def _remaining_budget_drop_csv_map(figures_dir: Path) -> dict[str, list[str]]:
    suffix = "-remaining-budget-panel-dropped_rows.csv"
    mapping: dict[str, list[str]] = {}
    for csv_path in sorted(figures_dir.glob(f"*{suffix}")):
        figure_stem = csv_path.stem.removesuffix("-remaining-budget-panel-dropped_rows")
        mapping.setdefault(figure_stem, []).append(csv_path.name)
    return mapping


def _figure_explanation_block(
    figure_name: str,
    study_period: list[int],
    global_drop_csv_file: Path,
    remaining_budget_drop_map: dict[str, list[str]],
) -> list[str]:
    lines = [f"Figure file: {figure_name}"]
    figure_stem = Path(figure_name).stem
    if figure_name.startswith("fig-processed-historical-emissions"):
        return lines + [
            (
                "What it shows: the historical baseline used for harmonization, "
                "constructed from PRIMAP-hist together with the Global Carbon Budget "
                "bunker CO2 time series, plus a red AR6 historical comparison overlay "
                "derived from the separate public download 'AR6_historical_emissions.csv'."
            ),
            (
                "How to read it: black and grey curves are the historical baseline "
                "used in processing; the red comparison line and uncertainty band are "
                "figure-only reference data and are not used to construct the "
                "harmonization baseline."
            ),
            (
                "Visual elements: solid black/grey lines show the aggregate "
                "historical baseline series, dashed coloured lines show component "
                "series including bunkers, and the red shaded band shows the "
                "lower-to-upper comparison range around the red AR6 reference line."
            ),
            (
                "Rows present: none from AR6 pathways; this figure is independent "
                "of scenario filtering."
            ),
            "Rows dropped: none at figure level.",
        ]
    if figure_name.startswith("fig-harmonization-pathways"):
        return lines + [
            (
                "What it shows: top row = retained AR6 pathways after preprocessing, "
                "bottom row = the same rows after harmonization to the historical "
                "baseline, split by output variable."
            ),
            (
                "Rows present: the same retained model-scenario variable set "
                "on the top and bottom panels."
            ),
            (
                "Visual elements: coloured pathway lines follow the AR6 category "
                "colours and the thick black line shows the historical series. "
                "In the lower row, the grey dashed vertical line marks the study "
                "start year used for harmonization."
            ),
            "Rows dropped before this figure: see the global dropped rows file below.",
            (
                "Pathway endpoints: if a retained row ends before 2100, its "
                "line stops at its last available year; no display extrapolation is added."
            ),
        ]
    if figure_name.startswith("fig-harmonization-stats-delta-tconv"):
        return lines + [
            (
                "What it shows: the absolute difference between each pathway's "
                "first model year with negative emissions and the harmonization "
                "convergence year t_conv."
            ),
            "Interpretation: larger values indicate greater separation between the two years.",
            (
                "Visual elements: the violin shows the distribution across "
                "harmonized rows, and the panel title 'n=...' reports the number "
                "of rows with a positive defined difference."
            ),
            "Rows present: harmonized rows only, via the harmonization log.",
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-harmonization-stats-"):
        return lines + [
            "What it shows: harmonization diagnostics by variable.",
            (
                "Top row: pathway cumulative emissions divided by historical "
                "cumulative emissions over the harmonization window."
            ),
            "Bottom row: yearly correction applied by harmonization.",
            (
                "Visual elements: violin bodies show the row distribution for "
                "each variable. In the top row, the red dashed line marks "
                "ratio = 1. In the bottom row, the black dashed line marks "
                "zero yearly correction. Panel titles include 'n=...' for the "
                "number of plotted rows."
            ),
            "Rows present: harmonized rows only, via the harmonization log.",
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-sequestration-contributions"):
        return lines + [
            (
                "What it shows: AR6 carbon sequestration component pathways "
                "used to build the total sequestration and gross alternative "
                "subtotal companion rows."
            ),
            (
                "Rows present: retained pre harmonization ORIGINAL_AR6 "
                "sequestration component, subtotal, and total rows after AR6 "
                "processing filters."
            ),
            (
                "Visual elements: each panel shows one sequestration variable "
                "within one AR6 category. Thin coloured lines are retained "
                "pathways and the panel label 'n=...' reports the number of "
                "plotted rows."
            ),
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-sequestration-budgets-for"):
        lines.extend(
            [
                (
                    "What it shows: top row = sequestration companion pathways, "
                    "middle row = distributions of cumulative sequestration over "
                    "the study period, bottom row = distributions of cumulative "
                    "sequestration after the study period up to the remaining "
                    "budget end year."
                ),
                f"Study period budget window: {int(study_period[0])}-{int(study_period[1])}.",
                (
                    "Rows present in the pathways and study-budget panels: "
                    "HARMONIZED_AR6 sequestration companion rows selected for "
                    "the model-scenario scope retained by each displayed "
                    "emissions variable. Gross alternative variables use "
                    "Carbon Sequestration|Subtotal_seq; other emissions "
                    "variables use Carbon Sequestration|Total."
                ),
                (
                    "Rows present in the remaining-budget panel: only "
                    "sequestration companion rows with a finite value at the "
                    "remaining-budget end year."
                ),
                (
                    "Visual elements: in the pathway row, coloured thin lines "
                    "show retained sequestration companions. In the budget "
                    "rows, violin bodies show distributions and black-edged "
                    "dots mark medians. Wider violins at integer category "
                    "positions aggregate all SSPs within a category; narrower "
                    "offset violins show SSP-specific subsets."
                ),
                "Rows dropped before this figure: see the global dropped rows file below.",
            ]
        )
        extra_csvs = remaining_budget_drop_map.get(figure_stem, [])
        lines.append(
            "Rows dropped only from the remaining-budget panel: "
            + (", ".join(extra_csvs) + "." if extra_csvs else "none.")
        )
        return lines
    if figure_name.startswith("fig-budgets-"):
        lines.extend(
            [
                (
                    "What it shows: top row = harmonized pathways, middle row = "
                    "distributions of cumulative emissions over the study period, "
                    "bottom row = distributions of cumulative emissions after the "
                    "study period up to the remaining-budget end year."
                ),
                f"Study period budget window: {int(study_period[0])}-{int(study_period[1])}.",
                (
                    "Rows present in the pathways and study-budget panels: "
                    "processed emissions rows kept by processing for the selected "
                    "CO2 or Kyoto Gases figure group, including net, gross, and "
                    "gross alternative variables when present in the processed scope."
                ),
                (
                    "Rows present in the remaining-budget panel: only "
                    "harmonized rows with a finite value at the remaining-budget "
                    "end year."
                ),
                (
                    "Visual elements: in the pathway row, coloured thin lines "
                    "show retained pathways, the thick black line shows the "
                    "historical series, and the grey shaded band marks the study "
                    "window integrated in the budget panels. In the budget rows, "
                    "violin bodies show distributions and black-edged dots mark "
                    "medians."
                ),
                (
                    "Visual elements in the budget rows: wider violins at the "
                    "integer category positions aggregate all SSPs within a "
                    "category, while narrower offset violins show SSP-specific "
                    "subsets. Category colours follow the AR6 category colour "
                    "scheme; remaining-budget violins are rendered in black."
                ),
                "Rows dropped before this figure: see the global dropped rows file below.",
            ]
        )
        extra_csvs = remaining_budget_drop_map.get(figure_stem, [])
        lines.append(
            "Rows dropped only from the remaining-budget panel: "
            + (", ".join(extra_csvs) + "." if extra_csvs else "none.")
        )
        return lines
    if figure_name.startswith("fig-median-warming"):
        return lines + [
            (
                "What it shows: distributions of the metadata field 'Median "
                "warming in 2100 (MAGICCv7.5.3)' by category and by "
                "category/SSP grouping."
            ),
            (
                "Rows present: original data rows retained after AR6 processing "
                "filters, matched back to source metadata by model-scenario."
            ),
            (
                "Visual elements: violin bodies show the distribution of the "
                "metadata value and black-edged dots mark medians. Wider violins "
                "at integer category positions aggregate all SSPs in a category, "
                "while narrower offset violins show SSP-specific subsets."
            ),
            (
                "Visual elements: rotated labels of the form 'SSP... (n=...)' "
                "identify the SSP-specific subset and its sample size."
            ),
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-LHSSRS-ratioproba"):
        return lines + [
            (
                "What it shows: for each processed emissions variable in the "
                "selected CO2 or Kyoto Gases figure group, the ratio of "
                "row-level selection probabilities under the package's "
                "LHS-labelled scheme relative to SRS."
            ),
            (
                "Interpretation: ratio > 1 means the row receives greater "
                "selection weight under the LHS-labelled scheme than under "
                "SRS; ratio < 1 means the opposite."
            ),
            (
                "Visual elements: bars are retained rows sorted by descending "
                "ratio. The red solid line marks parity at ratio = 1 for the "
                "probability ratio itself. The red dashed lines mark one order "
                "of magnitude above and below parity (10 and 0.1) for that same "
                "probability ratio, not for medians or means."
            ),
            (
                "Visual elements: the percentage label reports the share of "
                "rows with LHS/SRS ratio > 1 in the plotted variable panel."
            ),
            "Rows present: processed emissions rows kept in HARMONIZED_AR6.",
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-LHSSRS-ratiomedian"):
        return lines + [
            (
                "What it shows: for each processed emissions variable in the "
                "selected CO2 or Kyoto Gases figure group, the ratio between "
                "the LHS-labelled and SRS medians of the sampled study-period "
                "cumulative budget distributions."
            ),
            (
                "Visual elements: bars are category or category/SSP summary "
                "groups. The red solid line marks parity at ratio = 1 for the "
                "median ratio. The orange dotted line marks the minimum plotted "
                "median ratio and the orange dashed line marks the maximum "
                "plotted median ratio. No mean reference line is rendered."
            ),
            "Rows present: processed emissions rows kept in HARMONIZED_AR6.",
            "Rows dropped before this figure: see the global dropped rows file below.",
            "Rows dropped at figure level: none.",
        ]
    if figure_name.startswith("fig-LHSSRS-budgets-GHG") or figure_name.startswith(
        "fig-LHSSRS-budgets-CO2"
    ):
        lines.extend(
            [
                "What it shows: combined SRS versus LHS-labelled sampling "
                "comparison for the selected CO2 or Kyoto Gases processed "
                "emissions group and selected emissions mode.",
                "Top row left panels: LHS/SRS row-level probability ratios.",
                "Top row right panels: LHS/SRS ratios of sampled study-period medians.",
                (
                    "Lower rows: sampled study-period and remaining-budget "
                    "distributions, shown separately for SRS and LHS."
                ),
                (
                    "Rows present in the study-budget panels: retained "
                    "harmonized rows sampled with the corresponding SRS or "
                    "LHS probabilities."
                ),
                (
                    "Visual elements in the top-left panels: bars are row-level "
                    "LHS/SRS probability ratios. The red solid line marks "
                    "ratio = 1 for the probability ratio, the red dashed lines "
                    "mark 10 and 0.1 for that same probability ratio, and the "
                    "percentage label reports the share of rows with ratio > 1. "
                    "These guides are not related to medians or means."
                ),
                (
                    "Visual elements in the top-right panels: bars are "
                    "LHS/SRS ratios of sampled study-period medians. The red "
                    "solid line marks ratio = 1 for the median ratio, the "
                    "orange dotted line marks the minimum plotted median ratio, "
                    "and the orange dashed line marks the maximum plotted "
                    "median ratio. No mean reference line is rendered."
                ),
                (
                    "Visual elements in the lower panels: violin bodies show "
                    "sampled distributions and black-edged dots mark medians. "
                    "Wider violins at integer category positions aggregate all "
                    "SSPs within a category; narrower offset violins show "
                    "SSP-specific subsets."
                ),
                "Rows dropped before this figure: see the global dropped rows file below.",
            ]
        )
        extra_csvs = remaining_budget_drop_map.get(figure_stem, [])
        lines.append(
            "Rows dropped only from the remaining-budget panels: "
            + (", ".join(extra_csvs) + "." if extra_csvs else "none.")
        )
        return lines
    return lines + [
        "What it shows: no explanation template is defined for this file name.",
        f"Rows dropped before this figure: see {global_drop_csv_file.name}.",
    ]


def figures_explanation_text(
    figure_files: list[str],
    study_period: list[int],
    figures_dir: Path,
    global_drop_csv_file: Path,
) -> str:
    """Return the complete figures explanation TXT content."""
    remaining_budget_drop_map = _remaining_budget_drop_csv_map(figures_dir)
    lines = [
        "Figures Guide",
        "",
        (
            "Study period used for cumulative study-budget calculations: "
            f"{int(study_period[0])}-{int(study_period[1])}"
        ),
        f"Global dropped rows file for preprocessing/filtering: {global_drop_csv_file.name}",
        (
            "Rule of thumb for 'rows present': unless noted otherwise, figures "
            "use final processed HARMONIZED_AR6 rows retained after AR6 "
            "processing filters."
        ),
        (
            "Additional figure-level drop files exist only for remaining-budget "
            "panels, because those panels require a finite value at the "
            "remaining-budget end year."
        ),
        "Sampling abbreviations used below:",
        (
            "- Simple Random Sampling (SRS): independent random runs with "
            "equal row-level probability within each variable-category-SSP bucket."
        ),
        (
            "- Latin Hypercube Sampling (LHS, as labelled in these outputs): "
            "within each variable-category-SSP bucket, row-level probabilities "
            "are reweighted so each model contributes equal total sampling "
            "mass before multinomial runs are generated. Relative to SRS, "
            "this reduces over-representation of models with many retained rows."
        ),
        "",
    ]
    for figure_file in sorted(figure_files):
        figure_name = Path(figure_file).name
        lines.extend(
            _figure_explanation_block(
                figure_name, study_period, global_drop_csv_file, remaining_budget_drop_map
            )
        )
        lines.append("")
    return join_user_text_lines(lines, trailing_newline=True)


def write_figures_guide(
    *,
    figures_dir: Path,
    figure_files: list[str],
    study_period: list[int],
    global_drop_csv_file: Path,
) -> Path:
    """Write the figure explanation TXT and return its path."""
    guide_file = figures_dir / "figures_explanation.txt"
    guide_file.write_text(
        figures_explanation_text(figure_files, study_period, figures_dir, global_drop_csv_file),
        encoding="utf-8",
    )
    return guide_file


def ensure_figures_guide(
    *,
    figures_dir: Path,
    figure_files: list[str],
    study_period: list[int],
    global_drop_csv_file: Path,
    rewrite: bool,
) -> tuple[Path | None, bool]:
    """Write or reuse the figure explanation TXT and report whether it changed."""
    guide_file = figures_dir / "figures_explanation.txt"
    if (not figure_files) and (not guide_file.exists()):
        return None, False
    if rewrite or (not guide_file.exists()):
        return (
            write_figures_guide(
                figures_dir=figures_dir,
                figure_files=figure_files,
                study_period=study_period,
                global_drop_csv_file=global_drop_csv_file,
            ),
            True,
        )
    return guide_file, False
