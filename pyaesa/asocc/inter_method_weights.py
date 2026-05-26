"""Public allocated shares of carrying capacities (aSoCC) inter-method utilities."""

from typing import Any

from pyaesa.asocc.inter_method_tools.tree import (
    DEFAULT_INTER_METHOD_TREE_VERSION,
    build_inter_method_tree_frame,
    inter_method_preview_figure_base,
    inter_method_tree_path,
    load_valid_inter_method_tree_frame,
    plan_inter_method_tree_request,
)
from pyaesa.asocc.inter_method_tools.tree_artifacts import (
    InterMethodTreeArtifacts,
    write_inter_method_tree_artifacts,
    write_inter_method_tree_guide,
)
from pyaesa.asocc.inter_method_tools.tree_figure import render_inter_method_tree
from pyaesa.shared.figures.request_validation import normalize_figure_format


def write_asocc_weight_template(
    *,
    base_asocc_args: dict[str, Any] = {
        "project_name": None,
        "source": None,
        "agg_reg": False,
        "agg_sec": False,
        "agg_version": "",
        "years": None,
        "fu_code": None,
        "s_p": None,
        "r_p": None,
        "r_c": None,
        "r_f": None,
        "group_indices": False,
        "method_plan": "default",
        "l1_methods": None,
        "one_step_methods": None,
        "two_step_methods": None,
        "l1_l2_pairs": None,
        "l1_reg_aggreg": "post",
        "lcia_method": None,
        "reference_years": None,
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "projection_mode": "regression",
        "reg_window": None,
        "l2_reuse_years": None,
    },
    external_method: dict[str, Any] | None = None,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
) -> InterMethodTreeArtifacts:
    """Write equal weight inter-method templates for one method scope.

    The templates define custom weights for allocated shares of carrying
    capacities (aSoCC) inter-method uncertainty. Omit arguments to use their
    default.

    Args:
        base_asocc_args: Deterministic aSoCC selector envelope used to resolve
            the final method leaves represented by the tree. Write nested
            arguments as ``base_asocc_args={"project_name": "...", "source":
            "...", "fu_code": "..."}``.

            Nested keys:

            - ``project_name``: Required project name used to build
              ``<repo>/<project_name>``.
            - ``source``: MRIO source key (``"exiobase_396_ixi"``,
              ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
              ``"exiobase_3102_pxp"``, or ``"oecd_v2025"``), or ``"iso3"``
              for ISO3 only mode (L1 EG/PR(GDPcap) only).
            - ``agg_reg``: If ``True``, reclassify MRIO regions with the
              ``agg_reg_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native regions
              into one target label, or disaggregate one native region across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source regions.
            - ``agg_sec``: If ``True``, reclassify MRIO sectors with the
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native sectors
              into one target label, or disaggregate one native sector across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source sectors.
            - ``agg_version``: Name token used to resolve the matching
              ``agg_reg_<agg_version>.csv`` and/or
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation
              mapping files in ``data_raw/mrio/<source>/aggregation``.
              Required when ``agg_reg`` or ``agg_sec`` is True. Defaults to
              an empty string for native source classification. Use the same
              token in downstream calls that should reuse the processed
              classification. When a mapping file has a ``weight``
              column, weights must sum to ``1`` for each original label.
            - ``years``: Studied years. Accepts a single year, list, or
              range. If omitted, all available MRIO years for the selected
              source and ``agg_version`` are used.
            - ``fu_code``: Required functional unit code (for example
              ``"L1.a"``, ``"L2.c.b"``). See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for all available functional unit codes and the system
              boundaries each represents.
            - ``s_p``: Producing sector filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing sectors. To
              identify valid sector names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/.../agg_sec_template.csv``
              file. For EXIOBASE sector definitions, see
              ``data_raw/mrio/exiobase_3/sector_classification.xlsx``;
              EXIOBASE ixi and pxp use different sector lists.
            - ``r_p``: Producing region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_c``: Consuming region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid consuming regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_f``: Final demand region filter(s), single string or list.
              If this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid final demand regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``group_indices``: Whether multiple selected region or sector
              filter values are kept as separate result rows or summed into one
              result row after the function calculation has been performed.
              - ``False`` (default): keep selected values as independent rows.
              - ``True``: sum selected values into one result row.
              The function refuses to run when ``group_indices=True`` is used
              with ``L2.a.b``, ``L2.b.b``, or ``L2.c.b`` because summing output
              rows for CBA total demand boundaries can double count. For these
              functional units, change the upstream MRIO aggregation and disaggregation
              scope with ``agg_reg``, ``agg_sec``, and ``agg_version`` before
              running the study.
            - ``method_plan``: ``method_plan`` defaults to ``"default"`` and
              accepts ``"default"``, ``"one_step"``, ``"two_steps"``,
              ``"pairs"``, or ``"one_step_pairs"``. When omitted, all pyaesa
              allocation methods available for the selected ``fu_code`` are
              applied. See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for the allocation methods available per functional unit,
              including definitions and mathematical expressions.
            - ``l1_methods``: Optional L1 subset. Omit it to keep all L1
              methods allowed by ``method_plan``. In ``"default"``, this
              filters only L1 weights used by two step methods. In
              ``"two_steps"``, it filters the two step cartesian L1 side.
            - ``one_step_methods``: Optional one step L2 subset. Omit it to
              keep all one step methods allowed by ``method_plan``.
            - ``two_step_methods``: Optional two step L2 subset. Omit it to
              keep all two step L2 methods allowed by ``method_plan``.
            - ``l1_l2_pairs``: Explicit pair list formatted as
              ``"L1METHOD::L2METHOD"``. Omit it unless ``method_plan`` is
              ``"pairs"`` or ``"one_step_pairs"``.
            - ``l1_reg_aggreg``: L1 aggregation mode for methods where timing
              matters (``PR(GDPcap)``, ``PR-HR(Ecap)`` and ``AR(Ecap)``).
              ``"pre"`` aggregates regions before L1 computation. ``"post"``
              (default) computes on original regions, then aggregates.
            - ``lcia_method``: LCIA method(s) to characterize for EXIO
              processing (for example ``"pb_lcia"`` or
              ``["pb_lcia", "gwp100_lcia"]``). ``None`` means LCIA
              characterization is skipped. Defaults to ``None``. LCIA
              characterization is available only for EXIOBASE sources. To add
              a custom LCIA method, follow
              ``README_add_custom_lcia_characterization_matrices.txt`` in
              ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
              and pass the method file stem here.
            - ``reference_years``: Acquired rights (AR) methods reference
              year selector. Accepts a single year, list, or range. If
              omitted, AR routes use all historical years in the studied range
              up to the source registry historical cutoff. For EXIOBASE
              3.10.2 and OECD ICIO v2025, the cutoff is 2022; other supported
              MRIO sources use their own registry cutoff.
            - ``ssp_scenario``: SSP scenario name or list. Defaults to
              ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]`` and is applied
              when scenario dependent inputs are required.
            - ``projection_mode``: Projection policy for post historical
              years of L2 utilitarian (UT) methods (MRIO economic enacting
              metrics). Defaults to ``"regression"``. ``"regression"``
              projects UT inputs for future years. ``"historical_reuse"``
              reuses historical UT structures.
            - ``reg_window``: Historical regression fit window for regression
              mode. Provide it as ``range(start_year, end_year + 1)`` or as
              an explicit list of consecutive years in fit window order. When
              omitted, the source registry supplies the default fit window
              from the modeled year minimum through the source historical
              cutoff. For EXIOBASE 3.10.2 and OECD ICIO v2025, this resolves
              to 1995 to 2022; other supported MRIO sources use their own
              registry window.
            - ``l2_reuse_years``: Historical L2 reuse year selector used by
              all UT historical reuse routes. In
              ``projection_mode="historical_reuse"`` it applies to all UT
              methods; in ``projection_mode="regression"`` it applies to
              adjusted UT routes (``UT(FDa)``, ``UT(GVAa)``), which always
              use historical reuse as regression is not supported (would
              require regression on full MRIO). If omitted, defaults to
              ``reg_window`` when required.
        external_method: Optional external aSoCC method selector. Use
            ``{"l1_methods": [...]}`` for L1 functional units. For L2
            functional units use ``{"one_step_methods": [...]}`` and/or
            ``{"l1_l2_pairs": ["<l1_method>::<l2_method>", ...]}``.
            Omit this argument or pass ``None`` when using only native pyaesa
            deterministic aSoCC methods. Use ``prepare_external_inputs(...)``
            to import the external aSoCC README guidance and runnable CSV examples, then
            follow the imported README guidance for external method names, selector
            syntax, and deterministic or Monte Carlo external aSoCC CSV
            preparation.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.

    Returns:
        Paths and probabilities for the written editable tree CSV, guide text,
        and preview figure.

    Raises:
        ValueError: If the requested deterministic aSoCC method scope or
            figure settings are invalid.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Write the editable tree for ``L2.c.b`` producing sector ``Paper`` and
        consuming region ``FR``::

            from pyaesa import write_asocc_weight_template

            write_asocc_weight_template(
                base_asocc_args={
                    "project_name": "demo",
                    "source": "exiobase_3102_ixi",
                    "fu_code": "L2.c.b",
                    "s_p": ["Paper"],
                    "r_c": ["FR"],
                },
            )
    """
    request = plan_inter_method_tree_request(
        base_asocc_args=base_asocc_args,
        external_method=external_method,
    )
    return write_inter_method_tree_artifacts(
        tree_csv_path=inter_method_tree_path(
            proj_base=request.proj_base,
            version_name=DEFAULT_INTER_METHOD_TREE_VERSION,
        ),
        figure_base_path=inter_method_preview_figure_base(
            proj_base=request.proj_base,
            version_name=DEFAULT_INTER_METHOD_TREE_VERSION,
        ),
        frame=build_inter_method_tree_frame(candidates=request.candidates),
        candidates=request.candidates,
        figure_format=figure_format,
    )


def preview_asocc_weight_tree(
    *,
    base_asocc_args: dict[str, Any] = {
        "project_name": None,
        "source": None,
        "agg_reg": False,
        "agg_sec": False,
        "agg_version": "",
        "years": None,
        "fu_code": None,
        "s_p": None,
        "r_p": None,
        "r_c": None,
        "r_f": None,
        "group_indices": False,
        "method_plan": "default",
        "l1_methods": None,
        "one_step_methods": None,
        "two_step_methods": None,
        "l1_l2_pairs": None,
        "l1_reg_aggreg": "post",
        "lcia_method": None,
        "reference_years": None,
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "projection_mode": "regression",
        "reg_window": None,
        "l2_reuse_years": None,
    },
    version_name: str,
    external_method: dict[str, Any] | None = None,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
) -> InterMethodTreeArtifacts:
    """Validate an edited custom inter-method tree and render its preview.

    The preview is built for one allocated shares of carrying capacities
    (aSoCC) method scope. Omit arguments to use their default.

    Args:
        base_asocc_args: Deterministic aSoCC selector envelope used to rebuild
            the expected final method tree. Write nested arguments as
            ``base_asocc_args={"project_name": "...", "source": "...",
            "fu_code": "..."}``.

            Nested keys:

            - ``project_name``: Required project name used to build
              ``<repo>/<project_name>``.
            - ``source``: MRIO source key (``"exiobase_396_ixi"``,
              ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
              ``"exiobase_3102_pxp"``, or ``"oecd_v2025"``), or ``"iso3"``
              for ISO3 only mode (L1 EG/PR(GDPcap) only).
            - ``agg_reg``: If ``True``, reclassify MRIO regions with the
              ``agg_reg_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native regions
              into one target label, or disaggregate one native region across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source regions.
            - ``agg_sec``: If ``True``, reclassify MRIO sectors with the
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native sectors
              into one target label, or disaggregate one native sector across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source sectors.
            - ``agg_version``: Name token used to resolve the matching
              ``agg_reg_<agg_version>.csv`` and/or
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation
              mapping files in ``data_raw/mrio/<source>/aggregation``.
              Required when ``agg_reg`` or ``agg_sec`` is True. Defaults to
              an empty string for native source classification. Use the same
              token in downstream calls that should reuse the processed
              classification. When a mapping file has a ``weight``
              column, weights must sum to ``1`` for each original label.
            - ``years``: Studied years. Accepts a single year, list, or
              range. If omitted, all available MRIO years for the selected
              source and ``agg_version`` are used.
            - ``fu_code``: Required functional unit code (for example
              ``"L1.a"``, ``"L2.c.b"``). See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for all available functional unit codes and the system
              boundaries each represents.
            - ``s_p``: Producing sector filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing sectors. To
              identify valid sector names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/.../agg_sec_template.csv``
              file. For EXIOBASE sector definitions, see
              ``data_raw/mrio/exiobase_3/sector_classification.xlsx``;
              EXIOBASE ixi and pxp use different sector lists.
            - ``r_p``: Producing region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_c``: Consuming region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid consuming regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_f``: Final demand region filter(s), single string or list.
              If this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid final demand regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``group_indices``: Whether multiple selected region or sector
              filter values are kept as separate result rows or summed into one
              result row after the function calculation has been performed.
              - ``False`` (default): keep selected values as independent rows.
              - ``True``: sum selected values into one result row.
              The function refuses to run when ``group_indices=True`` is used
              with ``L2.a.b``, ``L2.b.b``, or ``L2.c.b`` because summing output
              rows for CBA total demand boundaries can double count. For these
              functional units, change the upstream MRIO aggregation and disaggregation
              scope with ``agg_reg``, ``agg_sec``, and ``agg_version`` before
              running the study.
            - ``method_plan``: ``method_plan`` defaults to ``"default"`` and
              accepts ``"default"``, ``"one_step"``, ``"two_steps"``,
              ``"pairs"``, or ``"one_step_pairs"``. When omitted, all pyaesa
              allocation methods available for the selected ``fu_code`` are
              applied. See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for the allocation methods available per functional unit,
              including definitions and mathematical expressions.
            - ``l1_methods``: Optional L1 subset. Omit it to keep all L1
              methods allowed by ``method_plan``. In ``"default"``, this
              filters only L1 weights used by two step methods. In
              ``"two_steps"``, it filters the two step cartesian L1 side.
            - ``one_step_methods``: Optional one step L2 subset. Omit it to
              keep all one step methods allowed by ``method_plan``.
            - ``two_step_methods``: Optional two step L2 subset. Omit it to
              keep all two step L2 methods allowed by ``method_plan``.
            - ``l1_l2_pairs``: Explicit pair list formatted as
              ``"L1METHOD::L2METHOD"``. Omit it unless ``method_plan`` is
              ``"pairs"`` or ``"one_step_pairs"``.
            - ``l1_reg_aggreg``: L1 aggregation mode for methods where timing
              matters (``PR(GDPcap)``, ``PR-HR(Ecap)`` and ``AR(Ecap)``).
              ``"pre"`` aggregates regions before L1 computation. ``"post"``
              (default) computes on original regions, then aggregates.
            - ``lcia_method``: LCIA method(s) to characterize for EXIO
              processing (for example ``"pb_lcia"`` or
              ``["pb_lcia", "gwp100_lcia"]``). ``None`` means LCIA
              characterization is skipped. Defaults to ``None``. LCIA
              characterization is available only for EXIOBASE sources. To add
              a custom LCIA method, follow
              ``README_add_custom_lcia_characterization_matrices.txt`` in
              ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
              and pass the method file stem here.
            - ``reference_years``: Acquired rights (AR) methods reference
              year selector. Accepts a single year, list, or range. If
              omitted, AR routes use all historical years in the studied range
              up to the source registry historical cutoff. For EXIOBASE
              3.10.2 and OECD ICIO v2025, the cutoff is 2022; other supported
              MRIO sources use their own registry cutoff.
            - ``ssp_scenario``: SSP scenario name or list. Defaults to
              ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]`` and is applied
              when scenario dependent inputs are required.
            - ``projection_mode``: Projection policy for post historical
              years of L2 utilitarian (UT) methods (MRIO economic enacting
              metrics). Defaults to ``"regression"``. ``"regression"``
              projects UT inputs for future years. ``"historical_reuse"``
              reuses historical UT structures.
            - ``reg_window``: Historical regression fit window for regression
              mode. Provide it as ``range(start_year, end_year + 1)`` or as
              an explicit list of consecutive years in fit window order. When
              omitted, the source registry supplies the default fit window
              from the modeled year minimum through the source historical
              cutoff. For EXIOBASE 3.10.2 and OECD ICIO v2025, this resolves
              to 1995 to 2022; other supported MRIO sources use their own
              registry window.
            - ``l2_reuse_years``: Historical L2 reuse year selector used by
              all UT historical reuse routes. In
              ``projection_mode="historical_reuse"`` it applies to all UT
              methods; in ``projection_mode="regression"`` it applies to
              adjusted UT routes (``UT(FDa)``, ``UT(GVAa)``), which always
              use historical reuse as regression is not supported (would
              require regression on full MRIO). If omitted, defaults to
              ``reg_window`` when required.
        version_name: Required custom version token. The function reads
            ``preview_inter_method_weights/weights__<version_name>.csv``.
            The token must be non empty, contain only letters, digits, and
            underscores. The reserved token ``"equal_weight_default"`` is
            excluded.
        external_method: Optional external aSoCC method selector. Use
            ``{"l1_methods": [...]}`` for L1 functional units. For L2
            functional units use ``{"one_step_methods": [...]}`` and/or
            ``{"l1_l2_pairs": ["<l1_method>::<l2_method>", ...]}``.
            Omit this argument or pass ``None`` when using only native pyaesa
            deterministic aSoCC methods. Use ``prepare_external_inputs(...)``
            to import the external aSoCC README guidance and runnable CSV examples, then
            follow the imported README guidance for external method names, selector
            syntax, and deterministic or Monte Carlo external aSoCC CSV
            preparation.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.

    Returns:
        Paths and probabilities for the validated custom tree CSV, guide text,
        and rendered preview figure.

    Raises:
        ValueError: If the custom tree file no longer matches the requested
            method scope or if sibling edge weights fail probability
            validation.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Validate an edited tree for ``L2.c.b`` producing sector ``Paper`` and
        consuming region ``FR``::

            from pyaesa import preview_asocc_weight_tree

            preview_asocc_weight_tree(
                base_asocc_args={
                    "project_name": "demo",
                    "source": "exiobase_3102_ixi",
                    "fu_code": "L2.c.b",
                    "s_p": ["Paper"],
                    "r_c": ["FR"],
                },
                version_name="custom_v1",
            )
    """
    request = plan_inter_method_tree_request(
        base_asocc_args=base_asocc_args,
        external_method=external_method,
    )
    tree_path = inter_method_tree_path(proj_base=request.proj_base, version_name=version_name)
    frame, probabilities = load_valid_inter_method_tree_frame(
        candidates=request.candidates,
        custom_path=tree_path,
    )
    figure = normalize_figure_format(figure_format)
    figure_paths = tuple(
        render_inter_method_tree(
            frame=frame,
            figure_base_path=inter_method_preview_figure_base(
                proj_base=request.proj_base,
                version_name=version_name,
            ),
            output_format=str(figure["format"]),
            dpi=int(figure["dpi"]),
        )
    )
    labels = tuple(candidate.candidate_label for candidate in request.candidates)
    guide_path = write_inter_method_tree_guide(tree_csv_path=tree_path)
    return InterMethodTreeArtifacts(
        tree_csv_path=tree_path,
        guide_path=guide_path,
        figure_paths=figure_paths,
        candidates=labels,
        probabilities=tuple(float(value) for value in probabilities),
        summary_lines=(
            f"tree_csv={tree_path}",
            f"guide={guide_path}",
            f"figure_paths={[str(path) for path in figure_paths]}",
            f"candidate_count={len(labels)}",
        ),
    )
