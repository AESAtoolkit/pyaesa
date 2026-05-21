"""Shared LCIA MRIO enacting metric key selection."""


def required_lcia_metric_keys_for_context(
    *,
    context,
    registry,
) -> tuple[set[str], set[str]]:
    """Return required LCIA level-1 and level-2 MRIO enacting metric keys for one run context."""
    l1_keys: set[str] = set()
    l2_keys: set[str] = set()
    for l1_method in context.selected_l1:
        if not registry.method_requires_lcia(l1_method, None):
            continue
        l1_keys.update(
            registry.lcia_enacting_metric_l1_metrics(
                l1_method,
                level="L1",
            )
        )
    for l2_method in context.selected_l2_one_step:
        if not registry.method_requires_lcia(l2_method, context.fu_code):
            continue
        l1_keys.update(
            registry.lcia_enacting_metric_l1_metrics(
                l2_method,
                level="L2",
                fu_code=context.fu_code,
                l1_weighting=False,
            )
        )
        l2_keys.update(
            registry.lcia_enacting_metric_l2_metrics(
                l2_method,
                level="L2",
                fu_code=context.fu_code,
                l1_weighting=False,
            )
        )
    for l2_method, _l1_method in context.combined:
        if not registry.method_requires_lcia(l2_method, context.fu_code):
            continue
        l1_keys.update(
            registry.lcia_enacting_metric_l1_metrics(
                l2_method,
                level="L2",
                fu_code=context.fu_code,
                l1_weighting=True,
            )
        )
        l2_keys.update(
            registry.lcia_enacting_metric_l2_metrics(
                l2_method,
                level="L2",
                fu_code=context.fu_code,
                l1_weighting=True,
            )
        )
    return l1_keys, l2_keys
