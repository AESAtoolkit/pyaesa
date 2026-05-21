# Architecture: Shared Figure Primitives

## Purpose

`pyaesa/shared/figures/` owns reusable, family neutral figure primitives used by
deterministic and uncertainty figure renderers.

The package does not own scientific figure selection, row semantics, or output
scope decisions for any family. Those contracts stay in the family figure
owners that call these helpers.

## Public Surface

There is no user facing public API under `shared/figures/`.

Public figure controls are exposed only through family functions such as
`deterministic_asocc(...)`, `deterministic_acc(...)`,
`deterministic_asr(...)`, `deterministic_io_lca(...)`,
`deterministic_ar6_cc(...)`, and their uncertainty counterparts.

## Responsibility Boundary

| Owner | Contract |
| --- | --- |
| `request_validation.py` | Validate public figure request dictionaries. |
| `contracts.py` | Validate shared figure output format and DPI values. |
| `save.py` | Save matplotlib figures with the validated output format and DPI contract. |
| `paths.py`, `output_stems.py` | Build reusable path and stem fragments. |
| `scope_values.py`, `scope_support.py`, `selector_slices.py` | Normalize display values. |
| `lcia_scope.py`, `lcia_scopes.py`, `lcia_metadata.py` | Shared LCIA display scope helpers. |
| `deterministic_variant_*`, `variant_selection.py` | Select deterministic variant displays. |
| `multi_year_transitions.py`, `asocc_transition_*` | Shared transition marker helpers. |
| `violin_summary.py`, `uncertainty_run_values.py` | Uncertainty plotting helpers. |

Family figure packages own:

| Responsibility | Owning layer |
| --- | --- |
| Which public rows are figure eligible | Family figure scope planner. |
| Which deterministic or uncertainty tables are read | Family row reader. |
| Scientific labels and family specific grouping | Family product renderer. |
| Output folder roots and metadata updates | Family path and metadata owners. |
| Public figure defaults | Public family entry point. |

## Runtime Contracts

- Shared figure helpers must accept already scoped family data. They must not
  read family deterministic outputs directly unless the helper contract is
  explicitly a generic persisted output helper.
- Public figure option validation belongs in `request_validation.py`. Family
  entry points choose which shared option keys are allowed.
- Shared helpers may format labels, select display variants, run reusable
  panels, and save figures. They must not recompute scientific values.
- Family figure metadata remains in the family package. Shared figure helpers
  must return paths or rendered objects to the caller rather than writing family
  manifests themselves.

## Testing And Quality Gates

Shared figure tests live under `tests/package/shared/` when the behavior is
family neutral. Family specific figure behavior is tested in the consuming
family suites under `tests/package/`.

Required validation for shared figure changes:

- `python -m ruff check <touched shared figure paths> <touched tests>`
- `python -m ruff format --check <touched shared figure paths> <touched tests>`
- `python -m pyright <touched shared figure paths>`
- targeted package tests with line and branch coverage for the touched owner

Changes to shared figure request validation require package tests for every
public family function whose accepted figure options change. Touched owners
must keep 100 percent line and branch coverage.
