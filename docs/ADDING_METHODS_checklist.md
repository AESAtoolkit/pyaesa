# Developer Guide For Adding aSoCC Allocation Methods To The Package

This document explains how to add or change aSoCC allocation methods
implemented in the package and used by `deterministic_asocc(...)`.

The central rule is: one scientific method must have one canonical
registry definition, one canonical calculation owner, and one canonical public
output contract. Add code only where that responsibility is owned.

This guide is for package methods. If a user only wants to provide
their own completed aSoCC values, use the external aSoCC input route instead
of adding a package method to the package.

## 1. Start With The Scientific Contract

Write the method contract before editing code. The contract should answer these
questions.

| Question | Required answer |
| --- | --- |
| Public label | Exact label users select and see in output tables. |
| Sharing principle | Principle and optional subprinciple represented by the label. |
| Enacting metric | Metric that computes the shares. |
| Method level | L1, L2 direct, L2 support inside L1, or more than one form. |
| Supported FU codes | Exact FU codes where the method is mathematically defined. |
| Public axes | Required output identity columns. |
| Required inputs | Input families and their canonical data owners. |
| Time behavior | Historical, SSP, reference year, reuse, or projection rule. |
| Normalization | Expected sum rule or boundedness rule for scientific validation. |
| Downstream review | Consumers affected by the new output identity. |

The public label matters beyond display. Inter-method weighting parses package
and external method labels into sharing principle, optional subprinciple, and
enacting metric. Labels that should participate in inter-method uncertainty
must follow the canonical method label grammar:

```text
<principle>(<metric>)
<principle><subprinciple>(<metric>)
```

Examples include `EG(Pop)`, `PR(GDPcap)`, `PR-HR(Ecap,cum^{PBA})`,
`AR(E^{CBA_FD})`, and `UT(FD)`.

## 2. Decide Whether A Package Method Is Needed

Use this decision table before adding code.

| Need | Correct path |
| --- | --- |
| New formula using existing package metrics | Add the package method only. |
| New formula requiring a new MRIO metric | Add the processed metric first. |
| New formula requiring another data source | Add or extend that source owner first. |
| User supplied completed aSoCC values | Use external aSoCC inputs, not a package method. |
| Only a new method weight scenario | Use the inter-method weight CSV. |

Package method code is justified only when pyaesa can compute the method
from package inputs.

## 3. Understand The Method Forms

The package has three package method forms.

| Form | What it computes | Output owner |
| --- | --- | --- |
| L1 method | Allocation share across an L1 region boundary. | L1 deterministic tables. |
| L2 direct method | Final L2 versus global share. | `l2_vs_global` tables. |
| L2 support method inside L1 | L2 share multiplied by compatible L1. | Support and final tables. |

A method may need more than one registry row when it exists in more than one
form. For example, an L2 method can have a direct row with
`l1_weighting=False` and a support row with `l1_weighting=True`. Those rows are
not duplicates because they have different public route semantics.

Do not implement the same scientific route twice. In particular, an AR L1
method paired with the equivalent AR L2 method is represented by the canonical
direct AR L2 route, not by an extra two step recomposition.

## 4. Use The FU Axis Contract

The registry owns FU support and public identity axes. Do not infer axes from
method names in equation or writer code.

Current package aSoCC FU axes are:

| FU code | Typical public row axes for L2 package methods |
| --- | --- |
| `L1.a` | L1 final demand boundary, region axis usually `r_f`. |
| `L1.b` | L1 production boundary, region axis usually `r_p`. |
| `L2.a.a` | `s_p`, `r_p`; LCIA routes also include `impact`. |
| `L2.a.b` | `s_p`, `r_p`; LCIA routes also include `impact`. |
| `L2.a.c` | `s_p`, `r_p`; LCIA routes also include `impact`. |
| `L2.b.a` | `s_p`, `r_f`, `r_p`; LCIA routes also include `impact`. |
| `L2.b.b` | `s_p`, `r_c`, `r_p`; LCIA routes also include `impact`. |
| `L2.c.a` | `s_p`, `r_f`; LCIA routes also include `impact`. |
| `L2.c.b` | `s_p`, `r_c`; LCIA routes also include `impact`. |

When a new method uses a different identity axis, update the registry and the
publication contract deliberately. A writer must not add identity columns as a
repair after an equation returns a malformed frame.

## 5. Add Or Update Registry Rows

Registry rows are the source of truth for method discovery and capability.

| File | Responsibility |
| --- | --- |
| [`pyaesa/asocc/methods/registry/specs/l1.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/specs/l1.py) | Declarative L1 rows. |
| [`pyaesa/asocc/methods/registry/specs/l2.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/specs/l2.py) | Declarative L2 rows. |
| [`pyaesa/asocc/methods/registry/specs/all_specs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/specs/all_specs.py) | Combined method spec inventory. |
| [`pyaesa/asocc/methods/registry/build/build.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/build/build.py) | Registry assembly from declarative specs. |
| [`pyaesa/asocc/methods/registry/registry.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/registry.py) | Registry facade and supported family inventory. |
| [`pyaesa/asocc/methods/registry/model/types.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/model/types.py) | Typed registry fields. |
| [`pyaesa/asocc/methods/registry/model/input_requirements.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/model/input_requirements.py) | Input coverage requirements. |
| [`pyaesa/asocc/methods/registry/queries/resolve.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/queries/resolve.py) | Canonical label resolution. |
| [`pyaesa/asocc/methods/registry/model/family_checks.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/registry/model/family_checks.py) | Import time registry integrity checks. |

Each registry row must declare:

| Field | Meaning |
| --- | --- |
| `name` | Canonical public method label. |
| `level` | `L1` or `L2`. |
| `fu_code` | `None` for L1 rows; exact FU code for L2 rows. |
| `l1_weighting` | `False` for direct output; `True` for L2 support combined with L1 output. |
| `needs_lcia` | Method requires LCIA payloads. |
| `needs_pop` | Method requires population data. |
| `needs_gdp` | Method requires GDP data. |
| `needs_utility` | Method requires utility or propagation metrics. |
| `needs_rp` | Method requires responsibility period settings. |
| `indices` | Public row identity axes owned by the method output. |
| `l1_kind` | Required LCIA boundary kind, such as `CBA_FD` or `PBA`, when relevant. |
| `l2_weight_axis` | L1 axis that multiplies an L2 support share by L1 weights. |
| `expand_ar_years` | AR specific year expansion behavior where applicable. |

Registry rows must stay declarative. They should not contain equations,
fallback rules, file names, or downstream uncertainty behavior.

When a method family needs a new registry level grouping, add the grouping at
the registry build owner and keep query modules read only. Do not duplicate
method family lists in deterministic figures, disaggregation, uncertainty, ACC,
or ASR consumers.

## 6. Add Selection Reachability

Public selection resolves method labels before any computation. Update this
area only when the method should be selectable by users or included in a
default method plan.

| File | Responsibility |
| --- | --- |
| [`pyaesa/asocc/runtime/selection/resolve.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/selection/resolve.py) | Normalize user method selectors and method plans. |
| [`pyaesa/asocc/runtime/selection/plans.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/selection/plans.py) | Resolve method plans. |
| [`pyaesa/asocc/runtime/selection/pair_policy.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/selection/pair_policy.py) | Validate L1 and L2 pair compatibility. |
| [`pyaesa/asocc/orchestration/setup/request/selection.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/setup/request/selection.py) | Build setup selection. |

Selection code should answer only these questions:

| Question | Owner |
| --- | --- |
| Can the user select this label? | Registry plus selection resolver. |
| Is the method valid for this FU? | Registry row. |
| Is an L1 and L2 pair compatible? | Pair policy using `l1_kind` and `l2_weight_axis`. |
| Does the default plan include the method? | Selection plan owner. |

Selection code must not compute allocation values, reshape outputs, or repair
registry omissions.

## 7. Reuse Existing Enacting Metrics When Possible

Existing package method families already use these package inputs.

| Input family | Examples | Existing owners |
| --- | --- | --- |
| Population | `EG(Pop)`, `AR(Ecap...)`, `PR-HR(...)` | Pop GDP and scenario input owners. |
| GDP | `PR(GDPcap)` | Pop GDP loaders and PR equation owner. |
| LCIA impacts | AR and PR-HR methods | LCIA loaders and AR or PR owners. |
| Responsibility periods | `PR-HR(Ecap,cum...)` | Responsibility period and PR-HR owners. |
| Final demand | `UT(FD)`, `UT(FDa)`, `UT(TD)` | Processed MRIO and UT owners. |
| Gross value added | `UT(GVA)`, `UT(GVAa)` | Processed MRIO and UT owners. |
| Utility propagation | `UT(FDa)`, `UT(GVAa)` | Processed utility and adjusted UT owners. |

If the new formula can be expressed from these inputs, do not add process
outputs. Add only the equation and the registry route needed by the new method.

## 8. Add A New MRIO Metric Only At The Process Owner

If a method requires an MRIO derived metric that is not already processed,
add the metric where MRIO processing owns it. Do not derive it ad hoc inside
`deterministic_asocc(...)`.

| Area | Canonical owner |
| --- | --- |
| Processed MRIO metric construction | `pyaesa/process/mrios/utils/uncasext_metrics/` |
| Processed MRIO public entry point | [`pyaesa/process/mrios/process_mrio.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/process/mrios/process_mrio.py) |
| Processed MRIO metadata | [`pyaesa/process/mrios/utils/io/metadata.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/process/mrios/utils/io/metadata.py) |
| Deterministic aSoCC metric loading | [`pyaesa/asocc/data/load_mrio.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/data/load_mrio.py) |
| Per year required metric planning | [`pyaesa/asocc/orchestration/yearly/shared/year_inputs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/year_inputs.py) |
| Optional enacting metric output recording | `pyaesa/asocc/orchestration/yearly/enacting_metric/` |

Required implementation order:

1. Define the processed metric shape and labels in the process owner.
2. Write the metric during `process_mrio(...)`.
3. Record enough metadata for deterministic loading and validation.
4. Add a loader under [`pyaesa/asocc/data/load_mrio.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/data/load_mrio.py).
5. Add the metric to required per year loading in [`year_inputs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/year_inputs.py).
6. Pass it to the method family dispatch.
7. Use it in the equation owner.
8. Add package tests for processing, loading, and deterministic public output.

Do not run `download_mrio(..., refresh=True)` or `process_mrio(..., refresh=True)`
unless that exact refresh operation is explicitly requested.

## 9. Add A New Non MRIO Metric At Its Data Owner

If the method needs a new data source that is not an MRIO derived metric, the
data owner must be added before the allocation method.

| Source type | Expected owner |
| --- | --- |
| Population or GDP variant | Pop GDP download, process, and loader owners. |
| LCIA characterization or carrying capacity support | LCIA data preparation owners. |
| Responsibility period table | Responsibility period preparation and PR-HR loading owners. |
| User supplied input | External input preparation route, not package method code. |

Allocation equations should receive validated data structures. They should not
download files, parse raw user files, or repair missing processed data.

## 10. Implement The Equation At The Family Owner

Scientific calculation belongs under [`pyaesa/asocc/methods/`](https://github.com/AESAtoolkit/pyaesa/tree/main/pyaesa/asocc/methods).

| File or folder | Responsibility |
| --- | --- |
| [`methods/compute_l1.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/compute_l1.py) | Dispatch L1 methods by registry family. |
| [`methods/compute_l2.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/compute_l2.py) | Dispatch L2 methods by registry family. |
| [`methods/run_ar.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/run_ar.py) | AR runtime, reference year routing, and AR caches. |
| [`methods/run_ut.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/run_ut.py) | UT runtime and preweight application. |
| [`methods/equations/`](https://github.com/AESAtoolkit/pyaesa/tree/main/pyaesa/asocc/methods/equations) | Pure equation functions for method families. |
| [`methods/equations/ar_result_indexing.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/equations/ar_result_indexing.py) | AR impact and reference year index levels. |
| [`methods/equations/ar_nan_outputs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/methods/equations/ar_nan_outputs.py) | Empty AR row shapes. |

Equation rules:

1. Accept explicit validated inputs.
2. Use vectorized pandas or NumPy operations for numeric work.
3. Return an indexed wide `DataFrame` with one year column.
4. Keep public identity axes in the index returned by the equation or route
   owner.
5. Raise clear errors only for reachable invalid inputs owned by that function.
6. Do not write files.
7. Do not make downstream writers repair missing axes or replace invalid values.

If a family dispatch becomes hard to read, add a family owner module instead
of adding large conditional blocks inside unrelated family files.

## 11. Connect Yearly Orchestration

Yearly orchestration loads the minimum input set required by the selected
methods, slices it to the requested public scope, and calls the method owners.

| Area | Responsibility |
| --- | --- |
| [`orchestration/yearly/run_year.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/run_year.py) | Per year input loading and dispatch. |
| [`orchestration/yearly/shared/year_inputs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/year_inputs.py) | Required annual inputs. |
| [`orchestration/yearly/l1/`](https://github.com/AESAtoolkit/pyaesa/tree/main/pyaesa/asocc/orchestration/yearly/l1) | L1 execution and L1 table storage. |
| [`orchestration/yearly/l2/`](https://github.com/AESAtoolkit/pyaesa/tree/main/pyaesa/asocc/orchestration/yearly/l2) | L2 execution, weighting, and recomposition. |
| [`orchestration/yearly/enacting_metric/`](https://github.com/AESAtoolkit/pyaesa/tree/main/pyaesa/asocc/orchestration/yearly/enacting_metric) | Optional intermediate metric recording. |
| [`orchestration/yearly/shared/scenario_processing.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/scenario_processing.py) | Per scenario execution. |
| [`orchestration/yearly/shared/scenario_routing.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/scenario_routing.py) | Scenario partition routing. |

The usual package method flow is:

1. Registry flags tell setup which input families are needed.
2. [`year_inputs.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/yearly/shared/year_inputs.py) loads only required metric keys.
3. L1 or L2 yearly owners call the family dispatch.
4. L2 support rows are multiplied by compatible L1 weights if needed.
5. Public writers receive already shaped method output frames.

Do not load all processed metrics just because one new method needs one extra
input. Extend the required metric planner so run and deterministic runs stay
bounded in memory.

## 12. Declare Time Behavior

Time behavior must be planned before yearly computation.

| Behavior | Owner |
| --- | --- |
| Historical coverage | Setup year plan and processed MRIO metadata readers. |
| SSP population or GDP routing | Yearly scenario input owners. |
| Reference years | AR owners and setup reference year validation. |
| Historical reuse | Projection reuse owners. |
| Regression projection | Projection config, payload, and regression owners. |

A method that has no projection behavior must not receive future year values by
accident. A method that is scenario invariant should keep that identity clear
in public rows while scenario partition files include it only where routing
requires it.

## 13. Publish Through The Existing Output Contract

Public aSoCC tables are wide tables. Identifier columns come first, followed
by year columns.

| Owner | Responsibility |
| --- | --- |
| [`runtime/output/contracts.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/output/contracts.py) | Output descriptors, identifier columns, and file tokens. |
| [`runtime/paths/published.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/paths/published.py) | Public deterministic roots. |
| [`runtime/paths/deterministic.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/runtime/paths/deterministic.py) | Deterministic logs and scope paths. |
| [`orchestration/write/writers/allocations.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/write/writers/allocations.py) | Table writing. |
| [`orchestration/write/metadata/payload.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/write/metadata/payload.py) | Deterministic scope manifest payloads. |
| [`orchestration/setup/reuse/completed_run_policy.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/orchestration/setup/reuse/completed_run_policy.py) | Reuse and append policy. |

Publication rules:

1. The writer receives complete scientific identity from method output.
2. `logs/scope_manifest.json` discovers completed deterministic scopes and is
   the deterministic reuse authority.
3. Public output scanning must not infer run completion.
4. `refresh=True` deletes only the resolved scope after file handlers are closed.
5. A new public identity axis requires explicit updates to output contracts,
   metadata, tests, and downstream consumers.

## 14. Keep Figures, Disaggregation, And Uncertainty As Review Gates

These areas are not part of the first implementation path for every method.
Review them only when the method contract affects them.

| Change | Required review |
| --- | --- |
| New method label or public identity axis | Deterministic figures and figure scope matching. |
| New final L2 row semantics | `disaggregate_asocc(...)` eligibility and row matching. |
| New LCIA based method | aSoCC LCIA uncertainty and ASR or ACC consumers. |
| New projection or reuse route | Projection uncertainty and deterministic reuse identity. |
| New reference year behavior | Reference year uncertainty. |
| New inter-method selectable method | Inter-method tree and method weight export. |
| New ACC or ASR used identity | Downstream deterministic and uncertainty row loading. |

Downstream code must consume the canonical registry or public deterministic
rows. It must not maintain a second method list.

## 15. Add Tests That Exercise Public Reachability

Package tests for aSoCC live under [`tests/package/asocc/`](https://github.com/AESAtoolkit/pyaesa/tree/main/tests/package/asocc). Scientific
allocation validation lives under [`tests/allocation_equation_validation/`](https://github.com/AESAtoolkit/pyaesa/tree/main/tests/allocation_equation_validation).

| Change | Required tests |
| --- | --- |
| Registry row | Registry validation and public selection reachability. |
| Selection behavior | Public `deterministic_asocc(...)` or selection resolver tests. |
| Equation formula | Equation success path and reachable invalid input path. |
| New MRIO metric | Process output, metadata, aSoCC loader, and public deterministic output tests. |
| New non MRIO metric | Data owner tests plus deterministic public output tests. |
| L2 support recomposition | Pair selection and recomposed `l2_vs_global` output tests. |
| Time behavior | Historical, SSP, reference year, reuse, or projection tests as applicable. |
| Public output identity | Writer, metadata, reuse, and refresh tests. |
| Figure, disaggregation, uncertainty impact | Targeted tests for affected contracts. |

Tests must use public functions or realistic file backed flows. Do not
monkeypatch private control flow to force unreachable branches. If a branch
cannot be reached from normal public behavior, delete the branch and the test.

## 16. Run Scientific Validation

For any method logic change, run allocation equation validation in addition to
package tests.

- [`tests/allocation_equation_validation/Allocation_Validation.ipynb`](https://github.com/AESAtoolkit/pyaesa/blob/main/tests/allocation_equation_validation/Allocation_Validation.ipynb):
  interactive validation workflow.
- [`tests/allocation_equation_validation/validation_function/test_alloc_methods.py`](https://github.com/AESAtoolkit/pyaesa/blob/main/tests/allocation_equation_validation/validation_function/test_alloc_methods.py):
  validation runner for sum rules.

Validation evidence should cover:

1. Every FU code declared in the registry.
2. L1, L2 direct, and L2 support forms where applicable.
3. LCIA methods and impact rows when the method is LCIA based.
4. SSP scenarios when population or GDP scenarios affect the method.
5. Historical and future years when reuse or projection is supported.
6. Reference years when AR behavior is involved.
7. The method specific normalization rule.

## 17. Update Documentation

Update documentation in the same change set as code.

| Documentation | When to update |
| --- | --- |
| [`pyaesa/asocc/ARCHITECTURE_asocc.md`](https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/ARCHITECTURE_asocc.md) | Method ownership or runtime changes. |
| [`docs/ADDING_METHODS_checklist.md`](ADDING_METHODS_checklist.md) | Developer process or package method addition path changes. |
| [`docs/api.rst`](api.rst) | Public API signature or user visible behavior changes. |
| Tutorials under `tutorials/` | User examples or method availability changes. |
| External template text files | External input schema or accepted label changes. |

Architecture docs must describe package contracts in present state terms.
Avoid change history and planning notes.

## 18. Closeout Gate

Before handoff:

1. Run `python -m ruff check <touched package paths> <touched tests>`.
2. Run `python -m ruff format --check <touched package paths> <touched tests>`.
3. Run `python -m pyright <touched package paths>`.
4. Run targeted `pytest` with `--cov=<touched owner> --cov-branch`.
5. Keep touched owners at 100 percent line and branch coverage.
6. Run allocation equation validation for scientific method changes.
7. Confirm all code paths are reachable by public functions or
   realistic file flows.
8. Confirm no duplicate validation remains for invariants enforced upstream.
9. Confirm path construction stays in path owner modules.
10. Confirm affected downstream areas were reviewed and either updated or
    documented as unaffected.
