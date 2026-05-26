# Functional Units And Allocation Methods For AESA Study Objectives

This note helps choose both the functional unit (`fu_code`) and the allocation
method selectors used by:

- `deterministic_asocc(...)` and `uncertainty_asocc(...)`
- `deterministic_acc(...)` and `uncertainty_acc(...)`
- `deterministic_asr(...)` and `uncertainty_asr(...)`
- Only FU code: `deterministic_io_lca(...)` and `uncertainty_io_lca(...)`

The functional unit (FU) and allocation method tables of this file summarize
<a href="../../methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf">methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf</a>.
That appendix provides the definitions and mathematical expressions for each
allocation method available in <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>.

## Functional units

Choose first the functional unit according to the case study goal, scope, and system
boundaries.

The accepted code syntax is the dot format shown below, for example
`fu_code="L2.c.b"`. 

### Symbols

**Table 1. Sector and region selectors.**

| Symbol | Definition |
| --- | --- |
| `s_p` | Producing sector. |
| `r_p` | Producing region. |
| `r_f` | Final demand region where final demand occurs. |
| `r_c` | Total demand region where the studied outputs are first sold, whether the first sale is to intermediate demand or final demand. |

The sector selector `s_p` is common to all level 2 (L2: sector level) FUs.

The region selectors (`r_f`, `r_p`, `r_c`) are specific per FU.

**Table 2. Accounting boundary symbols.**

| Symbol | Definition |
| --- | --- |
| `FD` | Final demand. |
| `TD` | Total demand, i.e. final demand + intermediate demand/B2B. |
| `CBA_FD` | Consumption-based accounting of final demand -> Scopes 1, 2, 3. |
| `CBA_TD` | Consumption-based accounting of total demand -> Scopes 1, 2, 3 |
| `PBA` | Production-based accounting -> Scope 1 |


### Functional unit table

**Table 3. Functional units available by study objective.**

| `fu_code` | Allocation level | Study objective represented by the FU | Accounting boundary | Required selectors |
| --- | --- | --- | --- | --- |
| `"L1.a"` | L1 | Final demand of goods and services in region(s) `r_f` in year `t`. | `CBA_FD` | `r_f` |
| `"L1.b"` | L1 | Total production of goods and services by producing region(s) `r_p` in year `t`. | `PBA` | `r_p` |
| `"L2.a.a"` | L2 | Total production of goods and services by sector `s_p` in producing region(s) `r_p` directly supplied to final demand worldwide in year `t`. | `CBA_FD` | `s_p`, `r_p` |
| `"L2.a.b"` | L2 | Total production of goods and services by sector `s_p` in producing region(s) `r_p` in year `t`. | `CBA_TD` | `s_p`, `r_p` |
| `"L2.a.c"` | L2 | Total production of goods and services by sector `s_p` in producing region(s) `r_p` in year `t`. | `PBA` | `s_p`, `r_p` |
| `"L2.b.a"` | L2 | Total production of goods and services by sector `s_p` in producing region(s) `r_p` directly supplied to final demand in region(s) `r_f` in year `t`. | `CBA_FD` | `s_p`, `r_p`, `r_f` |
| `"L2.b.b"` | L2 | Total production of goods and services by sector `s_p` in producing region(s) `r_p` directly supplied to total demand in region(s) `r_c` in year `t`. | `CBA_TD` | `s_p`, `r_p`, `r_c` |
| `"L2.c.a"` | L2 | Final demand in region(s) `r_f` in year `t` of goods and services produced by sector `s_p`. | `CBA_FD` | `s_p`, `r_f` |
| `"L2.c.b"` | L2 | Total demand in region(s) `r_c` in year `t` of goods and services produced by sector `s_p`. | `CBA_TD` | `s_p`, `r_c` |

**Reading L2 FU codes:**
Within `L2`, the middle letter describes how the sector and region scope is
defined:

**Table 4. L2 FU code sector and region scopes.**

| FU family | Scope represented |
| --- | --- |
| `L2.a.*` | Output of sector `s_p` produced in region(s) `r_p`, without selecting a demand region. |
| `L2.b.*` | Output of sector `s_p` produced in region(s) `r_p` and supplied to a selected demand region. |
| `L2.c.*` | Demand in a selected region for outputs of sector `s_p`, aggregated across producing regions. |

The last letter describes the accounting boundary:

**Table 5. L2 FU code accounting boundary suffixes.**

| Suffix | Boundary | Use when the ASR numerator (LCA) measures |
| --- | --- | --- |
| `.a` | `CBA_FD` | Consumption-based accounting of outputs consumed by final demand (Scopes 1, 2, and 3). |
| `.b` | `CBA_TD` | Consumption-based accounting of outputs consumed by total demand, i.e. final demand and intermediate demand/B2B (Scopes 1, 2, and 3). |
| `.c` | `PBA` | Production-based accounting of direct burdens in the producing sector-region pair (Scope 1). |

N.B., `group_indices=True` is not allowed for `L2.a.b`, `L2.b.b`, or `L2.c.b`
because summing total demand CBA output rows can double count. For these FUs,
define the upstream MRIO aggregation and disaggregation scope before running the study
with `process_mrio(...)` arguments: `agg_reg`, `agg_sec`, and `agg_version`.

## Allocation methods

The definitions and mathematical expressions for each allocation method are
provided in
<a href="../../methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf">methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf</a>.
The allocation method labels below use the syntax accepted by <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>.

### Allocation paths overview

![Overview of allocation paths and methods across allocation levels and accounting system boundaries](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-asocc-paths.svg)

**Figure 1: Overview of UNCASExt/<span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> allocation paths and methods across
allocation levels and accounting system boundaries.** The figure summarizes the
allocation routes and method labels available across levels and accounting
system boundaries.

<sup>a</sup> `UT(FDa)` and `UT(GVAa)` are adjusted variants of `UT(FD)` and
`UT(GVA)` designed to reflect overlapping supply-chain attributions of utility.

<sup>b</sup> As the total direct and indirect GVA embodied in one unit of output
equals that unit's value, i.e. its FD, the one-step allocation paths
`UT(GVA)` to `UT(GVAa)` and `UT(FD)` to `UT(FDa)` yield an identical result,
here named one-step `UT(TD)`. This one-step total demand approach is therefore
neither consumption- nor production-anchored, analogously to one-step
`AR(E^{CBA_TD})`.

*L3 and L4 are not yet covered by <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> directly and can only be covered by
doing "manual" postprocessing of <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> outputs. L3 and L4 require case study
specific data. Examples include company revenues to scale a sector-level
allocation to a firm-level allocation by comparing company turnover to total
sector output in the MRIO, or activity data (e.g., passenger-kilometers
delivered) benchmarked against corresponding totals at the relevant MRIO level.
A future update of <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> will include the possibility to address these levels
via user provided L3 and L4 datasets.*

### Method label syntax

**Table 6. Allocation method label abbreviations.**

| Abbreviation | Meaning |
| --- | --- |
| `SP` | Sharing principle. |
| `EM` | Enacting metric. |
| `UT` | Utilitarian sharing principle. |
| `EG` | Egalitarian sharing principle. |
| `PR` | Prioritarian sharing principle. |
| `HR` | Historical responsibility. |
| `AR` | Acquired rights, also called grandfathering. |
| `Pop` | Population. |
| `GDPcap` | Gross domestic product per capita. |
| `E` | Environmental pressure. |
| `Ecap` | Per capita environmental pressure. |
| `cum` | Cumulative value over the responsibility period. |
| `FD` | Final demand. |
| `TD` | Total demand, i.e. final demand + intermediate demand/B2B. |
| `FDa` | Adjusted final demand, propagated through downstream supply chains with the Ghosh inverse. |
| `GVA` | Gross value added. |
| `GVAa` | Adjusted gross value added, propagated through upstream supply chains with the Leontief inverse. |

### Method selection arguments

Use `method_plan="default"` unless the study intentionally constrains the
method set and provides justifications to do so.
For L1 functional units, select subset of methods with `l1_methods`.
For L2 functional units, select subsets of one-step methods with `one_step_methods`,
two-step L2 methods with `two_step_methods`, and explicit L1 to L2 pairs of two-step methods with `l1_l2_pairs`.

**Table 7. Accepted `method_plan` values.**

| Value | Meaning |
| --- | --- |
| `"default"` | Use all methods available for the selected FU. |
| `"one_step"` | Use only selected one-step L2 methods. |
| `"two_steps"` | Use only selected two-step L2 methods combined with compatible L1 methods. |
| `"pairs"` | Use only explicit `l1_l2_pairs` entries of two-step methods. |
| `"one_step_pairs"` | Use selected one-step L2 methods plus explicit `l1_l2_pairs` entries. |

Explicit pair strings use the syntax `L1METHOD::L2METHOD`, for example
`EG(Pop)::UT(FDa)`. The part before `::` is the L1 method and the part after
`::` is the L2 method. Method labels must match the registry labels exactly.

### L1 allocation methods by FU

**Table 8. L1 allocation methods available by FU.**

| `fu_code` | L1 methods |
| --- | --- |
| `L1.a` | `EG(Pop)`, `PR(GDPcap)`, `PR-HR(Ecap,cum^{CBA_FD})`, `AR(E^{CBA_FD})`, `AR(Ecap^{CBA_FD})` |
| `L1.b` | `EG(Pop)`, `PR(GDPcap)`, `PR-HR(Ecap,cum^{PBA})`, `AR(E^{PBA})`, `AR(Ecap^{PBA})` |

### L2 allocation methods by FU

One-step methods allocate the L2 FU directly against the global carrying
capacity.

Two-step methods first allocate an L1 share against the global
carrying capacity, then allocate the selected L2 FU inside that L1 share.
When the `L2 in L1` weight is consumption-based, the `L1 vs global` weight
uses `L1.a`. When the `L2 in L1` weight is production-based, the `L1 vs global`
weight uses `L1.b`.

**Table 9. L2 allocation route equations.**

| Route | Allocation structure |
| --- | --- |
| one-step | `L2_aSoCC = L2 vs global` |
| two-step | `L2_aSoCC = L1 vs global * L2 in L1` |

**Table 10. L2 allocation methods available by FU.**

| `fu_code` | one-step L2 vs global methods | L2 in L1 methods for two-step allocation | L1 vs global FU used for two-step methods |
| --- | --- | --- | --- |
| `L2.a.a` | `UT(FD)`, `AR(E^{CBA_FD})` | `UT(FD)`, `AR(E^{CBA_FD})` | `L1.a` |
| `L2.a.b` | `UT(TD)`, `AR(E^{CBA_TD})` | `UT(FDa)` with `L1.a`; `UT(GVAa)` with `L1.b` | `UT(FDa)` uses `L1.a`; `UT(GVAa)` uses `L1.b` |
| `L2.a.c` | `UT(GVA)`, `AR(E^{PBA})` | `UT(GVA)`, `AR(E^{PBA})` | `L1.b` |
| `L2.b.a` | `UT(FD)`, `AR(E^{CBA_FD})` | `UT(FD)`, `AR(E^{CBA_FD})` | `L1.a` |
| `L2.b.b` | `UT(TD)`, `AR(E^{CBA_TD})` | `UT(FDa)` with `L1.a`; `UT(GVAa)` with `L1.b` | `UT(FDa)` uses `L1.a`; `UT(GVAa)` uses `L1.b` |
| `L2.c.a` | `UT(FD)`, `AR(E^{CBA_FD})` | `UT(FD)`, `AR(E^{CBA_FD})` | `L1.a` |
| `L2.c.b` | `UT(TD)`, `AR(E^{CBA_TD})` | `UT(FDa)` with `L1.a`; `UT(GVAa)` with `L1.b` | `UT(FDa)` uses `L1.a`; `UT(GVAa)` uses `L1.b` |

## What to do next

Now that you have been through [tutorials/study_objectives/0_study_objectives.md](0_study_objectives.md)
and [tutorials/study_objectives/1_functional_units_and_allocation_methods.md](1_functional_units_and_allocation_methods.md), go to the tutorial
corresponding to your study end objective. The available study objective notebooks are listed in
[tutorials/study_objectives/0_study_objectives.md](0_study_objectives.md).
