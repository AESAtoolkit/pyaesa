# Study objectives

Study objectives are study endpoints from the user perspective. A study objective corresponds to an *expected output* for the user. In `pyaesa`, five study objectives are currently available:

| Study objective | Corresponding output for the user|
| --- | --- |
| `A` | Life-cycle assessment (LCA/IO-LCA) |
| `B.0` | Dynamic carrying capacity (CC) |
| `B.1` | Allocated share of carrying capacities (aSoCC) |
| `B.2` | Allocated carrying capacities (aCC) |
| `C` | Absolute sustainability ratio (ASR) |

The figure below provides a simplified high-level overview of the package architecture. This includes (i) the mandatory prerequisites for all uses, and (ii) study objectives for which the user must focus solely on the desired objective. 

![High-level overview of pyaesa with main functions, study objectives, and prerequisites.](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-pyaesa-high-level.svg)

## `pyaesa` automatically calls necessary functions to reach the desired study objective

**It is very important for the user to understand that to reach a desired study objective**, `pyaesa` *automatically* runs upstream computations needed to produce the desired endpoint, i.e., to ensure that all previous outputs are available before running the downstream function providing the endpoint. This is illustrated via the green arrows (automatic nesting) in the figure above. Consequently, *the user must focus solely on the desired study objective*, and run the *single* relevant function.

For instance:
- For B.2 study objectives (i.e., aCC endpoints), the final entry function can auto run `pyaesa` owned
deterministic aSoCC and dynamic AR6 CC outputs when needed.
- For C study objectives (i.e., ASR endpoints)
with `pyaesa` owned IO-LCA, the final entry function can auto run `pyaesa` owned
aCC and IO-LCA outputs when needed. All results for A and B.x study objectives will therefore be automatically generated and available after completion.
- For ASR with external aSoCC or external LCA,
`prepare_external_inputs(...)` creates the external input folders, and users
must stage the external files before the ASR call.

Choose the **study objective** (i.e., the endpoint) and call the corresponding deterministic or uncertainty function directly.

```{list-table}
:header-rows: 1
:widths: 16 24 60

* - Study objective
  - Final entry function
  - Reference notebooks
* - (A) IO-LCA results
  - `deterministic_io_lca(...)`  
    `uncertainty_io_lca(...)`
  - {doc}`Deterministic IO-LCA tutorial </tutorials/study_objectives/(A) LCA/Phase_A_iolca_deterministic>`  
    {doc}`Uncertainty IO-LCA tutorial </tutorials/study_objectives/(A) LCA/Phase_A_iolca_uncertainty>`
* - (B.0) Dynamic AR6 climate change CC
  - `deterministic_ar6_cc(...)`  
    `uncertainty_ar6_cc(...)`
  - {doc}`Deterministic dynamic AR6 CC tutorial </tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_deterministic>`  
    {doc}`Uncertainty dynamic AR6 CC tutorial </tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_uncertainty>`
* - (B.1) aSoCC results
  - `deterministic_asocc(...)`  
    `uncertainty_asocc(...)`
  - {doc}`Deterministic aSoCC tutorial </tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_deterministic>`  
    {doc}`Uncertainty aSoCC tutorial </tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_uncertainty>`
* - (B.2) aCC results
  - `deterministic_acc(...)`  
    `uncertainty_acc(...)`
  - {doc}`Deterministic aCC tutorial </tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_deterministic>`  
    {doc}`Uncertainty aCC tutorial </tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_uncertainty>`
* - (C) ASR results with `pyaesa` owned IO-LCA
  - `deterministic_asr(...)`  
    `uncertainty_asr(...)`
  - {doc}`Deterministic ASR tutorial </tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic>`  
    {doc}`Uncertainty ASR tutorial </tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty>`
* - (C) ASR results with external aSoCC or external LCA
  - `deterministic_asr(...)`  
    `uncertainty_asr(...)`
  - {doc}`External input staging tutorial </tutorials/optional_workflows/external_asocc_lca_input_staging>`  
    {doc}`Deterministic ASR tutorial </tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic>`  
    {doc}`Uncertainty ASR tutorial </tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty>`
```

# What to do next

Check out [tutorials/study_objectives/1_functional_units_and_allocation_methods.md](1_functional_units_and_allocation_methods.md) before discovering the notebooks provided for each study objective available in `pyaesa`.
