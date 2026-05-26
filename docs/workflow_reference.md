# Workflow Reference

This reference maps the main <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> workflows from data preparation to AESA
outputs. It identifies the data sources used by each phase, the prerequisite
steps to run, the public functions that produce each output, the matching
tutorials, and indicative runtime and storage requirements.

## Data Sources By AESA Phase

| AESA phase or route | Data source | Used for |
| --- | --- | --- |
| Phase A IO-LCA | EXIOBASE 3.10.2 MRIO | <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA results and ASR numerators. |
| Phase B aSoCC allocation | EXIOBASE 3.10.2 or OECD ICIO v2025 MRIO | Allocation enacting metrics, final demand, production, value added, and environmental extensions after LCIA characterization when available. |
| Phase B aSoCC retrospective scope | World Bank population/GDP | Historical population and GDP allocation inputs. |
| Phase B aSoCC prospective scope | SSP population/GDP | Future population and GDP allocation inputs. |
| Phase B dynamic climate change CC | AR6 climate pathways | Dynamic climate change carrying capacity pathways. |

EXIOBASE is supported for `ixi` and `pxp` source variants. The examples use
EXIOBASE 3.10.2 but EXIOBASE 3.9.6 is also supported.

### Data Source Licenses And Terms

The GPL 3.0 license applies to <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> source code only.
Downloaded datasets remain governed by their original providers' terms and
conditions. For all data sources, cite the original provider following the
[recommended citation guide](../methodological_notes/recommended_citations.txt).

| Data source | Used in <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> | Terms and conditions | Link |
| --- | --- | --- | --- |
| EXIOBASE 3 | EXIOBASE 3.10.2 and 3.9.6 MRIO downloads for IO-LCA and allocation inputs. | <ul><li>Noncommercial use: allowed for academic use under the EXIOBASE license.</li><li>Commercial use: requires a commercial license.</li></ul> | [EXIOBASE terms and conditions](https://exiobase.eu/terms-of-use/) and [Zenodo EXIOBASE record](https://doi.org/10.5281/zenodo.3583070) |
| OECD ICIO v2025 | OECD ICIO MRIO downloads for allocation inputs. | <ul><li>Noncommercial use: allowed under OECD data terms.</li><li>Commercial use: allowed under OECD data terms.</li></ul> | [OECD ICIO page](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html) and [OECD terms and conditions](https://www.oecd-ilibrary.org/en/about/terms-conditions.html) |
| World Bank population/GDP | Historical population and GDP allocation inputs. | <ul><li>Noncommercial use: allowed under CC BY 4.0.</li><li>Commercial use: allowed under CC BY 4.0.</li></ul> | [World Bank terms and conditions](https://data.worldbank.org/summary-terms-of-use) |
| IIASA SSP population/GDP | Prospective population and GDP allocation inputs. | <ul><li>Noncommercial use: allowed under the IIASA Public License.</li><li>Commercial use: allowed within the licensed rights; adapted material is limited to scientific research, science communication, or policy consultancy. Reproducing substantial database portions is prohibited.</li></ul> | [IIASA SSP terms and conditions](https://ssp.apps.ece.iiasa.ac.at/documentation/license) |
| IIASA AR6 climate pathways | Dynamic climate change carrying capacity pathways. | <ul><li>Noncommercial use: allowed under the AR6 Scenario Explorer Public License.</li><li>Commercial use: allowed within the licensed rights. Reproducing substantial database portions is prohibited.</li></ul> | [IIASA AR6 license terms and conditions](https://data.ene.iiasa.ac.at/ar6/static/License/Legal.html) |

Current EXIOBASE LCIA coverage is `gwp100_lcia` and `pb_lcia`. EF3.1 is
available for non LCIA based allocation routes, but it is not currently
available for <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA or LCIA based allocation methods.
The {doc}`process data tutorial <tutorials/core_prerequisites/2_process_data>` explains the
detailed process for adding LCIA methods with EXIOBASE characterization
matrices and matching carrying capacity thresholds, either for private project
use or for public package submission.

## Core Prerequisites

Run `set_workspace(...)` once at the beginning of each Python session.

Run the download and processing functions needed before the selected study
endpoint: MRIO processing for aSoCC and <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA, and population/GDP
processing for allocation methods that use those inputs. Once run, they are
kept on disk and can be reused across studies. For dynamic AR6 CC and
downstream routes that use it, download AR6 raw inputs; the matching processed
AR6 scope can be created by the downstream dynamic route when it is missing so
it does not need to be run separately.

| Function | What it computes or prepares and writes | Disk space | Runtime |
| --- | --- | ---: | ---: |
| {func}`~pyaesa.set_workspace` | Creates the workspace, output root, and packaged prerequisite files. | 2 MB | <1 min |
| {func}`~pyaesa.download_mrio` | Downloads raw MRIO files for the selected source and years. | See MRIO tables | See MRIO tables |
| {func}`~pyaesa.download_pop_gdp` | Downloads raw World Bank and SSP population/GDP files. | 1 MB | 1 min |
| {func}`~pyaesa.download_ar6` | Downloads raw AR6 climate pathway and historical baseline files. | 210 MB | 1 min |
| {func}`~pyaesa.process_mrio` | Builds processed MRIO matrices, optional region or sector aggregation and disaggregation scopes, metadata, economic enacting metrics such as final demand and value added, and environmental enacting metrics after LCIA characterization. These outputs are reused by aSoCC allocation methods and <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA. | See MRIO tables | See MRIO tables |
| {func}`~pyaesa.process_pop_gdp` | Builds harmonized historical and SSP population/GDP tables, aligns country coverage to the supported MRIO scopes, records missing value treatment, and harmonizes GDP PPP units. These outputs are reused by retrospective and prospective aSoCC allocation methods. | 2 MB | <1 min |
| {func}`~pyaesa.process_ar6` | Builds retained and optionally harmonized AR6 pathway workbooks for dynamic climate change CC, including Kyoto gases and CO2 variables with and without AFOLU, category and SSP budget summaries, logs, and optional diagnostic figures. | 14 MB without figures; 63 MB with figures | 1 min without figures; figures add about 2 min |

### MRIO Storage And Runtime

Raw MRIO download storage and runtime by source:

| Source | Disk space | Runtime |
| --- | ---: | ---: |
| EXIOBASE 3.10.2 ixi | 260 MB for one year; 7.5 GB for 1995 to 2024 | 20 s for one year; 10 min for all years |
| EXIOBASE 3.10.2 pxp | 230 MB for one year; 6.7 GB for 1995 to 2024 | 20 s for one year; 10 min for all years |
| OECD ICIO v2025 | 470 MB for one bundle, for example 1995 to 2000; 2.2 GB for 1995 to 2022 | 1 min for one bundle, for example 1995 to 2000; 4 min for all years |

Processed MRIO output storage and runtime by source:

| Source | Disk space | Runtime |
| --- | ---: | ---: |
| EXIOBASE 3.10.2 ixi | 230 MB for one year; 6.7 GB for 1995 to 2024 | 1 min for one year; 25 min for all years |
| EXIOBASE 3.10.2 pxp | 280 MB for one year; 8.2 GB for 1995 to 2024 | 1 min for one year; 36 min for all years |
| OECD ICIO v2025 | 210 MB for one year; 5.8 GB for 1995 to 2022 | <1 min for one year; 5 min for all years |

The measurements were taken on Windows 11 with Python 3.14, an 11th Gen Intel
Core i7 1165G7 CPU, 32 GB RAM.

## AESA Functions

| Phase | Mode | Function | What it computes and writes |
| --- | --- | --- | --- |
| A | deterministic | {func}`~pyaesa.deterministic_io_lca` | Computes <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA result tables from processed EXIOBASE assets and figures. |
| A | uncertainty | {func}`~pyaesa.uncertainty_io_lca` | Monte Carlo IO-LCA run tables, summaries, source logs, and figures. |
| B | deterministic | {func}`~pyaesa.deterministic_asocc` | Allocated shares of carrying capacity (aSoCC) tables and figures. |
| B | uncertainty | {func}`~pyaesa.uncertainty_asocc` | aSoCC Monte Carlo run tables, summaries, source logs, figures by default, and Sobol outputs when requested. |
| B | deterministic | {func}`~pyaesa.deterministic_ar6_cc` | Dynamic AR6 climate change carrying capacity (CC) pathway tables and figures. |
| B | uncertainty | {func}`~pyaesa.uncertainty_ar6_cc` | AR6 CC Monte Carlo trajectory run tables, summaries, source logs, and figures. |
| B | deterministic | {func}`~pyaesa.deterministic_acc` | Allocated carrying capacity (aCC) tables as `aSoCC * CC` and figures. |
| B | uncertainty | {func}`~pyaesa.uncertainty_acc` | aCC Monte Carlo run tables, summaries, source logs, figures by default, and Sobol outputs when requested. |
| C | deterministic | {func}`~pyaesa.deterministic_asr` | Absolute sustainability ratio (ASR) tables as `LCA / aCC` and figures. |
| C | uncertainty | {func}`~pyaesa.uncertainty_asr` | ASR Monte Carlo run tables, summaries, source logs, figures by default, and Sobol outputs when requested. |

## Support Functions

| Function | What it prepares or writes |
| --- | --- |
| {func}`~pyaesa.disaggregate_asocc` | Published disaggregated aSoCC source outputs and optional figures for matching sector resolution between MRIO sources. Use the {doc}`disaggregation tutorial <tutorials/optional_workflows/disaggregate_asocc_mrio_sectors>` for the required deterministic prerequisite chain. |
| {func}`~pyaesa.prepare_external_inputs` | Project scoped external aSoCC and external LCA folders, guidance files, and templates for user provided data. |
| {func}`~pyaesa.write_asocc_weight_template` | Editable inter-method weights tree, guide, and preview figure for the aSoCC inter-method uncertainty source. |
| {func}`~pyaesa.preview_asocc_weight_tree` | Validated inter-method tree and preview figure for proposed custom weights. |

## Study Objectives And Routes

Study objectives are study endpoints from the user perspective. A study
objective corresponds to an expected output.

Choose the study objective, i.e. the endpoint, and call the corresponding
deterministic or uncertainty function directly. <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> automatically runs
upstream computations needed to produce that endpoint, i.e. to ensure that all
previous outputs are available before running the downstream function providing
the endpoint.

| Study objective | Corresponding output |
| --- | --- |
| `A` | Life-cycle assessment (LCA/IO-LCA) |
| `B.0` | Dynamic carrying capacity (CC) |
| `B.1` | Assigned share of carrying capacities (aSoCC) |
| `B.2` | Assigned carrying capacities (aCC) |
| `C` | Absolute sustainability ratio (ASR) |

Route setup:

1. Run {func}`~pyaesa.set_workspace` once for the workspace.
2. Download the raw data families needed by the study endpoint.
3. Run {func}`~pyaesa.process_mrio` and {func}`~pyaesa.process_pop_gdp` when
   the endpoint needs processed MRIO or population/GDP assets.
   Direct {func}`~pyaesa.process_ar6` runs are optional for dynamic AR6 CC, aCC, and ASR
   endpoints because those routes can provision the matching processed AR6
   scope when it is missing.

## Set Of Tutorials

The tutorial notebooks are split into reusable prerequisite data preparation
notebooks, study endpoint notebooks, and optional workflow notebooks.

### Core prerequisites tutorials

| Key | Notebook |
| --- | --- |
| Workspace | {doc}`tutorials/core_prerequisites/0_set_workspace.ipynb <tutorials/core_prerequisites/0_set_workspace>` |
| Download | {doc}`tutorials/core_prerequisites/1_download_data.ipynb <tutorials/core_prerequisites/1_download_data>` |
| Process | {doc}`tutorials/core_prerequisites/2_process_data.ipynb <tutorials/core_prerequisites/2_process_data>` |

### Study objectives tutorials

| Key | Notebook |
| --- | --- |
| Study objectives | {doc}`tutorials/study_objectives/0_study_objectives.md <tutorials/study_objectives/0_study_objectives>` |
| Functional units and allocation methods | {doc}`tutorials/study_objectives/1_functional_units_and_allocation_methods.md <tutorials/study_objectives/1_functional_units_and_allocation_methods>` |
| Phase A IO-LCA | {doc}`tutorials/study_objectives/(A) LCA/Phase_A_iolca_deterministic.ipynb <tutorials/study_objectives/(A) LCA/Phase_A_iolca_deterministic>`<br>{doc}`tutorials/study_objectives/(A) LCA/Phase_A_iolca_uncertainty.ipynb <tutorials/study_objectives/(A) LCA/Phase_A_iolca_uncertainty>` |
| Phase B.0 dynamic AR6 CC | {doc}`tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_deterministic.ipynb <tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_deterministic>`<br>{doc}`tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_uncertainty.ipynb <tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_uncertainty>` |
| Phase B.1 aSoCC | {doc}`tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_deterministic.ipynb <tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_deterministic>`<br>{doc}`tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_uncertainty.ipynb <tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_uncertainty>` |
| Phase B.2 aCC | {doc}`tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_deterministic.ipynb <tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_deterministic>`<br>{doc}`tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_uncertainty.ipynb <tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_uncertainty>` |
| Phase C ASR | {doc}`tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic.ipynb <tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic>`<br>{doc}`tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty.ipynb <tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty>` |

### Optional tutorials

| Main use | Notebook |
| --- | --- |
| disaggregation | {doc}`tutorials/optional_workflows/disaggregate_asocc_mrio_sectors.ipynb <tutorials/optional_workflows/disaggregate_asocc_mrio_sectors>` |
| inter-method weights | {doc}`tutorials/optional_workflows/custom_asocc_method_weights.ipynb <tutorials/optional_workflows/custom_asocc_method_weights>` |
| external aSoCC and external LCA input staging | {doc}`tutorials/optional_workflows/external_asocc_lca_input_staging.ipynb <tutorials/optional_workflows/external_asocc_lca_input_staging>` |

## Methodological References

| Location | Notes |
| --- | --- |
| {download}`methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf <../methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf>` | Functional units and allocation methods. |
| {download}`methodological_notes/methodological_note__acc_prospective.pdf <../methodological_notes/methodological_note__acc_prospective.pdf>` | Prospective allocation. |
| {download}`methodological_notes/methodological_note__acc_uncertainty_sources.pdf <../methodological_notes/methodological_note__acc_uncertainty_sources.pdf>` | Uncertainty sources. |
| {download}`methodological_notes/methodological_note__steady_state__dynamic_cc.pdf <../methodological_notes/methodological_note__steady_state__dynamic_cc.pdf>` | Definition of steady state and dynamic carrying capacities. |

{func}`~pyaesa.set_workspace` copies these methodological references into the
active workspace under `data_raw/methodological_notes/`.

Use the {doc}`API reference <api>` for exact signatures and parameter
contracts. Use the {doc}`tutorial page <tutorial>` for the notebook index and
complete tutorial content.
