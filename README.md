# ![image description](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-pyaesa-logo.png)

**Documentation:** [pyaesa.readthedocs.io](https://pyaesa.readthedocs.io/)

<span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span> is a Python package for absolute environmental sustainability
assessment (AESA) workflows. It supports data download, data
processing, deterministic calculations, figure rendering, Monte Carlo
uncertainty and Sobol variance.

The package follows the three AESA phases described in the JRC guidance. The
calculation chain is:

1. Phase A builds life cycle assessment (LCA) results.
2. Phase B builds allocated carrying capacity (aCC) results:
   `aCC = aSoCC * CC`.
3. Phase C builds absolute sustainability ratio (ASR) results:
   `ASR = LCA / aCC`.

`aSoCC` means allocated share of carrying capacity. `CC` means carrying
capacity. CC can be static, for example with LCIA methods PB LCIA or EF3.1,
or dynamic through AR6 climate change pathways.

## Installation

Install the package from PyPI:

```bash
python -m pip install pyaesa
```

For local development from a repository clone, install in editable mode:

```bash
python -m pip install -e .
```
pyaesa requires at least **4 GB of available RAM** to run.

## Community Feature Ideas

<span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span> uses the GitHub Discussions **Ideas** category to collect feature ideas and development priorities from users.

Use Ideas to propose new features, upvote existing proposals, and comment with your use case, data source, expected workflow, or implementation constraints:

- [Propose or upvote feature ideas](https://github.com/AESAtoolkit/pyaesa/discussions/categories/ideas)

To propose directly code modifications, see `CONTRIBUTING.md`.

## High-level overview of the package

The figure below provides a high-level overview of the package, including main **data sources**, high-level public **functions** and **study objectives** supported by the package. More details are provided in the next sections.

![figure-high-level](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-pyaesa-high-level.svg)

## Data Sources By AESA Phase

| AESA phase or route | Data source | Used for |
| --- | --- | --- |
| Phase A IO-LCA | EXIOBASE 3.10.2 MRIO | pyaesa owned IO-LCA results and ASR numerators. |
| Phase B aSoCC allocation | EXIOBASE 3.10.2 or OECD ICIO v2025 MRIO | Allocation enacting metrics, final demand, production, value added, and environmental extensions after LCIA characterization when available. |
| Phase B aSoCC retrospective scope | World Bank population/GDP | Historical population and GDP allocation inputs. |
| Phase B aSoCC prospective scope | SSP population/GDP | Future population and GDP allocation inputs. |
| Phase B dynamic climate change CC | AR6 climate pathways | Dynamic climate change carrying capacity pathways. |

EXIOBASE is supported for `ixi` and `pxp` source variants. The
examples use EXIOBASE 3.10.2 but EXIOBASE 3.9.6 is also supported.

Current EXIOBASE LCIA coverage is `gwp100_lcia` and `pb_lcia`. EF3.1 is
available for non LCIA based allocation routes, but it is not currently
available for pyaesa owned IO-LCA or LCIA based allocation methods.
`tutorials/core_prerequisites/2_process_data.ipynb` explains the detailed
process for adding LCIA methods with EXIOBASE characterization matrices and
matching carrying capacity thresholds, either for private project use or for
public package submission.

## Public Workflow Function Map

### Core Prerequisites To Run Once

Run `set_workspace(...)` once at the beginning of each Python session.

Run the download and processing functions needed before the selected study endpoint: MRIO processing for aSoCC and <span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span> owned IO-LCA, and population/GDP processing for allocation methods that use those inputs (once run they are kept on disk and can be reused across studies). For dynamic AR6 CC and downstream routes that use it, download AR6 raw inputs; the matching processed AR6 scope can be
created by the downstream dynamic route when it is missing so it does not need to be run separately.

<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>What it computes or prepares and writes</th>
      <th>Disk space</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>set_workspace(...)</code></td>
      <td>Creates the workspace, output root, and packaged prerequisite files.</td>
      <td>2 MB</td>
      <td>&lt;1 min</td>
    </tr>
    <tr>
      <td><code>download_mrio(...)</code></td>
      <td>Downloads raw MRIO files for the selected source and years.</td>
      <td>See table below.</td>
      <td>See table below.</td>
    </tr>
    <tr>
      <td><code>download_pop_gdp(...)</code></td>
      <td>Downloads raw World Bank and SSP population/GDP files.</td>
      <td>1 MB</td>
      <td>1 min</td>
    </tr>
    <tr>
      <td><code>download_ar6(...)</code></td>
      <td>Downloads raw AR6 climate pathway and historical baseline files.</td>
      <td>210 MB</td>
      <td>1 min</td>
    </tr>
    <tr>
      <td><code>process_mrio(...)</code></td>
      <td>
        Builds processed MRIO matrices, optional grouped region or sector
        scopes, metadata, economic enacting metrics such as final demand and
        value added, and environmental enacting metrics after LCIA
        characterization. These outputs are reused by aSoCC allocation methods
        and pyaesa owned IO-LCA.
      </td>
      <td>See table below.</td>
      <td>See table below.</td>
    </tr>
    <tr>
      <td><code>process_pop_gdp(...)</code></td>
      <td>
        Builds harmonized historical and SSP population/GDP tables, aligns
        country coverage to the supported MRIO scopes, records missing value
        treatment, and harmonizes GDP PPP units. These outputs are reused by
        retrospective and prospective aSoCC allocation methods.
      </td>
      <td>2 MB</td>
      <td>&lt;1 min</td>
    </tr>
    <tr>
      <td><code>process_ar6(...)</code></td>
      <td>
        Builds retained and optionally harmonized AR6 pathway workbooks for
        dynamic climate change CC, including Kyoto gases and CO2 variables
        with and without AFOLU, category and SSP budget summaries, logs, and
        optional diagnostic figures.
      </td>
      <td>14 MB without figures; 63 MB with figures.</td>
      <td>1 min without figures; figures add about 2 min.</td>
    </tr>
  </tbody>
</table>

MRIO raw download storage and runtime by source:

| Source | Disk space | Runtime |
| --- | ---: | ---: |
| EXIOBASE 3.10.2 ixi | 260 MB for one year; 7.5 GB for 1995 to 2024 | 20 s for one year; 10 min for all years |
| EXIOBASE 3.10.2 pxp | 230 MB for one year; 6.7 GB for 1995 to 2024 | 20 s for one year; 10 min for all years |
| OECD ICIO v2025 | 470 MB for one bundle, for example 1995 to 2000; 2.2 GB for 1995 to 2022 | 1 min for one bundle, for example 1995 to 2000; 4 min for all years |

MRIO processed output storage and runtime by source:

| Source | Disk space | Runtime |
| --- | ---: | ---: |
| EXIOBASE 3.10.2 ixi | 230 MB for one year; 6.7 GB for 1995 to 2024 | 1 min for one year; 25 min for all years |
| EXIOBASE 3.10.2 pxp | 280 MB for one year; 8.2 GB for 1995 to 2024 | 1 min for one year; 36 min for all years |
| OECD ICIO v2025 | 210 MB for one year; 5.8 GB for 1995 to 2022 | <1 min for one year; 5 min for all years |

*The measurements were taken on Windows 11 with Python 3.14, an 11th Gen
Intel Core i7 1165G7 CPU, 32 GB RAM.*

### AESA Functions

| Phase | Mode | Function | What it computes and writes |
| --- | --- | --- | --- |
| A | deterministic | `deterministic_io_lca(...)` | Computes pyaesa owned IO-LCA result tables from processed EXIOBASE assets and figures. |
| A | uncertainty | `uncertainty_io_lca(...)` | Computes Monte Carlo IO-LCA run tables, summaries, source logs, and figures. |
| B | deterministic | `deterministic_asocc(...)` | Computes allocated shares of carrying capacity (aSoCC) tables and figures. |
| B | uncertainty | `uncertainty_asocc(...)` | Computes aSoCC Monte Carlo run tables, summaries, source logs, figures, and Sobol outputs when requested. |
| B | deterministic | `deterministic_ar6_cc(...)` | Computes dynamic AR6 climate change carrying capacity (CC) pathway tables and figures. |
| B | uncertainty | `uncertainty_ar6_cc(...)` | Computes AR6 CC Monte Carlo trajectory run tables, summaries, source logs, and figures. |
| B | deterministic | `deterministic_acc(...)` | Computes allocated carrying capacity (aCC) tables as `aSoCC * CC` and figures. |
| B | uncertainty | `uncertainty_acc(...)` | Computes aCC Monte Carlo run tables, summaries, source logs, figures, and Sobol outputs when requested. |
| C | deterministic | `deterministic_asr(...)` | Computes absolute sustainability ratio (ASR) tables as `LCA / aCC` and figures. |
| C | uncertainty | `uncertainty_asr(...)` | Computes ASR Monte Carlo run tables, summaries, source logs, figures, and Sobol outputs when requested. |

### Support Functions

<table>
  <thead>
    <tr>
      <th>Support function</th>
      <th>What it prepares or writes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>disaggregate_asocc(...)</code></td>
      <td>
        Published disaggregated aSoCC source outputs and optional figures for
        matching sector resolution between MRIO sources. Use the dedicated
        disaggregation notebook for the required deterministic prerequisite
        chain.
      </td>
    </tr>
    <tr>
      <td><code>prepare_external_inputs(...)</code></td>
      <td>
        Project scoped external aSoCC and external LCA folders, README files,
        and templates for user provided data.
      </td>
    </tr>
    <tr>
      <td><code>write_asocc_weight_template(...)</code></td>
      <td>
        Editable inter-method weights tree, guide, and preview figure for the
        aSoCC inter-method uncertainty source.
      </td>
    </tr>
    <tr>
      <td><code>preview_asocc_weight_tree(...)</code></td>
      <td>
        Validated inter-method tree and preview figure for proposed custom
        weights.
      </td>
    </tr>
  </tbody>
</table>

## Study objectives and recommended routes

### Route Setup

1. Run `set_workspace(...)` once for the workspace.
2. Download the raw data families needed by the study endpoint.
3. Run `process_mrio(...)` and `process_pop_gdp(...)` when the endpoint needs
   processed MRIO or population/GDP assets. Direct `process_ar6(...)` runs are
   optional for dynamic AR6 CC, aCC, and ASR endpoints because those routes can
   provision the matching processed AR6 scope when it is missing.

### Selecting and reaching a study objective

Study objectives are study endpoints from the user perspective. A study objective corresponds to an *expected output* for the user.\
In <span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span>, five study objectives are currently available:

| Study objective | Corresponding output |
| --- | --- |
| `A` | Life-cycle assessment (LCA/IO-LCA) |
| `B.0` | Dynamic carrying capacity (CC) |
| `B.1` | Assigned share of carrying capacities (aSoCC) |
| `B.2` | Assigned carrying capacities (aCC) |
| `C` | Absolute sustainability ratio (ASR) |

Choose the **study objective** (i.e., the endpoint) and call the corresponding deterministic or uncertainty function directly. <span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span> automatically runs upstream computations needed to produce that endpoint, i.e., to ensure that all previous outputs are available before running the downstream function providing the endpoint. The user hence only needs to focus on *what is the study objective of interest*, and run the relevant function.


Check out `tutorials/study_objectives/0_study_objectives.md` to understand how to select and reach study objectives in <span style="color:#366e9c"><strong><tt>py</tt></strong></span><span style="color:#c83737"><strong><tt>aesa</tt></strong></span>.

## Set of tutorials

The README is the tutorial navigator. The tutorial notebooks are
split into reusable prerequisite data preparation notebooks, study endpoint
notebooks, and optional workflow notebooks.

### Core prerequisites tutorials:

| Key | Notebook |
| --- | --- |
| Workspace | `tutorials/core_prerequisites/0_set_workspace.ipynb` |
| Download | `tutorials/core_prerequisites/1_download_data.ipynb` |
| Process | `tutorials/core_prerequisites/2_process_data.ipynb` |

### Study objectives tutorials:

| Key | Notebook |
| --- | --- |
| Study objectives | `tutorials/study_objectives/0_study_objectives.md` |
| Functional units and allocation methods | `tutorials/study_objectives/1_functional_units_and_allocation_methods.md` |
| Phase A IO-LCA | `tutorials/study_objectives/(A) LCA/Phase_A_iolca_deterministic.ipynb` |
|                | `tutorials/study_objectives/(A) LCA/Phase_A_iolca_uncertainty.ipynb` |
| Phase B.0 dynamic AR6 CC | `tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_deterministic.ipynb` |
|                          | `tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_uncertainty.ipynb` |
| Phase B.1 aSoCC | `tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_deterministic.ipynb` |
|                 | `tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_uncertainty.ipynb` |
| Phase B.2 aCC | `tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_deterministic.ipynb` |
|               | `tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_uncertainty.ipynb` |
| Phase C ASR | `tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic.ipynb` |
|             | `tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty.ipynb` |

### Optional tutorials:

| Main use | Notebook |
| --- | --- |
| disaggregation | `tutorials/optional_workflows/disaggregate_asocc_mrio_sectors.ipynb` |
| inter-method weights | `tutorials/optional_workflows/custom_asocc_method_weights.ipynb` |
| external aSoCC and external LCA input staging | `tutorials/optional_workflows/external_asocc_lca_input_staging.ipynb` |

### Methodological references:

| Tutorial folder reference | Notes |
| --- | --- |
| `methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf` | Functional units and allocation methods. |
| `methodological_notes/methodological_note__acc_prospective.pdf` | Prospective allocation. |
| `methodological_notes/methodological_note__acc_uncertainty_sources.pdf` | Uncertainty sources. |
| `methodological_notes/methodological_note__steady_state__dynamic_cc.pdf` | Definition of steady state and dynamic carrying capacities. |

`set_workspace(...)` copies these tutorial folder references into the active
workspace under `data_raw/methodological_notes/`.

Use the API reference in `docs/api.rst` for exact signatures and parameter
contracts. Use `docs/tutorial.rst` for the notebook index.
