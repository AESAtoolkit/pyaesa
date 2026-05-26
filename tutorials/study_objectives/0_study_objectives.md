# Study objectives

Study objectives are study endpoints from the user perspective. A study objective corresponds to an *expected output* for the user. In <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>, five study objectives are currently available:

| Study objective | Corresponding output for the user|
| --- | --- |
| `A` | Life-cycle assessment (LCA/IO-LCA) |
| `B.0` | Dynamic carrying capacity (CC) |
| `B.1` | Allocated share of carrying capacities (aSoCC) |
| `B.2` | Allocated carrying capacities (aCC) |
| `C` | Absolute sustainability ratio (ASR) |

![High-level overview of pyaesa with main functions, study objectives, and prerequisites.](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-pyaesa-high-level.svg)

## <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> automatically orchestrates functions to reach study objectives

**It is very important for the user to understand that to reach a desired study objective**, <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> automatically *orchestrates* the call of relevant functions to reach the desired endpoint.
This means that <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> automatically runs upstream computations needed to produce that endpoint, i.e., to ensure that all previous outputs are available before running the downstream function providing the endpoint. The user hence only needs to focus on *what is the study objective of interest*, and run the relevant function.

For instance:
- For B.2 study objectives (i.e., aCC endpoints), the final entry function can auto run <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned
deterministic aSoCC and dynamic AR6 CC outputs when needed.
- For C study objectives (i.e., ASR endpoints)
with <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA, the final entry function can auto run <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned
aCC and IO-LCA outputs when needed.
- For ASR with external aSoCC or external LCA,
`prepare_external_inputs(...)` creates the external input folders, and users
must stage the external files before the ASR call.

Choose the **study objective** (i.e., the endpoint) and call the corresponding deterministic or uncertainty function directly.\

<table>
  <thead>
    <tr>
      <th>Study objective</th>
      <th>Final entry function</th>
      <th>Reference notebook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(A)<br>IO-LCA results</td>
      <td>
        <code>deterministic_io_lca(...)</code><br>
        <code>uncertainty_io_lca(...)</code>
      </td>
      <td>
        <a href="phase_a_iolca_deterministic.html"><code>tutorials/study_objectives/(A) LCA/Phase_A_iolca_deterministic.ipynb</code></a><br>
        <a href="phase_a_iolca_uncertainty.html"><code>tutorials/study_objectives/(A) LCA/Phase_A_iolca_uncertainty.ipynb</code></a>
      </td>
    </tr>
    <tr>
      <td>(B.0)<br>Dynamic AR6 climate change CC</td>
      <td>
        <code>deterministic_ar6_cc(...)</code><br>
        <code>uncertainty_ar6_cc(...)</code>
      </td>
      <td>
        <a href="phase_b0_dynamic_cc_ar6_deterministic.html"><code>tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_deterministic.ipynb</code></a><br>
        <a href="phase_b0_dynamic_cc_ar6_uncertainty.html"><code>tutorials/study_objectives/(B.0) CC/Phase_B0_dynamic_CC_ar6_uncertainty.ipynb</code></a>
      </td>
    </tr>
    <tr>
      <td>(B.1)<br>aSoCC results</td>
      <td>
        <code>deterministic_asocc(...)</code><br>
        <code>uncertainty_asocc(...)</code>
      </td>
      <td>
        <a href="phase_b1_asocc_deterministic.html"><code>tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_deterministic.ipynb</code></a><br>
        <a href="phase_b1_asocc_uncertainty.html"><code>tutorials/study_objectives/(B.1) aSoCC/Phase_B1_asocc_uncertainty.ipynb</code></a><br>
      </td>
    </tr>
    <tr>
      <td>(B.2)<br>aCC results</td>
      <td>
        <code>deterministic_acc(...)</code><br>
        <code>uncertainty_acc(...)</code>
      </td>
      <td>
        <a href="phase_b2_acc_deterministic.html"><code>tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_deterministic.ipynb</code></a><br>
        <a href="phase_b2_acc_uncertainty.html"><code>tutorials/study_objectives/(B.2) aCC/Phase_B2_acc_uncertainty.ipynb</code></a>
      </td>
    </tr>
    <tr>
      <td>(C)<br>ASR results with <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> owned IO-LCA</td>
      <td>
        <code>deterministic_asr(...)</code><br>
        <code>uncertainty_asr(...)</code>
      </td>
      <td>
        <a href="phase_c_asr_deterministic.html"><code>tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic.ipynb</code></a><br>
        <a href="phase_c_asr_uncertainty.html"><code>tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty.ipynb</code></a>
      </td>
    </tr>
    <tr>
      <td>(C)<br>ASR results with external aSoCC or external LCA</td>
      <td>
        <code>deterministic_asr(...)</code><br>
        <code>uncertainty_asr(...)</code>
      </td>
      <td>
        <a href="../optional/external_asocc_lca_input_staging.html"><code>tutorials/optional_workflows/external_asocc_lca_input_staging.ipynb</code></a><br>
        <a href="phase_c_asr_deterministic.html"><code>tutorials/study_objectives/(C) ASR/Phase_C_asr_deterministic.ipynb</code></a><br>
        <a href="phase_c_asr_uncertainty.html"><code>tutorials/study_objectives/(C) ASR/Phase_C_asr_uncertainty.ipynb</code></a>
      </td>
    </tr>
  </tbody>
</table>

# What to do next

Check out [tutorials/study_objectives/1_functional_units_and_allocation_methods.md](1_functional_units_and_allocation_methods.md) before discovering the notebooks provided for each study objective available in <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>.
