# Package Overview

`pyaesa` supports AESA workflows from data preparation to deterministic and
uncertainty results. The package follows three calculation phases:

1. Phase A computes life-cycle assessment results, including IO-LCA from
   processed MRIO data.
2. Phase B computes allocated carrying capacities, with
   `aCC = aSoCC * CC`.
3. Phase C computes absolute sustainability ratios, with `ASR = LCA / aCC`.

![High level pyaesa package map](https://raw.githubusercontent.com/AESAtoolkit/pyaesa/main/images/fig-pyaesa-high-level.svg)

## Installation

`pyaesa` requires Python 3.11 to 3.14 and at least 4 GB of available RAM.

```bash
python -m pip install pyaesa
```

For data sources, prerequisite functions, public workflow functions, and study
objective routes, see the [Workflow Reference](workflow_reference.md).
