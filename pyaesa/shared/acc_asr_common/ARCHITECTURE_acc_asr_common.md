# Architecture: Shared aCC/ASR Internals (`shared/acc_asr_common/`)

## Purpose

`pyaesa/shared/acc_asr_common/` owns internal modules that are shared by
deterministic and uncertainty composite aCC and ASR families.

## Public Surface

There is no user facing public API under `pyaesa/shared/acc_asr_common/`.

Everything here is internal support code for:

- `deterministic_acc(...)`
- `deterministic_asr(...)`
- `uncertainty_acc(...)`
- `uncertainty_asr(...)`

## Responsibility Boundary

`pyaesa/shared/acc_asr_common/` owns:

- shared composite argument normalization for deterministic and uncertainty
  requests
- branch expansion across LCIA methods and CC families
- public persisted request payload construction for aCC/ASR deterministic and
  uncertainty runs
- deterministic downstream aSoCC share discovery, loading, and row selection
- deterministic branch identity and coverage guards shared by downstream
  branches

It does not own:

- aSoCC method resolution logic itself
- shared external aSoCC or external LCA file contracts
- carrying capacity data extraction
- IO-LCA computation
- package wide generic shared owners
- aCC prerequisite orchestration, which is owned by `pyaesa/acc/`
- ASR LCA numerator route normalization, which is owned by `pyaesa/asr/`
- ASR deterministic figure rendering and figure state, because aCC has no
  deterministic figure surface

## Internal Organization

| Path | Owner Contract |
| --- | --- |
| `branches/config.py` | Shared CC family configuration normalization and branch tokens |
| `branches/expand.py` | Shared CC branch expansion |
| `persistence/requests.py` | Public persisted payload builders for aCC/ASR runs |
| `scope/composite.py` | Shared composite MRIO and base aSoCC normalization |
| `deterministic/downstream/` | Shared downstream aSoCC share models, file IO, aSoCC share resolution, SSP transition metadata, and tabular contracts |
| `deterministic/state/scope_guard.py` | Deterministic branch identity and append coverage guards |

## Key Runtime Contracts

- this package is intentionally aCC and ASR scoped, not package wide
- shared branch, scope, and persisted request contracts serve deterministic
  and uncertainty aCC/ASR flows
- deterministic downstream support stays under the deterministic subtree
- public persisted payloads use public names such as
  `source`, grouping arguments, `aggreg_indices`, `base_asocc_args`,
  `base_cc_args`, `lca_args`, and `external_method`
- the first level is intentionally split between cross flow composite contracts
  (`branches/`, `persistence/`, and `scope/`) and deterministic only shared
  support (`deterministic/`)
- callers import deterministic only shared modules from the owning
  `deterministic/downstream/` and `deterministic/state/` modules rather than
  from a flatter mixed root
- shared SSP token normalization stays with the canonical selector owners
  under `pyaesa/shared/selectors/`, while
  `deterministic/downstream/scenarios.py` owns deterministic transition
  metadata and scenario label extraction
- `deterministic/downstream/` owns the full deterministic denominator support
  surface for aCC and ASR, including aSoCC share resolution, share file IO,
  transition metadata, and downstream tabular contracts
- selector builders used beyond the composite aCC/ASR boundary are owned by
  `pyaesa/shared/selectors/request_targets.py`, not by `shared/acc_asr_common/`
- owner specific modules are imported from their owning domains:
  - aSoCC selector derivation from `asocc`
  - shared external input contracts and loaders from `external_inputs`
  - generic selector normalization from `shared/selectors`
- aCC prerequisite orchestration is owned by `pyaesa/acc/deterministic/`.
  ASR deterministic prerequisites are owned by `pyaesa/asr/deterministic/`.
- ASR figure state and ASR LCA route normalization stay under `pyaesa/asr/`
  because no other downstream family consumes those contracts.

## Testing And Quality Gates

Package tests for behavior that consumes these shared modules live under:

| Consumer | Tests |
| --- | --- |
| Deterministic aCC and aCC uncertainty | `tests/package/acc/` |
| Deterministic ASR and ASR uncertainty | `tests/package/asr/` |
| Shared runtime contracts | `tests/package/shared/` when a reusable shared contract is exposed across owners |

For touched shared aCC ASR owners, run scoped `ruff`, scoped `pyright`, and
targeted package tests for every aCC or ASR public path that consumes the
touched module. Touched owners must keep 100 percent line and branch coverage.

