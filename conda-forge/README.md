# conda-forge Packaging

This directory stores the source repository copy of the conda-forge recipe used
to submit `pyaesa` for public distribution through `conda-forge/staged-recipes`.
The recipe is maintained here so release metadata, runtime dependencies, license
metadata, and the published source distribution hash can be reviewed with the
package release.

## Current Target

| Field | Value |
| --- | --- |
| Package | `pyaesa` |
| Release | `1.2.1` |
| Source archive | PyPI source distribution |
| Recipe path | `recipe/meta.yaml` |
| Staged recipes path | `recipes/pyaesa/meta.yaml` |

## Maintainer Workflow

For each conda-forge submission, the PyPI source distribution must be published
first. The conda recipe should then reference that immutable archive and its
SHA256 hash. Public conda distribution is handled by conda-forge review and the
feedstock created after the staged recipes pull request is accepted.

The source hash can be reproduced with:

```bash
python -m pip download --no-deps --no-binary :all: pyaesa==1.2.1
python - <<'PY'
from pathlib import Path
import hashlib

path = next(Path(".").glob("pyaesa-1.2.1.tar.gz"))
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
```
