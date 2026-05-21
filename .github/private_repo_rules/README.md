# Private Repository Rules

This folder provides local replacements for CI checks when the repository is
private.

All workflows in `.github/workflows/` are configured to run only when
`github.event.repository.private == false`.

CI workflows are PR oriented (with docs/markdown path ignores). This local
runner is intentionally broader and executes the core package and quality checks
on demand.

## First run (install dependencies)

If your environment is not ready yet, run:

```powershell
pwsh -File .github/private_repo_rules/run_private_checks.ps1 -Install
```

## Recommended command (usual PR checks except public specific publishing checks)

Use this command to run the same checks that are normally enforced by public repo
PR workflows (package tests + quality):

```powershell
pwsh -File .github/private_repo_rules/run_private_checks.ps1
```

This runs:

- `ruff check pyaesa tests/package`
- `ruff format --check pyaesa tests/package`
- `pyright pyaesa`
- `python -m pytest tests/package --cov=pyaesa --cov-branch --cov-report=term`
- `python -m build`
- `twine check --strict dist/*`



## Fast development mode (skip packaging checks)

Use this during iteration when you only want lint + tests:

```powershell
pwsh -File .github/private_repo_rules/run_private_checks.ps1 -SkipBuild
```

Skipped in this mode:

- `python -m build`
- `twine check --strict dist/*`
