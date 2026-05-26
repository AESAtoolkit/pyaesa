# Contributor Guide

Thank you for your interest in improving <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>.

<span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> is an open-source Python package for absolute
environmental sustainability assessment (AESA) studies. The project welcomes contributions in
the form of bug reports, feature requests, documentation improvements, tests,
and pull requests.

This project is distributed under the [GPL 3.0 license](https://github.com/AESAtoolkit/pyaesa/blob/main/LICENSE).
By contributing, you agree that your contribution is provided under the same
license.

## Code Authors

- Erwan Ike de Bantel, erwan.de-bantel@centralesupelec.fr
- Thibault Pirson, thibault.pirson@uclouvain.be
- Gonzalo Puig-Samper, gonzalo.puig-samper@list.lu
- Jan Marcus Hartmann, jan.hartmann@ltt.rwth-aachen.de

## License

Contributions to <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>
source code are accepted under the GPL 3.0 license.

Contributors must not add third party data, code, or documentation with licenses that are incompatible with GPL 3.0 distribution.

New data source integrations must document the terms and conditions of the
given datasets in `README.md`.

## Useful Resources

- [Source Code][Source Code]
- [Documentation Source](index.rst)
- [API Reference Source](api.rst)
- [Architecture Notes](architecture.rst)
- [Allocation Methods Extension Checklist](ADDING_METHODS_checklist.md)
- [Issue Tracker][Issue Tracker]
- [Pull Requests][Pull Requests]
- [Code of Conduct](https://github.com/AESAtoolkit/pyaesa/blob/main/CODE_OF_CONDUCT.md)

## How To Report A Bug

Report bugs on the [Issue Tracker][Issue Tracker].

When filing an issue, include:

- operating system
- Python version
- <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span> version or commit
- public function call or notebook step involved
- steps to reproduce the issue
- expected behavior
- observed behavior, including the full traceback when available

The best way to get a bug fixed is to provide a small reproducible example or
a focused test case.

## How To Request A Feature

Request features on the [Issue Tracker][Issue Tracker].

For AESA methodological features (e.g.: adding allocation methods, allocation levels, etc.), include the relevant equations,
data sources, references, expected outputs, and the workflow where the feature
would be used.

## How To Set Up A Development Environment

Supported Python versions are `>=3.10,<3.15`.

Install the package with test, quality, and documentation requirements:

```console
python -m pip install --upgrade pip
python -m pip install -e ".[test,quality,docs]"
```

The `test`, `quality`, and `docs` extras install the tools used for pytest,
Ruff, Pyright, and Sphinx.

Use the same Python environment for editable installation, tests, and
notebooks. If a notebook imports an older checkout, reinstall editable mode in
that notebook kernel and restart the kernel.

## How To Build The Documentation Locally

After installing the development requirements above, build the documentation
from the repository root:

```console
sphinx-build docs docs/_build
```

## How To Test The Project

The package test suite is located in `tests/package/` and uses `pytest`.

Run the package test gate:

```console
python -m pytest tests/package --cov=pyaesa --cov-branch --cov-report=term
```

The package coverage target is 100 percent line coverage and 100 percent branch
coverage for <span style="color:#366e9c"><strong>py</strong></span><span style="color:#c83737"><strong>aesa</strong></span>.

For focused development, run the relevant test file or folder first:

```console
python -m pytest tests/package/<area>
python -m pytest tests/package/<area>/test_file.py
```

The methodology validation suite under `tests/allocation_equation_validation/`
is a separate maintainer validation lane for allocation method additions. It
checks that methods respect the AESA additivity requirements for allocated
shares, and is not part of the normal package pytest gate.

## Code Quality Checks

Run the quality checks before submitting a pull request:

```console
python -m ruff check pyaesa tests/package
python -m ruff format --check pyaesa tests/package
python -m pyright pyaesa
```

Ruff is the formatting and linting baseline. Pyright is the static type checker
for source code.

## Contribution Guidelines

Follow these expectations when preparing a contribution:

- keep changes focused and reviewable
- keep scientific logic explicit
- add or update tests for changed behavior
- update public docstrings and `docs/api.rst` when public usage changes
- update tutorials or documentation when user workflows change
- follow the architecture notes for the package area being changed
- use the method extension checklist when adding or changing allocation methods
- do not include generated caches, local outputs, downloaded data, or notebook
  execution noise

## How To Submit Changes

Open a [pull request][Pull Requests] to submit changes.

For large changes, AESA methodological changes, or changes that affect public
outputs, open an issue first so the approach can be discussed before
implementation.

Your pull request should include:

- a short summary of the change
- the reason for the change
- tests and checks run
- documentation updates, when relevant
- any scientific assumptions or limitations, when relevant

Before requesting review, confirm that:

- tests pass for the changed area
- the package gate passes when package behavior changes
- Ruff check, Ruff format check, and Pyright pass
- public API documentation is synchronized with the code

[Source Code]: https://github.com/Ike-EDB/pyaesa
[Issue Tracker]: https://github.com/Ike-EDB/pyaesa/issues
[Pull Requests]: https://github.com/Ike-EDB/pyaesa/pulls
