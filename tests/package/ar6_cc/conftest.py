"""AR6 CC specific fixtures."""

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo


@pytest.fixture(scope="session")
def ar6_cc_dummy_repo_template(
    ar6_dummy_repo_template: "AR6DummyRepo",
    tmp_path_factory: pytest.TempPathFactory,
) -> "AR6DummyRepo":
    """Return one AR6 repository template with processed CC prerequisites."""
    from pyaesa import deterministic_ar6_cc
    from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
    from tests.package.helpers.ar6_dummy_repo import clone_ar6_dummy_repo

    template = clone_ar6_dummy_repo(
        ar6_dummy_repo_template,
        top_path=tmp_path_factory.mktemp("ar6_cc_dummy_template"),
    )
    deterministic_ar6_cc(
        years=range(2019, 2022),
        category=["C1"],
        ssp_scenario=["SSP1"],
        figures=False,
        refresh=False,
    )
    clear_default_repo_root()
    return template


@pytest.fixture
def ar6_dummy_repo(
    ar6_cc_dummy_repo_template: "AR6DummyRepo",
    tmp_path: Path,
) -> Generator["AR6DummyRepo", None, None]:
    """Return one AR6 repository clone with processed CC prerequisites."""
    from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
    from tests.package.helpers.ar6_dummy_repo import clone_ar6_dummy_repo

    yield clone_ar6_dummy_repo(ar6_cc_dummy_repo_template, top_path=tmp_path / "ar6_cc_dummy")
    clear_default_repo_root()
