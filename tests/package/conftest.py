"""Shared fixtures for package test suite."""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
import shutil

import matplotlib
import pytest

matplotlib.use("Agg")

if TYPE_CHECKING:
    from tests.package.helpers.allocation_dummy_repo import AllocationDummyRepo
    from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo
    from tests.package.helpers.io_lca_dummy_repo import IOLCADummyRepo


class _DeepCopyable(Protocol):
    def copy(self, deep: bool = True) -> object: ...


@pytest.fixture
def tmp_repo_root(tmp_path: Path) -> Path:
    """Return a deterministic temporary repo root path for path based tests."""
    root = tmp_path / "pyaesa_repo"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def project_repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return one session-scoped workspace repository template for fast cloning."""
    from pyaesa import set_workspace
    from pyaesa.workspace_initialisation.workspace import (
        clear_default_repo_root,
        get_default_repo_root,
    )

    clear_default_repo_root()
    set_workspace(tmp_path_factory.mktemp("project_repo_template"), refresh=True)
    repo_root = get_default_repo_root()
    clear_default_repo_root()
    return repo_root


@pytest.fixture
def project_repo(project_repo_template: Path, tmp_path: Path) -> Generator[Path, None, None]:
    """Create one real temporary workspace repository and return its root path."""
    from pyaesa.workspace_initialisation.workspace import (
        clear_default_repo_root,
        set_default_repo_root,
    )

    repo_root = tmp_path / "pyaesa"
    if repo_root.exists():
        shutil.rmtree(repo_root)
    shutil.copytree(project_repo_template, repo_root)
    set_default_repo_root(repo_root)
    yield repo_root
    clear_default_repo_root()


@pytest.fixture
def io_lca_dummy_repo_factory(
    tmp_path: Path,
) -> Callable[..., "IOLCADummyRepo"]:
    """Return a factory that builds reusable IO-LCA dummy repositories."""
    from tests.package.helpers.io_lca_dummy_repo import build_io_lca_dummy_repo

    def factory(*, name: str = "io_lca_dummy", **kwargs) -> "IOLCADummyRepo":
        return build_io_lca_dummy_repo(tmp_path / name, **kwargs)

    return factory


@pytest.fixture
def io_lca_dummy_repo(io_lca_dummy_repo_factory) -> "IOLCADummyRepo":
    """Return one shared minimal processed MRIO repository for IO-LCA tests."""
    return io_lca_dummy_repo_factory()


@pytest.fixture
def allocation_dummy_repo_factory(
    allocation_dummy_repo_template: "AllocationDummyRepo",
    tmp_path: Path,
) -> Callable[..., "AllocationDummyRepo"]:
    """Return a factory that builds reusable allocation methods repositories."""
    from tests.package.helpers.allocation_dummy_repo import clone_allocation_dummy_repo

    def factory(*, name: str = "allocation_dummy") -> "AllocationDummyRepo":
        clone_top_path = tmp_path / name
        clone_repo_root = clone_top_path / "pyaesa"
        if clone_repo_root.exists():
            shutil.rmtree(clone_repo_root)
        return clone_allocation_dummy_repo(
            allocation_dummy_repo_template,
            top_path=clone_top_path,
        )

    return factory


@pytest.fixture
def allocation_dummy_repo(allocation_dummy_repo_factory) -> "AllocationDummyRepo":
    """Return one shared minimal repository for asocc tests."""
    return allocation_dummy_repo_factory()


@pytest.fixture(scope="session")
def allocation_dummy_repo_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> "AllocationDummyRepo":
    """Return one session-scoped allocation dummy template for fast cloning."""
    from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
    from tests.package.helpers.allocation_dummy_repo import build_allocation_dummy_repo

    template = build_allocation_dummy_repo(tmp_path_factory.mktemp("allocation_dummy_template"))
    clear_default_repo_root()
    return template


@pytest.fixture
def ar6_dummy_repo(
    ar6_dummy_repo_template: "AR6DummyRepo",
    tmp_path: Path,
) -> Generator["AR6DummyRepo", None, None]:
    """Return one shared deterministic AR6 raw data repository scaffold."""
    from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
    from tests.package.helpers.ar6_dummy_repo import clone_ar6_dummy_repo

    yield clone_ar6_dummy_repo(ar6_dummy_repo_template, top_path=tmp_path / "ar6_dummy")
    clear_default_repo_root()


@pytest.fixture(scope="session")
def ar6_dummy_repo_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> "AR6DummyRepo":
    """Return one session-scoped AR6 raw data template for fast cloning."""
    from pyaesa import set_workspace
    from pyaesa.workspace_initialisation.workspace import (
        clear_default_repo_root,
        get_default_repo_root,
    )
    from tests.package.helpers.ar6_dummy_repo import build_ar6_dummy_repo

    set_workspace(tmp_path_factory.mktemp("ar6_dummy_template"), refresh=True)
    template = build_ar6_dummy_repo(get_default_repo_root())
    clear_default_repo_root()
    return template


@pytest.fixture(scope="session")
def ar6_processed_pathway_outputs_template(
    ar6_dummy_repo_template: "AR6DummyRepo",
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    """Return one processed AR6 pathway payload shared by figure contract tests."""
    from pyaesa.download.ar6.utils.config import DEFAULT_VARIABLES_OUTPUT
    from pyaesa.process.ar6.utils.pipeline.loaders import scenario_metadata_from_wide
    from pyaesa.process.ar6.utils.pipeline.processing_modes import build_pathway_outputs
    from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
    from tests.package.helpers.ar6_dummy_repo import clone_ar6_dummy_repo
    from tests.package.helpers.ar6_imports import collection_explorer

    ar6_repo = clone_ar6_dummy_repo(
        ar6_dummy_repo_template,
        top_path=tmp_path_factory.mktemp("ar6_pathway_outputs"),
    )
    explorer = collection_explorer.read_explorer_csv(ar6_repo.explorer_csv_path)
    pathway_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=["C1", "C2", "C3", "C4"],
        ssps=[1, 2, 3, 4, 5],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        study_period=[2019, 2060],
        database_raw_dir=ar6_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=True,
        harmonization_method="reduced_offset",
    )
    source_meta = scenario_metadata_from_wide(explorer.data)
    clear_default_repo_root()
    return {
        "harmonized_data": pathway_outputs["final_all"],
        "original_data": pathway_outputs["original_all"],
        "harmonization_log": pathway_outputs["harmonization_log_all"],
        "historical_data": pathway_outputs["historical_emissions"],
        "source_metadata": source_meta,
    }


@pytest.fixture
def ar6_processed_pathway_outputs(
    ar6_processed_pathway_outputs_template: dict[str, object],
) -> dict[str, object]:
    """Return per-test copies of the shared processed AR6 pathway payload."""
    return {
        key: cast(_DeepCopyable, value).copy(deep=True) if hasattr(value, "copy") else value
        for key, value in ar6_processed_pathway_outputs_template.items()
    }


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Generator[None, None, None]:
    """Close all matplotlib figures after each package test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
