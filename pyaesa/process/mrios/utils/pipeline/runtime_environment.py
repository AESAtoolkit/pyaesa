"""Runtime environment metadata for processed MRIO outputs."""

import platform
from importlib import metadata as importlib_metadata
from typing import Sequence


def runtime_env_versions(
    *,
    package_names: Sequence[str] = ("numpy", "pandas", "scipy", "pymrio", "pyarrow"),
    version_resolver=importlib_metadata.version,
) -> dict[str, str]:
    """Return runtime environment versions stored with processed outputs."""
    env: dict[str, str] = {"python": platform.python_version()}
    for package_name in package_names:
        try:
            env[package_name] = version_resolver(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return env
