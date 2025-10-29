#!/usr/bin/env python3
"""Ensure the branch-local legacy Cython installation is present."""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
from distutils.version import LooseVersion


def ensure_legacy_cython(target_dir: pathlib.Path, version_pin: str) -> None:
    """Install the pinned legacy Cython into target_dir if required."""
    required_cap = LooseVersion("3.0")
    target_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(target_dir))

    needs_install = True
    installed_version = None

    try:
        import Cython  # type: ignore
    except ImportError:
        pass
    else:
        installed_version = Cython.__version__
        if LooseVersion(installed_version) < required_cap and LooseVersion(installed_version) == LooseVersion(
            version_pin
        ):
            needs_install = False

    if needs_install:
        print(f"Installing Cython=={version_pin} into {target_dir}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(target_dir),
                f"Cython=={version_pin}",
            ]
        )
        importlib.invalidate_caches()
        sys.path.insert(0, str(target_dir))
        sys.modules.pop("Cython", None)
        for name in [mod for mod in sys.modules if mod.startswith("Cython.")]:
            sys.modules.pop(name, None)
        import Cython  # type: ignore

        print(f"Installed Cython {Cython.__version__} in {target_dir}")
    else:
        print(f"Using legacy Cython {installed_version} from {target_dir}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: ensure_legacy_cython.py <target_dir> <version_pin>", file=sys.stderr)
        return 2

    target_dir = pathlib.Path(argv[1]).expanduser().resolve()
    version_pin = argv[2]
    ensure_legacy_cython(target_dir, version_pin)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
