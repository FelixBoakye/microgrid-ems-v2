"""Shared test fixtures.

The `ga` and `mpc` fixtures load an optional external GA baseline from
`scripts/` if present; tests that need it are skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SCRIPTS_V2 = ROOT / "scripts_v2"
sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    if not path.exists():
        pytest.skip(f"optional baseline not included: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def ga():
    """The original 09_ga_optimizer.py, unmodified."""
    return _load(SCRIPTS / "09_ga_optimizer.py", "orig_ga_optimizer")


@pytest.fixture(scope="session")
def mpc(ga):
    """The original 10_mpc_evaluation.py, unmodified.

    As delivered it does `from scripts.ga_optimizer import ...`, which does not
    resolve. We register that module name so the file can be imported for
    testing; the file itself is untouched.
    """
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [str(SCRIPTS)]
    sys.modules["scripts"] = pkg
    sys.modules["scripts.ga_optimizer"] = ga
    return _load(SCRIPTS / "10_mpc_evaluation.py", "orig_mpc_evaluation")
