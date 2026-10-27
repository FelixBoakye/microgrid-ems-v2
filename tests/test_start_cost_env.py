"""The start cost a run asks for is the start cost every default cost model uses."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _probe(env_extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MG_START_COST_GHS", "MG_START_COST_STRICT")}
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT)
    code = ("from scripts_v2 import cost_model as cm, heuristic as H;"
            "print(cm.CORRECTED.genset.start_cost_ghs, cm.corrected('generac_sd040').genset.start_cost_ghs,"
            " cm.corrected(start_cost_ghs=50.625).genset.start_cost_ghs)")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def test_default_unchanged_without_env():
    r = _probe({})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["150.0", "150.0", "50.625"]


def test_env_sets_every_default():
    r = _probe({"MG_START_COST_GHS": "33.75", "MG_START_COST_STRICT": "1"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["33.75", "33.75", "50.625"]


def test_strict_rejects_the_legacy_default():
    r = _probe({"MG_START_COST_STRICT": "1"})
    assert r.returncode != 0
    assert "MG_START_COST_GHS is unset" in r.stderr
