"""
Faithful reproduction harness for the original delivered pipeline.

The original scripts in scripts/ are NEVER edited. This harness loads them by
path and calls their main(), with cwd set to the repository root so their
relative "data/..." paths resolve.

Two shipped defects prevent the pipeline from running as delivered; both are
worked around here in the *harness*, not in the scripts:

  (a) 10_mpc_evaluation.py does `from scripts.ga_optimizer import ...`, but no
      such module exists (the file is 09_ga_optimizer.py, and "09_..." is not a
      valid Python identifier). We register the module in sys.modules.
  (b) Filenames begin with digits, so they cannot be imported normally.

Usage:  python repro/run_original.py [step ...]
        python repro/run_original.py 01 02 03 04 05 08
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

STEPS = {
    "01": "01_pv_model.py",
    "02": "02_load_profile_generator.py",
    "03": "03_battery_baseline_ems.py",
    "04": "04_build_ml_dataset.py",
    "05": "05_train_val_test_split.py",
    "06": "06_rf_forecast.py",
    "07": "07_lstm_forecast.py",
    "08": "08_xgb_forecast.py",
    "10": "10_mpc_evaluation.py",
    "11": "11_final_results.py",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_ga_shim():
    """Make `from scripts.ga_optimizer import ...` resolve to 09_ga_optimizer.py."""
    ga = load_module(SCRIPTS / "09_ga_optimizer.py", "ga_optimizer")
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [str(SCRIPTS)]
    sys.modules["scripts"] = pkg
    sys.modules["scripts.ga_optimizer"] = ga
    pkg.ga_optimizer = ga
    return ga


def env_manifest() -> dict:
    import numpy, pandas, scipy, sklearn
    man = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import xgboost
        man["xgboost"] = xgboost.__version__
    except Exception:
        man["xgboost"] = None
    try:
        man["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        man["git_commit"] = None
    return man


def main(argv):
    os.chdir(ROOT)
    steps = argv or ["01", "02", "03", "04", "05", "08", "10", "11"]
    install_ga_shim()

    manifest = env_manifest()
    manifest["steps"] = steps
    manifest["durations_s"] = {}

    for s in steps:
        fname = STEPS[s]
        print(f"\n{'=' * 70}\n  STEP {s}: {fname}\n{'=' * 70}", flush=True)
        t0 = time.time()
        mod = load_module(SCRIPTS / fname, f"orig_{s}")
        mod.main()
        dt = time.time() - t0
        manifest["durations_s"][s] = round(dt, 2)
        print(f"  [step {s} done in {dt:.1f}s]", flush=True)

    out = ROOT / "manifests" / f"repro_original_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
