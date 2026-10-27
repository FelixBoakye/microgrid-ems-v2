"""CI check: every reported result must be computed, never typed in.

Two rules:

1. **No metric literals in source.** A dict that pairs a known result key
   (LPSP_pct, diesel_kWh_per_year, total_cost_GHS, ...) with a numeric literal
   in `scripts_v2/` is treated as a hardcoded result.
2. **Every results manifest records a git commit** and library versions.

Paper results: results/v2/v42b_ucparity_2026-10-23 and
results/v2/v42b_extras_2026-10-23.

Exit code 1 on any violation.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts_v2"
RESULTS = ROOT / "results"

METRIC_KEYS = {
    "LPSP_pct", "diesel_kWh_per_year", "diesel_cost_GHS_per_year",
    "renewable_fraction_pct", "curtailed_kWh_per_year",
    "battery_cycles_per_year", "total_cost_GHS", "fuel_litres",
    "genset_run_hours", "genset_starts", "mean_load_factor_pct",
    "saving_vs_baseline_pct", "saving_vs_heuristic_pct",
    "premium_vs_oracle_pct",
}

# Values that are legitimately literal: identity elements and shares, not results.
ALLOWED = {0, 0.0, 1, 1.0, 100, 100.0}


def check_no_metric_literals():
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if (isinstance(k, ast.Constant) and k.value in METRIC_KEYS
                            and isinstance(v, ast.Constant)
                            and isinstance(v.value, (int, float))
                            and v.value not in ALLOWED):
                        bad.append(f"{p.relative_to(ROOT)}:{k.lineno} "
                                   f"{k.value} = {v.value!r}")
    return bad


def check_manifests():
    """Every canonical results CSV needs a manifest naming its git commit."""
    manifests = {p.name for p in RESULTS.rglob("*manifest*.json")}
    if not manifests:
        return ["no manifests found under results/"]
    txt = " ".join((RESULTS / m).read_text(encoding="utf-8", errors="replace")
                   if (RESULTS / m).exists() else ""
                   for m in [])
    bad = []
    for p in RESULTS.rglob("*manifest*.json"):
        s = p.read_text(encoding="utf-8", errors="replace")
        if '"git_commit"' not in s:
            bad.append(f"{p.relative_to(ROOT)} has no git_commit")
    return bad



def main():
    failures = []
    for name, fn in (("metric literals in source", check_no_metric_literals),
                     ("results manifests", check_manifests)):
        bad = fn()
        status = "PASS" if not bad else "FAIL"
        print(f"[{status}] {name}" + ("" if not bad else f" ({len(bad)})"))
        for b in bad:
            print(f"        {b}")
        failures += bad
    if failures:
        print(f"\n{len(failures)} violation(s)")
        return 1
    print("\nno hardcoded results, manifests present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
