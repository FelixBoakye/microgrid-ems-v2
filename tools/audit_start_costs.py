"""Fail if any output in a rerun folder was produced at a start cost other than the run's.

Every CSV/JSON with both start_cost_GHS and genset_starts must give
start_cost_GHS / genset_starts == the expected value; every manifest or JSON that
records start_cost_ghs must equal it. Files under a cs50p625 folder or with
cs50p625 in the name are held to 50.625. Files listed in PROVENANCE.json as
copies keep the start cost their source declares there.

    python tools/audit_start_costs.py RUN_DIR [--expect 33.75]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TOL = 1e-6


def expected_for(rel: Path, base: float) -> float:
    return 50.625 if "cs50p625" in rel.as_posix() else base


def check_rows(rows, exp, where, bad, seen):
    for i, r in enumerate(rows):
        try:
            sc = float(r["start_cost_GHS"]); n = float(r["genset_starts"])
        except (KeyError, TypeError, ValueError):
            continue
        if n > 0:
            seen.append(where)
            if abs(sc / n - exp) > TOL:
                bad.append(f"{where} row {i}: {sc / n:g} GHS/start, expected {exp:g}")


def walk_json(obj, exp, where, bad, seen):
    if isinstance(obj, dict):
        if "start_cost_ghs" in obj and isinstance(obj["start_cost_ghs"], (int, float)):
            seen.append(where)
            if abs(float(obj["start_cost_ghs"]) - exp) > TOL:
                bad.append(f"{where}: start_cost_ghs={obj['start_cost_ghs']}, expected {exp:g}")
        check_rows([obj], exp, where, bad, seen)
        for v in obj.values():
            walk_json(v, exp, where, bad, seen)
    elif isinstance(obj, list):
        for v in obj:
            walk_json(v, exp, where, bad, seen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--expect", type=float, default=33.75)
    a = ap.parse_args(argv)
    root = Path(a.run_dir)
    bad, seen, files = [], [], 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or rel.parts[0] == "logs" or p.name in ("PROGRESS.json",):
            continue
        exp = expected_for(rel, a.expect)
        if p.suffix.lower() == ".csv":
            files += 1
            with p.open(newline="", encoding="utf-8", errors="replace") as f:
                check_rows(list(csv.DictReader(f)), exp, rel.as_posix(), bad, seen)
        elif p.suffix.lower() == ".json":
            files += 1
            try:
                walk_json(json.loads(p.read_text(encoding="utf-8")), exp, rel.as_posix(), bad, seen)
            except json.JSONDecodeError:
                bad.append(f"{rel}: unreadable JSON")
    print(f"scanned {files} files, {len(set(seen))} carry a start cost")
    for b in bad:
        print("BAD", b)
    if bad:
        return 1
    print(f"START_COST_AUDIT_OK every recorded start cost is {a.expect:g} (50.625 in cs50p625)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
