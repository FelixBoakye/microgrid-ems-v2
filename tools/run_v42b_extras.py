"""V4.3 extras on top of v42b (180ceeb): flat-tie numbers, Table XV timing, paper figures.

Writes ONLY to a new folder results/v2/v42b_extras_<date>[_rN]; never into an existing one.
Reads results/v2/v42b_ucparity_2026-10-23 (read-only). Every file in the older result
folders and in that v42b folder is hashed at the start and re-checked at the end.
Offline: no network access is needed or attempted.

Stages
  a  flat_tie      full-year LP vs flat rule, 2025, load_v1 and load_v2   (~1 min)
  b  timing        scripts_v2.timing (Table XV)                           (~10-20 min)
  b2 replan_times  every re-plan of the 2025 rolling MILP timed, 24 h    (~10-15 min)
  c  figures       paper Figs 1b, 2-6 from the v42b folder                (~1 min)

    .venv\\Scripts\\python.exe tools\\run_v42b_extras.py           real run (idle PC)
    python tools/run_v42b_extras.py --smoke                        quick smoke test
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
BASE = ROOT / "results" / "v2"
V42B = Path(os.environ.get("MG_V42B_SRC") or (BASE / "v42b_ucparity_2026-10-23"))
GUARDED = [BASE / "full_rerun_2026-10-16", BASE / "v42_rerun_2026-10-18c",
           BASE / "v42_rerun_2026-10-21_sc3375", V42B]
EXPECTED_BASE = "180ceeb"
PY = Path(sys.executable)

stem = "v42b_extras_" + time.strftime("%Y-%m-%d") + ("_SMOKE" if SMOKE else "")
OUT, _n = BASE / stem, 1
while OUT.exists():
    _n += 1
    OUT = BASE / f"{stem}_r{_n}"
LOGS = OUT / "logs"
STATUS = LOGS / "status.txt"

ENV = {k: v for k, v in os.environ.items() if not k.startswith("MG_")}
ENV.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", MG_RESULTS_ROOT=str(OUT),
           MG_START_COST_GHS="33.75", MG_START_COST_STRICT="1", MG_UC_PARITY="1",
           MG_LOAD="v2", HF_HUB_OFFLINE="1", NO_PROXY="*")
if SMOKE:
    ENV["MG_SMOKE_HOURS"] = "504"


def status(line):
    LOGS.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    print(line, flush=True)


def die(msg):
    status("STAGE_FAIL " + msg)
    raise SystemExit(1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def snapshot():
    rows = []
    for b in GUARDED:
        if b.exists():
            for p in sorted(b.rglob("*")):
                if p.is_file():
                    rows.append(f"{sha256(p)}  {p.as_posix()}")
    return rows


def run(name, args, env, est):
    t0 = time.time()
    status(f"START {name} (estimate {est})")
    log = LOGS / f"{name}.log"
    with log.open("w", encoding="utf-8") as f:
        f.write("CMD " + " ".join(map(str, args)) + "\n"); f.flush()
        rc = subprocess.Popen([str(a) for a in args], cwd=str(ROOT), env=env,
                              stdout=f, stderr=subprocess.STDOUT).wait()
        f.write(f"\nEXIT={rc}\n")
    if rc != 0:
        die(f"{name} exit {rc} (see logs/{name}.log)")
    status(f"STAGE_OK {name} {(time.time() - t0) / 60:.1f}min")


def preflight():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    if not SMOKE and subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD"],
                                    cwd=str(ROOT)).returncode != 0:
        die(f"HEAD {head} does not contain {EXPECTED_BASE}")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "scripts_v2",
                                     "tools", "reproduce.py"], cwd=str(ROOT), text=True).strip()
    if dirty:
        die("code not committed: " + dirty.replace("\n", "; "))
    st = V42B / "logs" / "status.txt"
    if not SMOKE and not (st.exists() and "ALL_DONE" in st.read_text(encoding="utf-8", errors="replace")):
        die(f"{V42B} is not ALL_DONE")
    for rel in ("replan_sweep_corrected.csv", "phase5_ladder_lv2_sourced_mean.csv", "week_trace.csv",
                "sensitivity/sizing_sweep.csv", "phase5_sourced_mean/oracle_dispatch.npz",
                "direct/direct_pred_2025_seed42.npz"):
        if not (V42B / rel).exists():
            die(f"missing input {V42B / rel}")
    for name in ("pv_output.csv", "synthetic_load.csv", "synthetic_load_lv2.csv",
                 "baseline_simulation.csv", "v2_forecasts.csv"):
        if not (ROOT / "data" / name).exists():
            die(f"missing data/{name}")
    OUT.mkdir(parents=True)
    (OUT / "guard_hashes_start.txt").write_text("\n".join(snapshot()) + "\n", encoding="utf-8")
    src = V42B / "direct" / "direct_pred_2025_seed42.npz"
    dst = OUT / "direct" / src.name
    dst.parent.mkdir(parents=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        die("direct cache copy mismatch")
    (OUT / "RUN_INFO.json").write_text(json.dumps({
        "git_commit": head, "out": str(OUT), "v42b_source": str(V42B), "smoke": SMOKE,
        "start_cost_ghs": 33.75, "uc_parity": True, "python": sys.version.split()[0],
        "direct_cache": {"copied_from": str(src), "sha256": sha256(dst)}}, indent=2), encoding="utf-8")
    status(f"STAGE_OK preflight HEAD {head[:7]} out {OUT.name}")


def finish():
    before = (OUT / "guard_hashes_start.txt").read_text(encoding="utf-8")
    if "\n".join(snapshot()) + "\n" != before:
        die("an older result folder changed during the run")
    rows = [f"{sha256(p)}  {p.relative_to(OUT).as_posix()}" for p in sorted(OUT.rglob("*"))
            if p.is_file() and p.name not in ("HASHES.txt", "status.txt")]
    (OUT / "HASHES.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    status(f"ALL_DONE guards unchanged ({len(before.splitlines())} files)")


def main():
    t0 = time.time()
    preflight()
    run("a_flat_tie_v1", [PY, ROOT / "tools" / "v42b_flat_tie.py"], dict(ENV, MG_LOAD="v1"), "<1 min")
    run("a_flat_tie_v2", [PY, ROOT / "tools" / "v42b_flat_tie.py"], ENV, "<1 min")
    tim = ["--ga-windows", "2", "--lp-windows", "20", "--milp-windows", "20"] if SMOKE else \
          ["--ga-windows", "5", "--lp-windows", "200", "--milp-windows", "300"]
    run("b_timing", [PY, "-m", "scripts_v2.timing", *tim], ENV, "10-20 min")
    run("b2_replan_times_h24", [PY, ROOT / "tools" / "v42b_replan_times.py", "--horizon", "24"],
        ENV, "10-15 min")
    run("c_figures", [PY, ROOT / "tools" / "v42b_extras_figs.py"], dict(ENV, MG_FIG_SRC=str(V42B)), "<1 min")
    finish()
    status(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
