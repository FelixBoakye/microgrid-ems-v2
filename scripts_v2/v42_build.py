"""Resumable driver for the V4.2 side stages (comparators, ablations, quiet hours).

Every sub-job (job x year x start cost) runs as its own process. PROGRESS.json
in --out records it as started, then done with output hashes. On restart, done
sub-jobs whose outputs still match are skipped; started-but-not-done ones are
redone from scratch (their outputs are written to *.tmp and renamed only on
completion, so a partial file is never taken as finished).

  python -m scripts_v2.v42_build --out DIR --stage v42    [--years 2022 2023 ...]
  python -m scripts_v2.v42_build --out DIR --stage quiet  (33.75 GHS only)
  --parallel N  runs up to N independent sub-jobs at once (default 4)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .progress import Ledger

OUTPUTS = {"rules": ["baseline", "heuristic", "heuristic_v40grid", "merit_constrained",
                     "load_following", "cycle_charging"],
           "q_rules": ["q_heuristic"]}
# Waves: a job may depend on a job of an earlier wave in the same year/cost.
V42_WAVES = [["rules", "oracle", "roll24", "abl_cs0", "abl_nobank"],
             ["abl_heurz", "abl_oraclez"]]
QUIET_WAVES = [["q_rules", "q_oracle", "q_roll24"]]


def ydir(out, year, cs):
    return Path(out) / f"{year}_cs{cs:g}".replace(".", "p")


def outputs(out, job, year, cs):
    d = ydir(out, year, cs)
    names = OUTPUTS.get(job, [job])
    return [d / f"{n}{ext}" for n in names for ext in (".npz", ".json")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("v42", "quiet"), required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--costs", type=float, nargs="+", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--jobs", nargs="+", default=None, help="restrict to these jobs")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    led = Ledger(out)
    waves = V42_WAVES if a.stage == "v42" else QUIET_WAVES
    costs = a.costs or ([33.75, 50.625] if a.stage == "v42" else [33.75])
    ok = True
    for wave in waves:
        jobs = [(j, y, c) for y in a.years for c in costs for j in wave
                if a.jobs is None or j in a.jobs]
        # Ledger writes happen on the main thread's object; serialise them.
        import threading
        lock = threading.Lock()

        def task(t):
            j, y, c = t
            with lock:
                done = led.is_done(f"{j}/{y}/cs{c:g}")
            if done:
                print(f"skip  {j}/{y}/cs{c:g} (done, hashes match)", flush=True)
                return True
            with lock:
                led.started(f"{j}/{y}/cs{c:g}")
            log = out / "logs" / f"{j}_{y}_cs{c:g}.log"
            with open(log, "w") as f:
                rc = subprocess.call([sys.executable, "-m", "scripts_v2.v42_runs",
                                      "--job", j, "--year", str(y), "--start-cost", str(c),
                                      "--out", str(out)], stdout=f, stderr=subprocess.STDOUT)
            outs = outputs(out, j, y, c)
            if rc == 0 and all(p.exists() for p in outs):
                with lock:
                    led.done(f"{j}/{y}/cs{c:g}", outs)
                print(f"done  {j}/{y}/cs{c:g}", flush=True)
                return True
            print(f"FAIL  {j}/{y}/cs{c:g} rc={rc} (see {log})", flush=True)
            return False

        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            ok &= all(ex.map(task, jobs))
    for y in a.years:
        for c in costs:
            subprocess.call([sys.executable, "-m", "scripts_v2.v42_runs", "--job", "collect",
                             "--year", str(y), "--start-cost", str(c), "--out", str(out)],
                            stdout=subprocess.DEVNULL)
    print("ALL DONE" if ok else "SOME SUB-JOBS FAILED (re-run the same command to retry)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
