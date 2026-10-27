"""V4.2b: full re-run with unit-commitment parity (MG_UC_PARITY=1) into a NEW folder.

Baseline and heuristic now obey the MILP's genset rules (min stable load, min up/down,
surplus -> battery then curtail; parameters from GensetModel). Every number that
contains a baseline or heuristic row is re-run: phase-5 ladders (+50.625 check),
walk-forward at 33.75 and 50.625 (dispatch, collect, seasonal, bootstrap), all window
sensitivities, sizing, seeds, linear, re-planning, tables, week trace, figures.
Only the direct-forecast caches (no start cost, no dispatch) are copied in.
Resume-safe: re-running the same command continues from PROGRESS.json.

    .venv\\Scripts\\python.exe tools\\run_v42b_ucparity.py        real run
    python tools/run_v42b_ucparity.py --smoke                     quick smoke test
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
SMOKE_HOURS = "504"
SRC21 = ROOT / "results" / "v2" / "v42_rerun_2026-10-21_sc3375"


def _pick_out():
    """results/v2/v42b_ucparity_<date>[_SMOKE]. Resume an unfinished folder of that
    name (or the newest unfinished _rN); never write into a finished one."""
    base = ROOT / "results" / "v2"
    env_out = os.environ.get("MG_V42B_OUT", "").strip()
    if env_out:
        return base / env_out
    stem = "v42b_ucparity_" + time.strftime("%Y-%m-%d") + ("_SMOKE" if SMOKE else "")
    # an unfinished v42b folder from an earlier day is resumed first
    for p in sorted(base.glob("v42b_ucparity_*"), reverse=True):
        if (("_SMOKE" in p.name) == SMOKE and (p / "PROGRESS.json").exists()
                and not _finished(p)):
            return p
    cand, n = base / stem, 1
    while cand.exists() and any(cand.iterdir()):
        n += 1
        cand = base / f"{stem}_r{n}"
    return cand


def _finished(p):
    st = p / "logs" / "status.txt"
    return st.exists() and "ALL_DONE" in st.read_text(encoding="utf-8", errors="replace")


OUT = _pick_out()
SRC18 = ROOT / "results" / "v2" / "v42_rerun_2026-10-18c"
OLD16 = ROOT / "results" / "v2" / "full_rerun_2026-10-16"
LOGS = OUT / "logs"
STATUS = LOGS / "status.txt"
PY = Path(sys.executable)
EXPECTED_BASE = "0ce993a"
START = "33.75"

sys.path.insert(0, str(ROOT))

ENV = os.environ.copy()
if not SMOKE and ENV.get("MG_SMOKE_HOURS", "").strip():
    raise SystemExit("MG_SMOKE_HOURS is set; a real run must not be truncated")
ENV.update(MG_LOAD="v2", PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
           MG_RESULTS_ROOT=str(OUT), MG_START_COST_GHS=START, MG_START_COST_STRICT="1", MG_UC_PARITY="1")
if SMOKE:
    ENV["MG_SMOKE_HOURS"] = SMOKE_HOURS

# Test env: no rerun root, no start-cost override, so tests see the code as committed.
TEST_ENV = {k: v for k, v in ENV.items()
            if k not in ("MG_RESULTS_ROOT", "MG_START_COST_GHS", "MG_START_COST_STRICT",
                         "MG_SMOKE_HOURS", "MG_UC_PARITY")}
ENV50 = dict(ENV, MG_START_COST_GHS="50.625")

# Walk-forward first: bootstrap needs the folds, tables/week_trace need fold_2025.
WF = [("walkforward", ENV), ("walkforward_collect", ENV), ("seasonal", ENV),
      ("walkforward_50", ENV50), ("walkforward_50_collect", ENV50), ("seasonal_50", ENV50),
      ("bootstrap", ENV), ("bootstrap_50", ENV50)]
# Order matters: tables need phase5 (with roll24p) and mix12; linear needs phase5.
SENS = ["replan", "replan_corrected",
        "phase5", "phase5_generac", "phase5_cummins", "phase5_kohler",
        "phase5_mix12", "phase5_loadv1",
        "sens_linear", "sens_window", "sens_genset", "sens_demand", "sizing", "seeds",
        "tables", "week_trace"]
# Rough 18c-based minutes for each stage, padded (status lines and the ETA file use them).
PADDED_MIN = {"replan": 10, "replan_corrected": 120, "phase5": 60, "phase5_generac": 45,
              "phase5_cummins": 45, "phase5_kohler": 45, "phase5_mix12": 40,
              "phase5_loadv1": 40, "sens_linear": 5, "sens_window": 20, "sens_genset": 150,
              "sens_demand": 100, "sizing": 130, "seeds": 110, "tables": 5, "week_trace": 5,
              "walkforward": 270, "walkforward_collect": 2, "seasonal": 2,
              "walkforward_50": 270, "walkforward_50_collect": 2, "seasonal_50": 2,
              "bootstrap": 5, "bootstrap_50": 5, "phase5_cs50p625": 60, "figures": 10}

# Read-only inputs: direct-forecast caches only (model outputs, no start cost, no
# dispatch). ensure_direct retrains if a cache does not match. No folds are copied.
COPIES = {}
for _y in (2022, 2023, 2024, 2025):
    COPIES[f"direct/direct_pred_{_y}_seed42.npz"] = f"direct/direct_pred_{_y}_seed42.npz"
COPIES["mix12/direct/direct_pred_2025_seed42.npz"] = "mix12/direct/direct_pred_2025_seed42.npz"
COPIES["v1/direct/direct_pred_2025_seed42.npz"] = "v1/direct/direct_pred_2025_seed42.npz"
COPY_GLOBS = {"nominal/direct/demand": "sensitivity/direct",
              "nominal/direct/seeds": "sensitivity/direct"}


def status(line):
    LOGS.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line.rstrip() + "\n")
        f.flush(); os.fsync(f.fileno())
    print(line, flush=True)


def die(msg):
    status("STAGE_FAIL " + msg)
    raise SystemExit(1)


def ledger():
    from scripts_v2.progress import Ledger
    return Ledger(OUT)


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_logged(args, log: Path, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", buffering=1) as f:
        f.write("CMD " + " ".join(map(str, args)) + "\n")
        rc = subprocess.Popen([str(a) for a in args], cwd=str(ROOT), env=env,
                              stdout=f, stderr=subprocess.STDOUT).wait()
        f.write(f"\nEXIT={rc}\n")
    return rc


def run_retry(key, args, log: Path, env, tries=3):
    """A 10 s re-plan can stop at HiGHS's time limit with no feasible point.
    Retry the identical command, same settings, at most `tries` times."""
    for attempt in range(1, tries + 1):
        rc = run_logged(args, log, env)
        if rc == 0:
            return 0
        text = log.read_text(encoding="utf-8", errors="replace")
        if "Time limit reached" not in text or attempt == tries:
            return rc
        keep = log.with_name(f"{log.stem}.try{attempt}.log")
        shutil.copyfile(log, keep)
        status(f"RETRY {key} after solver time-limit stop (attempt {attempt + 1}/{tries}, same settings)")
    return rc


def snapshot_old():
    """Hashes of every file in the older result folders. Checked again at the end."""
    rows = []
    for base in (OLD16, SRC18, SRC21):
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rows.append(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}")
    return rows


def preflight():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    if not SMOKE and subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD"],
                                    cwd=str(ROOT)).returncode != 0:
        die(f"HEAD {head} does not contain {EXPECTED_BASE}")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "scripts_v2", "tests",
                                     "tools", "reproduce.py"], cwd=str(ROOT), text=True).strip()
    if dirty:
        die("code is not committed, so the manifests would not name it: " + dirty.replace("\n", "; "))
    from scripts_v2 import cost_model as cm
    if not hasattr(cm, "nominal_start_cost"):
        die("sc3375 patch is not applied (cost_model.nominal_start_cost missing)")
    import inspect
    from scripts_v2 import dispatch_core as dc
    if not hasattr(dc, "uc_parity_default") or "uc_parity" not in inspect.signature(dc.baseline).parameters:
        die("v42b UC-parity patch is not applied (dispatch_core.baseline has no uc_parity)")
    os.environ["MG_UC_PARITY"] = "1"
    if not dc.uc_parity_default():
        die("MG_UC_PARITY does not resolve to parity")
    for p in (SRC18 / "SECTION4.md", SRC18 / "nominal" / "phase5_lv2_ladder_2026-10-16.csv"):
        if not p.exists():
            die(f"missing 18c input {p}")
    data = ROOT / "data"
    for name in ("pv_output.csv", "synthetic_load_lv2.csv", "lv2_forecasts.csv"):
        if not (data / name).exists():
            die(f"missing data/{name}")
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < 5:
        die(f"only {free:.1f} GiB free")
    guard = OUT / "guard_hashes_start.txt"
    if not guard.exists():
        guard.write_text("\n".join(snapshot_old()) + "\n", encoding="utf-8")
    (OUT / "RUN_INFO.json").write_text(json.dumps({
        "git_commit": head, "start_cost_ghs": float(START),
        "check_start_cost_ghs": 50.625, "python": sys.version.split()[0],
        "out": str(OUT), "source_18c": str(SRC18), "smoke_hours": SMOKE_HOURS if SMOKE else None,
        "heuristic_rule": "year-1 (2024) selection, walk-forward THETAS x PSIS",
        "uc_parity": True, "uc_rules": "min stable load, min up/down from GensetModel; "
        "surplus -> battery then curtail"}, indent=2),
        encoding="utf-8")
    status(f"STAGE_OK preflight HEAD {head[:7]} free {free:.0f} GiB")


def gate():
    key = "gate"
    if ledger().is_done(key):
        return status("STAGE_OK gate skip")
    ledger().started(key)
    log = LOGS / "gate.log"
    rc = run_logged([PY, "-m", "pytest", "tests/test_no_lookahead.py", "tests/test_direct_forecast.py",
                     "tests/test_start_cost_env.py", "tests/test_uc_parity.py", "-q", "--tb=short"], log, TEST_ENV)
    if rc != 0:
        die("gate pytest failed")
    ledger().done(key, [log])
    status("STAGE_OK gate")


def copy_inputs():
    key = "copy_inputs"
    if ledger().is_done(key):
        return status("STAGE_OK copy_inputs skip")
    ledger().started(key)
    prov = {}
    pairs = dict(COPIES)
    for sdir, ddir in COPY_GLOBS.items():
        for f in sorted((SRC18 / sdir).glob("*.npz")):
            pairs[f"{sdir}/{f.name}"] = f"{ddir}/{f.name}"
    for src_rel, dst_rel in pairs.items():
        src, dst = SRC18 / src_rel, OUT / dst_rel
        if not src.exists():
            if "_seed42.npz" in src_rel and src_rel.startswith("direct/") and "2025" not in src_rel:
                status(f"NOTE {src_rel} not in 18c; ensure_direct will retrain it (adds time)")
                continue
            die(f"missing 18c input {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(src) != sha256(dst):
            die(f"copy mismatch {dst_rel}")
        prov[dst_rel] = {"copied_from": src.relative_to(ROOT).as_posix(), "sha256": sha256(dst),
                         "what": "direct-forecast cache (no start cost, no dispatch)"}
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    ledger().done(key, [OUT / d for d in prov] + [OUT / "PROVENANCE.json"])
    status(f"STAGE_OK copy_inputs {len(prov)} files")


def sens_stage(name, env=None):
    env = ENV if env is None else env
    key = f"sc/{name}"
    if ledger().is_done(key):
        return status(f"STAGE_OK {key} skip")
    ledger().started(key)
    t0 = time.time()
    status(f"START {key} (padded estimate {PADDED_MIN.get(name, 30)} min)")
    log = LOGS / f"s_{name}.log"
    rc = run_retry(key, [PY, str(ROOT / "reproduce.py"), "--stage", name], log, env)
    if rc != 0:
        die(f"{key} exit {rc}")
    ledger().done(key, [log])
    status(f"STAGE_OK {key} {(time.time() - t0) / 60:.1f}min")


def check_50():
    """Same 2025 ladder at 50.625, for the attribution split at the higher start cost."""
    key = "sc/phase5_cs50p625"
    env = ENV50
    if not ledger().is_done(key):
        ledger().started(key)
        t0 = time.time()
        rc = run_retry(key, [PY, "-m", "scripts_v2.run_phase5", "--curve", "sourced_mean",
                             "--start-cost", "50.625",
                             "--outdir", "v2/cs50p625/phase5_sourced_mean",
                             "--tag", "_lv2_sourced_mean_cs50p625",
                             "--config", "baseline,heuristic,oracle,roll24,roll24p", "--collect"],
                       LOGS / "s_phase5_cs50p625.log", env)
        if rc != 0:
            die(f"{key} exit {rc}")
        ledger().done(key, [OUT / "phase5_ladder_lv2_sourced_mean_cs50p625.csv"])
        status(f"STAGE_OK {key} {(time.time() - t0) / 60:.1f}min")
    key = "sc/attribution_cs50p625"
    if not ledger().is_done(key):
        ledger().started(key)
        rc = run_logged([PY, "-m", "scripts_v2.full_rerun_tables", "--attribution-only",
                         "phase5_ladder_lv2_sourced_mean_cs50p625.csv", "attribution_cs50p625.csv"],
                        LOGS / "s_attribution_cs50p625.log", env)
        if rc != 0:
            die(f"{key} exit {rc}")
        ledger().done(key, [OUT / "attribution_cs50p625.csv"])
        status(f"STAGE_OK {key}")


def figures():
    key = "sc/figures"
    if ledger().is_done(key):
        return status("STAGE_OK sc/figures skip")
    ledger().started(key)
    rc = run_logged([PY, "-m", "scripts_v2.make_figures"], LOGS / "s_figures.log", ENV)
    if rc != 0:
        die(f"figures exit {rc}")
    ledger().done(key, sorted((OUT / "figures").glob("*.png")))
    status("STAGE_OK sc/figures")


def audit_and_crosscheck():
    rc = run_logged([PY, str(ROOT / "tools" / "audit_start_costs.py"), str(OUT), "--expect", START],
                    LOGS / "s_audit_start_costs.log", TEST_ENV)
    if rc != 0:
        die("start-cost audit found an output not at 33.75/50.625 (see logs/s_audit_start_costs.log)")
    status("STAGE_OK start-cost audit")
    # Every heuristic in this folder must come from the year-1 walk-forward rule.
    import pandas as pd
    bad = []
    for p in list(OUT.rglob("heuristic.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if "walk-forward rule" not in str(j.get("heuristic_selected_on", "")):
            bad.append(p.relative_to(OUT).as_posix())
    for p in (OUT / "sensitivity").glob("*.csv"):
        if p.name == "linear_sensitivity.csv":
            continue
        d = pd.read_csv(p)
        col = d.get("heuristic_selected_on")
        if col is None or not col.astype(str).str.contains("walk-forward rule").all():
            bad.append(p.relative_to(OUT).as_posix())
    if bad:
        die("heuristic not selected by the walk-forward rule in: " + ", ".join(bad))
    status("STAGE_OK heuristic-rule audit")
    # Regression against the 33.75 single-year ladders the 18c nominal runner already made.
    # Report only: rule-based rows must agree exactly, MILP rows within solver noise.
    import pandas as pd
    lines = ["| ladder | config | this run GHS | 18c nominal GHS | diff % |", "|---|---|---:|---:|---:|"]
    pairs = [("phase5_ladder_lv2_sourced_mean.csv", "nominal/phase5_lv2_ladder_2026-10-16.csv", {}),
             ("mix12/phase5_ladder_mix12.csv", "nominal/mix12_ladder_2026-10-16.csv", {}),
             ("v1/phase5_ladder_loadv1.csv", "nominal/loadv1_ladder_2026-10-16.csv", {})]
    for curve, tag in (("generac_sd040", "generac"), ("cummins_c40d6", "cummins"), ("kohler_40reozk", "kohler")):
        pairs.append((f"phase5_ladder_lv2_{tag}.csv", f"nominal/fuel/{curve}/ladder.csv", {}))
    worst_rule = 0.0
    for new_rel, old_rel, alias in pairs:
        new_p, old_p = OUT / new_rel, SRC18 / old_rel
        if not (new_p.exists() and old_p.exists()):
            lines.append(f"| {new_rel} | missing | | | |")
            continue
        n = pd.read_csv(new_p).set_index("config")
        o = pd.read_csv(old_p).set_index("config")
        for cfg in n.index:
            oc = alias.get(cfg, cfg)
            if oc not in o.index:
                continue
            a, b = float(n.loc[cfg, "total_cost_GHS"]), float(o.loc[oc, "total_cost_GHS"])
            d = (a / b - 1) * 100
            if cfg == "heuristic":
                continue  # selected on a different grid in 18c; not comparable
            if cfg == "baseline":
                worst_rule = max(worst_rule, abs(d))
            lines.append(f"| {new_rel} | {cfg} | {a:,.2f} | {b:,.2f} | {d:+.4f} |")
    lines.append("")
    lines.append(f"Largest baseline difference: {worst_rule:.4f}%. EXPECTED to be non-zero: "
                 "this run's baseline obeys the UC rules, 18c's did not. MILP rows (oracle, roll24, "
                 "roll24p) should agree within solver noise. Heuristic rows not listed.")
    if SMOKE:
        lines.append("SMOKE RUN: series truncated, differences are meaningless.")
    (OUT / "CROSSCHECK_vs_18c_nominal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(f"STAGE_OK crosscheck written; baseline shift vs 18c up to {worst_rule:.3f}% (expected)")


def verify():
    key = "sc/verify"
    if ledger().is_done(key):
        return status("STAGE_OK sc/verify skip")
    ledger().started(key)
    rc = run_logged([PY, str(ROOT / "reproduce.py"), "--verify"], LOGS / "s_verify.log", TEST_ENV)
    if rc != 0:
        die("reproduce --verify failed")
    ledger().done(key, [LOGS / "s_verify.log"])
    status("STAGE_OK sc/verify")


def finish():
    before = (OUT / "guard_hashes_start.txt").read_text(encoding="utf-8").splitlines()
    now = snapshot_old()
    if before != now:
        changed = sorted(set(before) ^ set(now))[:20]
        die("an older result folder changed during the run: " + "; ".join(changed))
    rows = [f"{sha256(p)}  {p.relative_to(OUT).as_posix()}" for p in sorted(OUT.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".csv", ".json", ".npz", ".png")
            and p.relative_to(OUT).parts[0] != "logs"]
    (OUT / "HASHES.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    status(f"STAGE_OK hashes {len(rows)} files; older folders unchanged")


def write_eta():
    stages = [n for n, _ in WF] + SENS + ["phase5_cs50p625", "figures"]
    tot = sum(PADDED_MIN.get(n, 30) for n in stages)
    done = sum(PADDED_MIN.get(n, 30) for n in stages if ledger().is_done(f"sc/{n}"))
    left = tot - done
    eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + left * 60))
    lines = [f"Padded estimate {tot} min ({tot / 60:.1f} h) total, {left} min left; ETA {eta} (PC clock).",
             "Walk-forward 33.75 and 50.625 are ~4 h each; side studies ~6 h."]
    lines += [f"{n}: {PADDED_MIN.get(n, 30)} min" for n in stages]
    (OUT / "RUNTIME_ESTIMATE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(lines[0])


_LOCK = None


def hold_lock():
    """One driver per output folder. The OS lock is released automatically if the
    process dies, so there is never a stale lock to clean up."""
    global _LOCK
    f = open(LOGS / "run.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit("another run_v42b_ucparity driver is already running on this folder; not starting a second one")
    _LOCK = f


def main():
    if OUT.exists() and any(OUT.iterdir()) and not (OUT / "PROGRESS.json").exists():
        raise SystemExit(f"{OUT} has files but no PROGRESS.json; refusing to write into it")
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    hold_lock()
    (LOGS / "driver.pid").write_text(str(os.getpid()), encoding="ascii")
    try:
        preflight()
        gate()
        copy_inputs()
        write_eta()
        for name, env in WF:
            sens_stage(name, env)
        for name in SENS:
            sens_stage(name)
        check_50()
        figures()
        audit_and_crosscheck()
        verify()
        finish()
        status("ALL_DONE")
    except SystemExit:
        raise
    except Exception as exc:
        die(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
