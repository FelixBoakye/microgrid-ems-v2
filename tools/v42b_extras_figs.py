"""Paper figures (V4.3 numbering) from a finished v42b results folder.

Reads MG_FIG_SRC (read-only) and writes PNGs to MG_RESULTS_ROOT/figures with the
paper's numbering:
  fig1b  re-planning interval, full cost      (replan_sweep_corrected.csv)
  fig2   genset load factor, 2025             (phase5_sourced_mean/*_dispatch.npz)
  fig3   cost waterfall                       (phase5_ladder_lv2_sourced_mean.csv)
  fig4   dispatch week 14-20 Apr 2025          (week_trace.csv)
  fig5   sensitivity tornado                   (sensitivity/*)
  fig6   battery x genset sizing               (sensitivity/sizing_sweep.csv)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ["MG_RESULTS_ROOT"]) / "figures"
SRC = Path(os.environ["MG_FIG_SRC"])
os.environ["MG_RESULTS_ROOT"] = str(SRC)          # make_figures reads from SRC

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scripts_v2 import make_figures as MF  # noqa: E402

TMP = OUT / "_raw"
TMP.mkdir(parents=True, exist_ok=True)
MF.figs_dir = lambda: TMP                         # never write into SRC

RENAME = {
    "fig3b_replan_interval_fullcost.png": "fig1b_replan_interval_fullcost.png",
    "fig1_genset_load_factor.png": "fig2_genset_load_factor.png",
    "fig2_cost_waterfall.png": "fig3_cost_waterfall.png",
    "fig6_tornado.png": "fig5_sensitivity_tornado.png",
    "fig4_sizing_heatmap.png": "fig6_sizing_heatmap.png",
}


def fig4_week():
    w = pd.read_csv(SRC / "week_trace.csv", parse_dates=["datetime"])
    lab = {"baseline": ("merit-order baseline", MF.C_BASE),
           "heuristic": ("reactive heuristic", MF.C_HEUR),
           "roll24": ("rolling MILP", MF.C_MILP),
           "oracle": ("perfect-foresight bound", MF.C_CEIL)}
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    o = w[w.controller == "oracle"]
    for name, (lbl, col) in lab.items():
        g = w[w.controller == name]
        if g.empty:
            continue
        axes[0].plot(g.datetime, g.soc_kWh, lw=1.4, color=col, label=lbl)
        axes[1].step(g.datetime, g.genset_kW, where="mid", lw=1.2, color=col, label=lbl)
    axes[1].axhline(0.30 * MF.RATED, color=MF.INK, ls="--", lw=1.0)
    axes[2].fill_between(o.datetime, o.pv_available_kW, step="mid", color="#f0c419", alpha=0.6, label="PV available")
    axes[2].plot(o.datetime, o.load_kW, lw=1.4, color=MF.INK, label="load")
    MF._style(axes[0], "Dispatch week, 14-20 April 2025", "", "battery SOC (kWh)")
    MF._style(axes[1], "", "", "genset output (kW)")
    MF._style(axes[2], "", "date", "kW")
    fig.tight_layout()
    MF._save(fig, "fig4_dispatch_week.png")


def main():
    MF.fig3_replan()
    MF.fig1_load_factor(MF.load_ladder_dispatch())
    MF.fig2_waterfall()
    fig4_week()
    MF.fig6_tornado()
    MF.fig4_sizing()
    for p in TMP.glob("*.png"):
        dst = RENAME.get(p.name, p.name)
        if p.name == "fig3_replan_interval.png":
            dst = "aux_flat_replan_interval.png"
        shutil.move(str(p), OUT / dst)
    TMP.rmdir()
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
