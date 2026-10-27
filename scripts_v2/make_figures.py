"""Phase 7 figures. One message per figure, no multi-panel composites.

  fig1  genset load-factor histogram — the most persuasive plot in the paper
  fig2  cost waterfall, baseline to perfect-foresight ceiling
  fig3  cost vs re-planning interval, oracle as a horizontal floor
  fig4  sizing heatmap — where intelligence pays and where capacity substitutes
  fig5  representative week: SOC, genset commitment, curtailment
  fig6  tornado — what actually moves the headline

Figure 4 substitutes for the brief's "reliability-cost frontier under stress
scenarios". Every controller sits at 0.0000% LPSP on this system, so reliability
does not discriminate between them and a frontier plot would be a single point.
The sizing sweep answers the question a reviewer actually asks — why not buy more
battery instead of intelligence — and it has an answer that is not flattering,
which makes it the more useful figure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cost_model as cm
from . import paths as P

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INK, GRID = "#1b1b1b", "#d9d9d9"
C_BASE, C_HEUR, C_MILP, C_CEIL = "#B45309", "#7C3AED", "#1D4ED8", "#059669"
RATED = 40.0


def figs_dir():
    d = P.results_subdir("figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8.5)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5)


def _save(fig, name):
    p = figs_dir() / name
    fig.savefig(p, bbox_inches="tight", dpi=170)
    plt.close(fig)
    try:
        shown = p.relative_to(ROOT)
    except ValueError:
        shown = p
    print(f"  {shown}")
    return p


def load_fold(year=2025):
    z = np.load(P.results_subdir("walkforward") / f"fold_{year}_dispatch.npz")
    pv = pd.read_csv(DATA / "pv_output.csv", parse_dates=["datetime"])
    ld = pd.read_csv(P.load_csv(), parse_dates=["datetime"])
    d = pv[["datetime", "PV_power_kW"]].merge(
        ld[["datetime", "total_load_kW"]], on="datetime").sort_values("datetime")
    d = d[d.datetime.dt.year == year].reset_index(drop=True)
    return z, d


def load_ladder_dispatch(ladder="phase5_sourced_mean"):
    """Hourly dispatch of the Phase 5 ladder run (sourced-mean fuel curve,
    heuristic theta=70%, psi=50%) -- the run that writes
    results/phase5_ladder_lv2_sourced_mean.csv, i.e. Table VIII."""
    out = {}
    for name in ("baseline", "heuristic", "roll24", "oracle"):
        npz = np.load(P.results_subdir(ladder) / f"{name}_dispatch.npz")
        out[f"{name}_diesel"] = npz["diesel"]
    return out


# ------------------------------------------------------------------ fig 1
def fig1_load_factor(z, out_name="fig1_genset_load_factor.png"):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bins = np.arange(0, 1.0001, 0.05)
    for name, lbl, col in (("baseline", "merit-order baseline", C_BASE),
                           ("heuristic", "reactive heuristic", C_HEUR),
                           ("roll24", "rolling MILP", C_MILP),
                           ("oracle", "perfect-foresight ceiling", C_CEIL)):
        g = z[f"{name}_diesel"]
        lf = g[g > 1e-9] / RATED
        ax.hist(lf, bins=bins, histtype="step", lw=1.9, color=col,
                label=f"{lbl} — {len(lf):,} run-h, mean {lf.mean() * 100:.0f}%")
    ax.axvline(0.30, color=INK, ls="--", lw=1.1)
    top = ax.get_ylim()[1]
    ax.annotate("30% minimum recommended\nloading (wet stacking)",
                xy=(0.30, top * 0.34), xytext=(0.38, top * 0.46),
                fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    _style(ax, "Genset load factor while running, test year 2025",
           "load factor (fraction of 40 kW rating)", "hours per year",
           legend=False)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    return _save(fig, out_name)


# ------------------------------------------------------------------ fig 2
def _phase5_ladder_csv():
    if os.environ.get("MG_RESULTS_ROOT", "").strip():
        return P.results_root() / "phase5_ladder_lv2_sourced_mean.csv"
    return ROOT / "results" / "phase5_ladder_lv2_sourced_mean.csv"


def fig2_waterfall():
    d = pd.read_csv(_phase5_ladder_csv()).set_index("config")
    base, heur = d.loc["baseline", "total_cost_GHS"], d.loc["heuristic", "total_cost_GHS"]
    milp, ceil = d.loc["roll24", "total_cost_GHS"], d.loc["oracle", "total_cost_GHS"]
    steps = [("merit-order\nbaseline", base, None),
             ("loading\npolicy", -(base - heur), C_HEUR),
             ("foresight\n(rolling MILP)", -(heur - milp), C_MILP),
             ("perfect\ninformation", -(milp - ceil), C_CEIL),
             ("ceiling", ceil, None)]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    lo = ceil - (base - ceil) * 0.22
    hi = base + (base - ceil) * 0.22
    span = hi - lo
    run = base
    for i, (lbl, v, col) in enumerate(steps):
        if col is None:
            ax.bar(i, v - lo, bottom=lo, color="#54606b", width=0.6)
            ax.text(i, v + span * 0.025, f"{v:,.0f}", ha="center", fontsize=9,
                    color=INK)
            run = v
        else:
            ax.bar(i, v, bottom=run, color=col, width=0.6)
            ax.text(i, run + v - span * 0.075,
                    f"{v:,.0f}\n({v / base * 100:+.1f}%)", ha="center",
                    fontsize=8.5, color=INK)
            run += v
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    _style(ax, "Where the saving comes from (corrected cost model, load_v2)",
           "", "annual cost (GHS)", legend=False)
    return _save(fig, "fig2_cost_waterfall.png")


# ------------------------------------------------------------------ fig 3
def fig3_replan():
    s = pd.read_csv(P.results_subdir("replan_sweep.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for name, g in s[s.interval_h > 0].groupby("configuration"):
        g = g.sort_values("interval_h")
        hier = "hierarch" in name.lower()
        ax.plot(g.interval_h, g.premium_vs_oracle_pct, marker="o", ms=5, lw=1.9,
                color=C_MILP if hier else C_BASE,
                label="hierarchical execution" if hier else "setpoint execution")
    ax.axhline(0, color=C_CEIL, ls="--", lw=1.5,
               label="perfect-foresight oracle (floor)")
    ax.set_xticks([1, 2, 3, 6, 12, 24])
    _style(ax, "Flat cost model (as published): execution layer, not interval",
           "re-planning interval (hours)", "premium vs oracle (%)")
    out = _save(fig, "fig3_replan_interval.png")
    pc = P.results_subdir("replan_sweep_corrected.csv")
    if pc.exists():
        c = pd.read_csv(pc)
        g = c[c.interval_h > 0].sort_values("interval_h")
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(g.interval_h, g.premium_vs_oracle_pct, marker="o", ms=5, lw=1.9,
                color=C_MILP, label="rolling MILP, hierarchical execution")
        ax.axhline(0, color=C_CEIL, ls="--", lw=1.5, label="perfect-foresight bound")
        ax.set_xticks([1, 2, 3, 6, 12, 24])
        _style(ax, f"Full cost model, c_s = {float(c.start_cost_ghs.iloc[0]):g} GHS/start",
               "re-planning interval (hours)", "premium vs bound (%)")
        _save(fig, "fig3b_replan_interval_fullcost.png")
    return out


# ------------------------------------------------------------------ fig 4
def fig4_sizing():
    s = pd.read_csv(P.results_subdir("sensitivity") / "sizing_sweep.csv")
    piv = s.pivot(index="battery_kwh", columns="pv_kwp",
                  values="milp_vs_baseline_pct").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    lim = np.abs(piv.values).max()
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:.0f}" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=9.5,
                    color=INK, weight="bold" if v < 0 else "normal")
    fig.colorbar(im, ax=ax, label="MILP saving vs baseline (%)")
    _style(ax, "Where intelligence pays — and where capacity substitutes for it",
           "PV (kWp)", "battery (kWh)", legend=False)
    ax.grid(False)
    return _save(fig, "fig4_sizing_heatmap.png")


# ------------------------------------------------------------------ fig 5
def fig5_week(z, d, start_month=6, start_day=16):
    i0 = int(d.index[(d.datetime.dt.month == start_month)
                     & (d.datetime.dt.day == start_day)
                     & (d.datetime.dt.hour == 0)][0])
    sl = slice(i0, i0 + 168)
    t = np.arange(168)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=True)

    ax = axes[0]
    for name, lbl, col in (("baseline", "baseline", C_BASE),
                           ("roll24", "rolling MILP", C_MILP)):
        ax.plot(t, z[f"{name}_soc"][sl], lw=1.6, color=col, label=lbl)
    _style(ax, f"Representative week — {d.datetime.iloc[i0]:%d %b %Y}", "",
           "SOC (kWh)")

    ax = axes[1]
    ax.fill_between(t, z["baseline_diesel"][sl], step="mid", color=C_BASE,
                    alpha=0.45, label="baseline")
    ax.fill_between(t, z["roll24_diesel"][sl], step="mid", color=C_MILP,
                    alpha=0.65, label="rolling MILP")
    ax.axhline(0.30 * RATED, color=INK, ls="--", lw=1.0)
    ax.text(2, 0.30 * RATED + 1.0, "30% min loading", fontsize=7.5, color=INK)
    _style(ax, "", "", "genset (kW)")

    ax = axes[2]
    ax.fill_between(t, d.PV_power_kW.values[sl], step="mid", color="#f0c419",
                    alpha=0.45, label="PV available")
    ax.plot(t, d.total_load_kW.values[sl], lw=1.4, color=INK, label="load")
    _style(ax, "", "hour of week", "kW")
    fig.tight_layout()
    return _save(fig, "fig5_representative_week.png")


# ------------------------------------------------------------------ fig 6
def fig6_tornado():
    sub = json.loads((P.results_subdir("sensitivity") /
                      "window_check_sub.json").read_text())
    nom = sub["milp_vs_heuristic_pct"]
    g = pd.read_csv(P.results_subdir("sensitivity") / "genset_sensitivity.csv")
    dm = pd.read_csv(P.results_subdir("sensitivity") / "demand_sensitivity.csv")
    lin = pd.read_csv(P.results_subdir("sensitivity") / "linear_sensitivity.csv")
    sd = pd.read_csv(P.results_subdir("sensitivity") / "seed_variance.csv")

    rows = []

    def add(label, vals, unsourced=False):
        v = np.asarray(vals, float)
        rows.append((label, v.min(), v.max(), unsourced))

    for p in ("start cost $c_s$", "battery throughput $c_b$"):
        key = "start cost c_s" if "c_s" in p else "battery throughput cost c_b"
        add(p, lin[lin.parameter == key].milp_vs_heuristic_pct, True)
    for lbl, pre in (("min up/down time", "min up/down"),
                     ("fuel curve $A$", "fuel A"),
                     ("min load ratio", "min load ratio"),
                     ("fuel curve $B$", "fuel B")):
        add(lbl, g[g.label.str.startswith(pre)].milp_vs_heuristic_pct)
    for lbl, pre in (("temperature response $\\beta$", "temp response"),
                     ("productive day-of-week", "day-of-week"),
                     ("AR(1) coefficient $\\phi$", "phi =")):
        add(lbl, dm[dm.label.str.startswith(pre)].milp_vs_heuristic_pct)
    add("forecaster seed (noise)", sd.milp_vs_heuristic_pct)

    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (lbl, lo, hi, uns) in enumerate(rows):
        ax.barh(i, hi - lo, left=lo, height=0.62,
                color="#B45309" if uns else "#1D4ED8", alpha=0.9)
        ax.text(hi + 0.25, i, f"{hi - lo:.1f} pp", va="center", fontsize=8.5,
                color=INK)
    ax.axvline(nom, color=INK, ls="--", lw=1.2)
    ax.text(nom + 0.15, len(rows) - 0.4, f"nominal {nom:.1f}%", fontsize=8,
            color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    _style(ax, "What moves the headline (amber = still unsourced)",
           "MILP saving vs heuristic (%)", "", legend=False)
    return _save(fig, "fig6_tornado.png")


def main():
    z, d = load_fold(2025)
    print("figures:")
    fig1_load_factor(load_ladder_dispatch())
    fig2_waterfall()
    fig3_replan()
    fig4_sizing()
    fig5_week(z, d)
    fig6_tornado()


if __name__ == "__main__":
    main()
