"""Phase 7 figures. One message per figure, no multi-panel composites.

  fig1  genset load-factor histogram, baseline vs corrected-cost controller
  fig2  attribution waterfall
  fig3  cost vs re-planning interval, oracle as a horizontal floor
  fig4  reliability-cost frontier under stress
  fig5  representative week: SOC, genset commitment, PV curtailment
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

INK = "#1b1b1b"
GRID = "#d8d8d8"
C_BASE = "#B45309"      # amber — the reactive baseline
C_EMS = "#1D4ED8"       # blue — the forecast-driven controller
C_ORACLE = "#059669"    # green — the bound


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def fig1_genset_loading(diesel_by_controller, rated_kw=40.0, path=None):
    """The single most persuasive plot: how much of the year the genset spends
    idling at a load factor that destroys it."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    bins = np.arange(0, 1.0001, 0.05)
    colors = {0: C_BASE, 1: C_EMS, 2: C_ORACLE}
    for i, (label, d) in enumerate(diesel_by_controller.items()):
        d = np.asarray(d, float)
        lf = d[d > 1e-9] / rated_kw
        ax.hist(lf, bins=bins, alpha=0.62, label=f"{label} ({len(lf):,} run-h)",
                color=colors.get(i, C_EMS), edgecolor="white", linewidth=0.5)
    ax.axvline(0.30, color=INK, linestyle="--", linewidth=1.1)
    ax.text(0.315, ax.get_ylim()[1] * 0.93, "30% minimum\nrecommended loading",
            fontsize=8, color=INK, va="top")
    _style(ax, "Genset load factor while running",
           "load factor (fraction of 40 kW rating)", "hours per year")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = path or FIGS / "fig1_genset_load_factor.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2_attribution_waterfall(steps, path=None):
    """steps: list of (label, delta_pct); first is the published premium."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
    labels = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    running = np.concatenate([[0.0], np.cumsum(vals)[:-1]])
    for i, (lo, v) in enumerate(zip(running, vals)):
        ax.bar(i, v, bottom=lo, color=C_BASE if v > 0 else C_ORACLE,
               edgecolor="white", linewidth=0.8, width=0.62)
        ax.text(i, lo + v + (0.4 if v >= 0 else -0.9), f"{v:+.1f}",
                ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    _style(ax, "Decomposition of the reported diesel premium", "",
           "premium vs oracle (%)")
    fig.tight_layout()
    p = path or FIGS / "fig2_attribution_waterfall.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3_replan_sweep(df, path=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    floor = df[df.interval_h == 0]
    for name, g in df[df.interval_h > 0].groupby("configuration"):
        g = g.sort_values("interval_h")
        ax.plot(g.interval_h, g.premium_vs_oracle_pct, marker="o", markersize=5,
                linewidth=1.8, label=name,
                color=C_EMS if "hierarch" in name.lower() else C_BASE)
    ax.axhline(0, color=C_ORACLE, linestyle="--", linewidth=1.4,
               label="perfect-foresight oracle (floor)")
    _style(ax, "Cost is set by the execution layer, not the re-planning interval",
           "re-planning interval (hours)", "premium vs oracle (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = path or FIGS / "fig3_replan_sweep.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig4_reliability_cost_frontier(df, path=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
    for name, g in df.groupby("controller"):
        ax.scatter(g.LPSP_pct, g.total_cost_GHS, s=54,
                   color=C_BASE if "base" in name.lower() else C_EMS,
                   label=name, zorder=3, edgecolor="white", linewidth=0.8)
        for _, r in g.iterrows():
            ax.annotate(r.scenario, (r.LPSP_pct, r.total_cost_GHS),
                        fontsize=7.5, xytext=(4, 4),
                        textcoords="offset points", color=INK)
    _style(ax, "Reliability-cost frontier under stress",
           "LPSP (%)", "annual cost (GHS, corrected model)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = path or FIGS / "fig4_reliability_cost_frontier.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig5_representative_week(res, start=None, hours=168, label="", path=None):
    i0 = start if start is not None else 0
    sl = slice(i0, i0 + hours)
    t = np.arange(hours)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), dpi=160, sharex=True)

    ax = axes[0]
    ax.plot(t, res.soc[sl], color=C_EMS, linewidth=1.5)
    ax.fill_between(t, res.soc[sl], alpha=0.12, color=C_EMS)
    _style(ax, f"Representative week{(' — ' + label) if label else ''}", "",
           "SOC (kWh)")

    ax = axes[1]
    d = res.diesel[sl]
    ax.fill_between(t, d, step="mid", color=C_BASE, alpha=0.75)
    ax.plot(t, np.where(d > 1e-9, 40.0, 0.0), color=INK, linewidth=0.8,
            drawstyle="steps-mid", alpha=0.35)
    _style(ax, "", "", "genset (kW)")

    ax = axes[2]
    ax.fill_between(t, res.curtail[sl], step="mid", color=C_ORACLE, alpha=0.7)
    _style(ax, "", "hour of week", "curtailed PV (kW)")

    fig.tight_layout()
    p = path or FIGS / "fig5_representative_week.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
