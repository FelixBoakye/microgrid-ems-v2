"""Load-variant file resolution.

Set MG_LOAD=v2 to run the whole pipeline against the `load_v2` demand series;
unset or MG_LOAD=v1 keeps the published `load_v1` series and its existing files
untouched.

Naming note: the existing `v2_*` dataset files mean "the D5-corrected dataset",
not "load version 2". To avoid collision the load-v2 artefacts use `lv2_*`.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def variant() -> str:
    return os.environ.get("MG_LOAD", "v1").strip().lower()


def is_v2() -> bool:
    return variant() == "v2"


def load_csv() -> Path:
    """The realised demand series for the active variant."""
    v = variant()
    if v == "v1":
        return DATA / "synthetic_load.csv"
    if v == "v2":
        return DATA / "synthetic_load_lv2.csv"
    return DATA / f"synthetic_load_{v}.csv"


def prefix() -> str:
    """Filename prefix for derived dataset/forecast artefacts."""
    v = variant()
    if v == "v1":
        return "v2_"          # the D5-corrected dataset for the published series
    if v == "v2":
        return "lv2_"
    return f"l{v}_"


def data(name: str) -> Path:
    return DATA / f"{prefix()}{name}"


def results_root() -> Path:
    """Directory that holds this variant's result files.

    ``MG_RESULTS_ROOT`` sends a rerun to its own tree. Load v2 writes directly
    into that directory. Every other variant gets a subdirectory, so a mix12
    run cannot overwrite the v2 files or the canonical results.
    """
    override = os.environ.get("MG_RESULTS_ROOT", "").strip()
    if override:
        root = Path(override)
        if not root.is_absolute():
            root = ROOT / root
        v = variant()
        p = root if v == "v2" else (root / v)
        p.mkdir(parents=True, exist_ok=True)
        return p
    v = variant()
    return RESULTS if v == "v1" else (RESULTS / v)


def results_subdir(base: str) -> Path:
    """results/<base> for load_v1, results/v2/<base> for load_v2.

    With ``MG_RESULTS_ROOT`` set, the same relative name lands under
    ``results_root()`` and the canonical tree is left untouched.
    """
    p = results_root() / base
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def smoke_hours() -> int:
    """Driver smoke test only: MG_SMOKE_HOURS truncates every series to N hours
    and keeps one variant per group. Unset in every real run (the driver refuses it)."""
    raw = os.environ.get("MG_SMOKE_HOURS", "").strip()
    return int(raw) if raw else 0


def smoke_trim(x):
    n = smoke_hours()
    return x[:n] if n else x


def smoke_groups(items, key=lambda it: it):
    """Keep the first item of each group under smoke; everything otherwise."""
    if not smoke_hours():
        return list(items)
    seen, out = set(), []
    for it in items:
        g = key(it)
        if g not in seen:
            seen.add(g)
            out.append(it)
    return out


def results_file(name: str) -> Path:
    """A results path that may already start with ``v2/``.

    Under ``MG_RESULTS_ROOT`` the leading ``v2/`` is dropped for the v2
    variant, because ``results_root()`` is already that rerun directory.
    """
    rel = Path(name)
    if os.environ.get("MG_RESULTS_ROOT", "").strip():
        if variant() == "v2" and rel.parts and rel.parts[0] == "v2":
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
        return results_root() / rel
    return RESULTS / rel


def tag() -> str:
    v = variant()
    return "" if v == "v1" else f"_load{v}"
