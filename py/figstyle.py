"""figstyle.py — figure conventions on top of figkit, the authors' in-house figure layout and QA package (not
distributed; every plotted value is in results/).

Journal profile: Rheumatology (OUP): upper-case bold panel letters; widths 84 / 129 / 174 mm, <= 234 mm high.
Colours come only from figkit: compartment / response / weighting colours are the [semantic.ra_rx.*] tables of
rules.toml; compartments always carry a marker shape as well, so colour is never the only cue.
Every saved figure is registered in DuckDB (figure_registry: files + sha256 + QA verdict + provenance).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import figkit as fk
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROFILE = "rheumatology"

SEM = fk.sem("ra_rx")
COMP_ORDER = ["T/NK", "B/plasma", "Myeloid", "Endothelial", "Stromal"]
COMP_COLOR = SEM["compartment"]
COMP_MARKER = {"T/NK": "o", "B/plasma": "s", "Myeloid": "^", "Endothelial": "v", "Stromal": "D", "Mast": "P",
               "Unresolved": "."}
RESP_COLOR = SEM["response"]            # responder / non_responder
WEIGHT_COLOR = SEM["weighting"]         # RA / OA
C = fk.c                                # palette + neutrals (fk.c.ink, fk.c.grey, fk.c["blue@t0.5"], ...)

TRAIT_LABEL = {"ra_eur": "RA (Ishigaki)", "ra_eur_noMHC": "RA, MHC removed", "ra_eur_seropos": "RA seropositive",
               "fg_ra": "RA (FinnGen)", "fg_ra_noMHC": "FinnGen RA, MHC removed",
               "fg_ra_seropos": "FinnGen seropositive", "fg_ra_seroneg": "FinnGen seronegative",
               "fg_oa": "Osteoarthritis", "fg_height": "Height"}


def figure(width: str = "double", height: float | None = None, kind: str = "main"):
    return fk.figure(width, height, profile=PROFILE, kind=kind)


def fmt_p(p: float) -> str:
    """P value for figure text: two decimals down to 0.01, then 'm × 10^e' in mathtext (no 1e-05 notation)."""
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10$^{{{int(e)}}}$"


def pt(role: str) -> float:
    return fk.pt(role, PROFILE)


def save(fig, name: str, provenance: dict, sub: str = "") -> "fk.Saved":
    """fk.save (strict QA) -> figures/<sub>/<name>.{pdf,png}; register files + QA + provenance in DuckDB."""
    import sys
    sys.path.insert(0, str(ROOT / "py"))
    from archive import connect, read, write_table  # noqa: E402
    out = (FIG / sub if sub else FIG)
    out.mkdir(parents=True, exist_ok=True)
    saved = fk.save(fig, out / name)
    rep = saved.report
    verdict = "PASS" if rep.ok and not rep.count("WARN") else (f"{rep.count('WARN')} warn" if rep.ok else "FAIL")
    rows = [dict(figure=name, file=str(Path(p).relative_to(ROOT)), sha256=h, qa=verdict,
                 abbreviations=json.dumps(saved.abbreviations or {}), provenance=json.dumps(provenance),
                 figkit_rules=_rules_version(), saved_at=datetime.now().isoformat(timespec="seconds"))
            for p, h in saved.paths.items()]
    new = pd.DataFrame(rows)
    con = connect(read_only=True)
    try:
        have = "figure_registry" in set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()
    if have:
        old = read("figure_registry")
        new = pd.concat([old[old.figure != name], new], ignore_index=True)
    write_table(new, "figure_registry", script=provenance.get("scripts", ["?"])[-1],
                datasets=["figures"], note="figkit-saved figures: files, sha256, QA verdict, provenance")
    return saved


def _rules_version() -> str:
    import importlib
    return importlib.import_module("figkit.rules").version()
