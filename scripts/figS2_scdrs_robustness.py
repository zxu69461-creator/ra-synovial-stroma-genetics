"""figS2_scdrs_robustness.py — Supplementary Fig S2: robustness of state-level RA genetic load (scDRS z).

a  gene-set size: top-500 vs the primary top-1,000 MAGMA genes (Ishigaki RA, MHC removed)
b  gene-set size: top-2,000 vs top-1,000 MAGMA genes (Ishigaki RA, MHC removed)
c  MHC removed vs included (Ishigaki RA, top-1,000 genes)
d  FinnGen seronegative vs seropositive RA (top-1,000 genes)
Axes: scDRS association z. Each point = one of 54 states; colour + marker = compartment; dashed line = identity;
Spearman rho annotated.
Sources: scdrs_group_state, scdrs_group_state_top500, scdrs_group_state_top2000.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402


def z(table, trait):
    t = read(table)
    return t[t.trait == trait].set_index("state").assoc_mcz


def main():
    grp = read("SELECT DISTINCT state, state_group FROM atlas_cells_state")
    grp = grp.assign(state_group=grp.state_group.astype(str)).set_index("state").state_group
    pairs = [
        (z("scdrs_group_state", "ra_eur_noMHC"), z("scdrs_group_state_top500", "ra_eur_noMHC"),
         "Top-1,000 genes", "Top-500 genes", "Smaller gene set"),
        (z("scdrs_group_state", "ra_eur_noMHC"), z("scdrs_group_state_top2000", "ra_eur_noMHC"),
         "Top-1,000 genes", "Top-2,000 genes", "Larger gene set"),
        (z("scdrs_group_state", "ra_eur"), z("scdrs_group_state", "ra_eur_noMHC"),
         "MHC included", "MHC removed", "MHC region"),
        (z("scdrs_group_state", "fg_ra_seropos"), z("scdrs_group_state", "fg_ra_seroneg"),
         "Seropositive RA", "Seronegative RA", "Serotype (FinnGen)"),
    ]
    fig = figure("double", 65, kind="supp")
    gs = fig.add_gridspec(1, 4)
    for k, ((x, y, xl, yl, title), lab) in enumerate(zip(pairs, "abcd")):
        ax = fig.add_subplot(gs[k])
        d = pd.DataFrame({"x": x, "y": y.reindex(x.index)}).join(grp)
        for comp in COMP_ORDER:
            s = d[d.state_group == comp]
            ax.scatter(s.x, s.y, s=9, c=COMP_COLOR[comp], marker=COMP_MARKER[comp], edgecolor=C.white,
                       linewidth=0.2, label=comp, zorder=3)
        lo, hi = min(d.x.min(), d.y.min()) - 0.5, max(d.x.max(), d.y.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", color=C.grey, lw=0.6, zorder=2)
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ticks = [t for t in ax.get_yticks() if lo <= t <= hi - 0.12 * (hi - lo)]   # keep the letter's corner clear
        ax.set_yticks(ticks)
        ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= hi])
        r, _ = stats.spearmanr(d.x, d.y)
        ax.text(0.04, 0.96, f"ρ = {r:.2f}\nn = {len(d)}", transform=ax.transAxes, va="top", fontsize=pt("annot"))
        ax.set(xlabel=f"{xl} (z)", ylabel=f"{yl} (z)")
        fk.panel(ax, lab, title)
        if k == 0:
            ax.legend(loc="lower right", frameon=False, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.3)
    saved = save(fig, "FigS2_scdrs_robustness", provenance={
        "source": "db/ra_rx.duckdb::scdrs_group_state{,_top500,_top2000}",
        "scripts": ["scripts/21_scdrs.py", "scripts/figS2_scdrs_robustness.py"],
        "n": "54 states"})
    print(saved.report)


if __name__ == "__main__":
    main()
