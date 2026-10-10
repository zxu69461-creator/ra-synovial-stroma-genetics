"""fig2_genetic_load.py — Figure 2: RA genetic risk maps to immune, not stromal, synovial states.

a  scDRS association z (54 states x 9 GWAS), states grouped by compartment (colour strip) and ordered by RA z
   (Ishigaki, MHC removed) within compartment; dot = BH-FDR < 0.05 within trait. Colour scale clipped at |z| = 8
   (values beyond are shown at the end colour; exact values in scdrs_state_load).
b  per-state z for RA (Ishigaki, MHC removed), osteoarthritis and height (FinnGen), by compartment
   (each point = one state; bar = compartment median).
c  replication: Ishigaki vs FinnGen RA (both MHC removed), Spearman rho.
d  trait specificity: RA (Ishigaki, MHC removed) vs height, Spearman rho.
Source: DuckDB scdrs_state_load (scripts/20_magma.py, 21_scdrs.py, 22_scdrs_summary.py).
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.transforms import ScaledTranslation
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import fmt_p, C, COMP_COLOR, COMP_MARKER, COMP_ORDER, TRAIT_LABEL, figure, fk, pt, save  # noqa: E402

TRAITS = ["ra_eur", "ra_eur_noMHC", "ra_eur_seropos", "fg_ra", "fg_ra_noMHC", "fg_ra_seropos", "fg_ra_seroneg",
          "fg_oa", "fg_height"]
ZCLIP = 8
COL_LABEL = {"ra_eur": "All", "ra_eur_noMHC": "No MHC", "ra_eur_seropos": "Sero+", "fg_ra": "All",
             "fg_ra_noMHC": "No MHC", "fg_ra_seropos": "Sero+", "fg_ra_seroneg": "Sero−", "fg_oa": "OA",
             "fg_height": "Height"}


def scatter_by_comp(ax, d, x, y, loc="left"):
    for comp in COMP_ORDER:
        s = d[d.state_group == comp]
        ax.scatter(s[x], s[y], s=12, c=COMP_COLOR[comp], marker=COMP_MARKER[comp], edgecolor=C.white,
                   linewidth=0.3, label=comp, zorder=3)
    ax.axhline(0, color=C.grey, lw=0.5, zorder=1)
    ax.axvline(0, color=C.grey, lw=0.5, zorder=1)
    r, p = stats.spearmanr(d[x], d[y])
    xt, ha = (0.03, "left") if loc == "left" else (0.97, "right")
    ax.text(xt, 0.03 if loc == "lower right" else 0.97, f"ρ = {r:.2f}\nP = {fmt_p(p)}", transform=ax.transAxes,
            va="bottom" if loc == "lower right" else "top", ha=ha,
            fontsize=pt("annot"))
    return r, p


def panel_heat(ax, w, f):
    Z = w[TRAITS].to_numpy()
    norm = mpl.colors.TwoSlopeNorm(vmin=-ZCLIP, vcenter=0, vmax=ZCLIP)
    im = ax.imshow(np.clip(Z, -ZCLIP, ZCLIP), aspect="auto", cmap=fk.cmap(kind="diverging"), norm=norm,
                   interpolation="nearest", extent=(-0.5, len(TRAITS) - 0.5, len(w) - 0.5, -0.5))
    yy, xx = np.where(f[TRAITS].to_numpy() < 0.05)
    ax.scatter(xx, yy, s=1.6, c=C.ink, marker="o", linewidths=0)
    # compartment strip left of the cells
    for comp in COMP_ORDER:
        idx = np.where(w.state_group == comp)[0]
        ax.add_patch(mpl.patches.Rectangle((-1.05, idx.min() - 0.5), 0.4, len(idx), color=COMP_COLOR[comp],
                                           lw=0))
        if idx.max() < len(w) - 1:
            ax.axhline(idx.max() + 0.5, color=C.white, lw=1.0)
    ax.axvline(6.5, color=C.ink, lw=0.7)
    ax.set_xlim(-1.1, len(TRAITS) - 0.5)
    ax.set_ylim(len(w) - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(TRAITS)), [COL_LABEL[t] for t in TRAITS], rotation=90)
    # GWAS group headers above the column names (offset in points, so they follow the axes through re-layout)
    fig = ax.figure
    fig.canvas.draw()
    h = max(t.get_window_extent().height for t in ax.get_xticklabels()) * 72 / fig.dpi + 2.5
    tr = ax.get_xaxis_transform() + ScaledTranslation(0, h / 72, fig.dpi_scale_trans)
    for x0, x1, name in [(0, 2, "Ishigaki RA"), (3, 6, "FinnGen RA"), (7, 8, "Other")]:
        ax.plot([x0 - 0.42, x1 + 0.42], [1, 1], color=C.ink, lw=0.6, transform=tr, clip_on=False)
        ax.text((x0 + x1) / 2, 1, name, transform=tr + ScaledTranslation(0, 1 / 72, fig.dpi_scale_trans),
                ha="center", va="bottom", fontsize=pt("tick"))
    ax._figkit_title_pad = pt("tick") * 1.2 + 4
    ax.set_yticks(range(len(w)), w.state, fontsize=pt("fine"))
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([0.12, -0.035, 0.76, 0.012])
    cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=[-8, -4, 0, 4, 8])
    cb.ax.set_xticklabels(["≤ −8", "−4", "0", "4", "≥ 8"])
    cb.outline.set_linewidth(0.4)
    cb.set_label("scDRS z (dot: FDR < 0.05)", fontsize=pt("base"))
    fk.panel(ax, "a", "State × trait genetic load")


def panel_strip(ax, w):
    rng = np.random.default_rng(1)
    show = [("ra_eur_noMHC", "RA (MHC removed)"), ("fg_oa", "Osteoarthritis"), ("fg_height", "Height")]
    step = len(COMP_ORDER) + 1.2
    for k, (t, lab) in enumerate(show):
        for j, comp in enumerate(COMP_ORDER):
            v = w.loc[w.state_group == comp, t].to_numpy()
            x = k * step + j + rng.uniform(-0.2, 0.2, len(v))
            ax.scatter(x, v, s=9, c=COMP_COLOR[comp], marker=COMP_MARKER[comp], edgecolor=C.white, linewidth=0.2,
                       zorder=3, label=comp if k == 0 else None)
            ax.plot([k * step + j - 0.32, k * step + j + 0.32], [np.median(v)] * 2, color=C.ink, lw=0.9, zorder=4)
    ax.axhline(0, color=C.grey, lw=0.5)
    ax.set_xticks([k * step + 2 for k in range(len(show))], [s[1] for s in show])
    ax.set_xlim(-0.8, (len(show) - 1) * step + len(COMP_ORDER) - 0.2)
    ax.set_ylabel("scDRS z per state")
    ax.tick_params(axis="x", length=0)
    fk.legend_top(ax)
    fk.panel(ax, "b", "RA load in immune states; OA and height load elsewhere")


def main():
    g = read("scdrs_state_load")
    g["state_group"] = g.state_group.astype(str)
    w = g.pivot_table(index=["state_group", "state"], columns="trait", values="assoc_mcz")[TRAITS].reset_index()
    f = g.pivot_table(index=["state_group", "state"], columns="trait", values="fdr")[TRAITS].reset_index()
    w["_c"] = w.state_group.map({c: i for i, c in enumerate(COMP_ORDER)})
    order = w.sort_values(["_c", "ra_eur_noMHC"], ascending=[True, False]).index
    w, f = w.loc[order].reset_index(drop=True), f.loc[order].reset_index(drop=True)

    fig = figure("double", 155)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.72, 1], height_ratios=[1, 1.05])
    right = gs[1, 1].subgridspec(1, 2)
    panel_heat(fig.add_subplot(gs[:, 0]), w, f)
    panel_strip(fig.add_subplot(gs[0, 1]), w)

    axc = fig.add_subplot(right[0])
    scatter_by_comp(axc, w, "ra_eur_noMHC", "fg_ra_noMHC", loc="lower right")
    axc.set(xlabel="scDRS z, Ishigaki RA", ylabel="scDRS z, FinnGen RA")
    fk.panel(axc, "c", "Ishigaki vs FinnGen")
    axd = fig.add_subplot(right[1])
    scatter_by_comp(axd, w, "ra_eur_noMHC", "fg_height", loc="right")
    axd.set(xlabel="scDRS z, Ishigaki RA", ylabel="scDRS z, height")
    fk.panel(axd, "d", "RA vs height")

    saved = save(fig, "Fig2_genetic_load", provenance={
        "source": "db/ra_rx.duckdb::scdrs_state_load",
        "scripts": ["scripts/20_magma.py", "scripts/21_scdrs.py", "scripts/22_scdrs_summary.py",
                    "scripts/fig2_genetic_load.py"],
        "transformations": ["scDRS 1,000 control gene sets; BH-FDR within trait over 54 states",
                            f"colour scale clipped at |z| = {ZCLIP}", "strip x-jitter seed 1 (display only)"],
        "n": "54 curated synovial states, 83,503 cells (E-MTAB-11791)"})
    print(saved.report)


if __name__ == "__main__":
    main()
