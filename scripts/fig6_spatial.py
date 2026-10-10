"""fig6_spatial.py — Supplementary Figure 4 (FigS4_spatial): spatial niches of genetically loaded and decoupled
states (Xenium 5k, 17 patients).
Descriptive only.

a,b  patient 1 (lympho-myeloid, remission): annotated cells coloured by compartment (a) and by niche (b; niche codes
     above the map, full names in f)
c    patient 16 (pauci-immune, non-remission): niches (section rotated 90 degrees to fit the row)
d    per-patient niche fractions, grouped by pathotype; 6-month outcome marked above the bars. Niche fractions did not
     differ by remission status (xenium_niche_response: Mann-Whitney, remission vs non-remission, all P > 0.3)
e    niche identity: log2 enrichment of the 12 most variable states in each niche's neighbourhood composition
f    mean neighbourhood genetic load (scDRS z, RA MHC removed; mean over the 20 nearest cells) of cells in each niche;
     the bar labels double as the niche key for a-e. Neighbourhood load largely reflects the niche's own cell types:
     across states, load of cross-compartment neighbours correlates negatively with the state's own load
     (rho = -0.51, xenium_state_load_niche), so no spatial segregation of genetic load is claimed.
Spatial panels are rasterised (100k+ points); scale bars 1,000 um.
Sources: xenium_cells, xenium_cells_niche, xenium_niche_profile, xenium_patient_niche, xenium_niche_response,
xenium_state_load_niche.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_ORDER, figure, fk, pt, save  # noqa: E402

# short display names; keyed by the automatic "N<k>: top-2 enriched states" label from 62_xenium_niches.py (seed 0);
# niche colours: extended palette + rule tints/shades, deliberately distinct from the compartment colours so that
# panel b (niches) reads as territories rather than repeating panel a; the set was chosen so that no pair (niches and
# compartments together) is confusable under simulated deutan / protan / tritan vision (figkit cvd_pairs, dE >= 8)
NICHE = {
    "N0": ("Lining / MMP2+ sublining", "Fib PRG4+ lining", "vermilion@s0.35"),
    "N1": ("Arterial & lymphatic vessels", "EC arterial", "sky"),
    "N2": ("SPP1+ hypoxic macrophages", "Mac SPP1+", "olive"),
    "N3": ("B-cell aggregate", "B unswitched memory", "magenta"),
    "N4": ("T-cell aggregate (Tph)", "T CD4+ naive", "indigo"),
    "N5": ("Lining + resident macrophages", "Fib PRG4+ CLIC5+ lining", "amber"),
    "N6": ("Resident macrophage / SCARA5+ sublining", "Mac TIMD4+ LYVE1+ resident", "cyan"),
    "N7": ("Plasma cell / ABC niche", "Plasma cell", "rose@t0.6"),
    "N8": ("Venous vessels", "EC venous LIFR+", "black"),
    "N9": ("CD34+ sublining", "Fib CD34+ sublining", "wine"),
}
RESP_MARK = {"remission": ("o", C.white), "borderline": ("s", C.grey), "non-remission": ("X", C.ink)}
PATHO_ORDER = ["Lympho-myeloid", "Diffuse Myeloid", "Pauci Immune"]
PATHO_LABEL = {"Lympho-myeloid": "Lympho-myeloid", "Diffuse Myeloid": "Diffuse myeloid", "Pauci Immune": "Pauci-immune"}
SCALE_UM = 1000


def spatial(ax, x, y, colors, s, title=None, letter=None):
    ax.scatter(x, y, s=s, c=colors, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)
    x0, y0, y1 = x.min(), y.min(), y.max()
    pad = 0.09 * (y1 - y0)
    ax.plot([x0, x0 + SCALE_UM], [y0 - pad, y0 - pad], color=C.ink, lw=1.2, solid_capstyle="butt")
    ax.text(x0 + SCALE_UM + 150, y0 - pad, f"{SCALE_UM:,} µm", va="center", fontsize=pt("annot"))
    ax.margins(0.02)
    if letter:
        fk.panel(ax, letter, title)


def main():
    cells = read("SELECT c.patient, c.x_centroid, c.y_centroid, c.compartment, n.niche_name, n.nb_load "
                 "FROM xenium_cells c JOIN xenium_cells_niche n USING (cell)")
    niches = sorted(cells.niche_name.unique(), key=lambda s: int(s.split(":")[0][1:]))
    code = {n: n.split(":")[0] for n in niches}
    for n in niches:                       # guard: display names must match the automatic niche identity
        assert NICHE[code[n]][1] in n, f"niche {n} no longer matches display name {NICHE[code[n]]}"
    ncol = {n: fk.c[NICHE[code[n]][2]] for n in niches}
    nres = read("xenium_niche_response")

    fig = figure("double", 210, kind="supp")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.86, 0.95, 1.25])
    top = gs[0].subgridspec(1, 2)
    mid = gs[1].subgridspec(1, 2, width_ratios=[0.85, 1.15])
    bot = gs[2].subgridspec(1, 2, width_ratios=[0.8, 1.2])

    # a, b: patient 1
    p1 = cells[cells.patient == "1"]
    axa = fig.add_subplot(top[0])
    spatial(axa, p1.x_centroid, p1.y_centroid, p1.compartment.map(COMP_COLOR), 0.12, None)
    fk.legend_top(axa, handles=[mpl.lines.Line2D([], [], ls="", marker="o", ms=3.5, color=COMP_COLOR[c], label=c)
                                for c in COMP_ORDER], handletextpad=0.1, columnspacing=0.7, align="right")
    fk.panel(axa, "a", "Patient 1 (lympho-myeloid, remission): compartments")
    axb = fig.add_subplot(top[1])
    spatial(axb, p1.x_centroid, p1.y_centroid, p1.niche_name.map(ncol), 0.12)
    fk.legend_top(axb, handles=[mpl.lines.Line2D([], [], ls="", marker="o", ms=3.5, color=ncol[n], label=code[n])
                                for n in niches], handletextpad=0.05, columnspacing=0.5, align="right")
    fk.panel(axb, "b", "Patient 1: niches (key in F)")

    # c: patient 16, rotated 90 degrees clockwise (x' = y, y' = -x)
    p16 = cells[cells.patient == "16"]
    axc = fig.add_subplot(mid[0])
    spatial(axc, p16.y_centroid, -p16.x_centroid, p16.niche_name.map(ncol), 1.2,
            "Patient 16 (pauci-immune, non-remission): niches", "c")

    # d: per-patient niche fractions
    axd = fig.add_subplot(mid[1])
    pn = read("xenium_patient_niche")
    W = pn.pivot(index="patient", columns="niche_name", values="fraction")[niches]
    meta = pn.drop_duplicates("patient").set_index("patient")[["pathotype", "response"]]
    rank = {"remission": 0, "borderline": 1, "non-remission": 2}
    porder = meta.assign(_p=meta.pathotype.map({p: i for i, p in enumerate(PATHO_ORDER)}),
                         _r=meta.response.map(rank), _n=meta.index.astype(int)).sort_values(["_p", "_r", "_n"]).index
    xs = np.arange(len(porder))
    bottom = np.zeros(len(porder))
    for n in niches:
        v = W.loc[porder, n].to_numpy()
        axd.bar(xs, v, bottom=bottom, color=ncol[n], width=0.82, linewidth=0)
        bottom += v
    for i, p in enumerate(porder):
        m, c = RESP_MARK[meta.loc[p, "response"]]
        axd.scatter(i, 1.07, marker=m, s=14, facecolor=c, edgecolor=C.ink, linewidth=0.5)
    axd.set_xticks(xs, porder)
    axd.tick_params(axis="x", length=0)
    axd.set_xlim(-0.6, len(porder) - 0.4)
    axd.set_ylim(0, 1.13)
    axd.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axd.spines["left"].set_bounds(0, 1)
    axd.set_ylabel("Fraction of annotated cells")
    band = mpl.transforms.blended_transform_factory(axd.transData, axd.transAxes)
    for p in PATHO_ORDER:
        idx = [i for i, q in enumerate(porder) if meta.loc[q, "pathotype"] == p]
        axd.plot([min(idx) - 0.4, max(idx) + 0.4], [-0.1, -0.1], color=C.grey_dark, lw=0.7, transform=band,
                 clip_on=False)
        axd.text((min(idx) + max(idx)) / 2, -0.13, PATHO_LABEL[p], transform=band, ha="center", va="top",
                 fontsize=pt("annot"))
    pmin = nres.p.min()
    axd.set_xlabel(f"Patient (remission vs non-remission, every niche P ≥ {np.floor(pmin * 100) / 100:.2f})",
                   labelpad=17)
    fk.legend_top(axd, handles=[mpl.lines.Line2D([], [], ls="", marker=m, ms=3.8, mfc=c, mec=C.ink, mew=0.5,
                                                 label=r.capitalize()) for r, (m, c) in RESP_MARK.items()],
                  align="right")
    fk.panel(axd, "d", "Niche composition per patient")

    # e: niche identity (transposed heatmap: states x niches)
    axe = fig.add_subplot(bot[0])
    prof = read("xenium_niche_profile")
    E = prof.pivot(index="niche", columns="state", values="enrichment").loc[niches]
    L = np.log2(E.clip(lower=1 / 16, upper=16))
    top_states = L.var().sort_values(ascending=False).index[:12]
    M = L[top_states].T
    im = axe.imshow(M.to_numpy(), cmap=fk.cmap(kind="diverging"), vmin=-4, vmax=4, aspect="auto",
                    interpolation="nearest")
    axe.set_yticks(range(len(top_states)), top_states)
    axe.set_xticks(range(len(niches)), [code[n] for n in niches])
    axe.tick_params(length=0)
    axe.set_ylim(len(top_states) - 0.5, -1.3)
    for j, n in enumerate(niches):
        axe.add_patch(mpl.patches.Rectangle((j - 0.45, -1.2), 0.9, 0.55, color=ncol[n], lw=0))
    cb = fig.colorbar(im, ax=axe, location="bottom", shrink=0.7, aspect=18, pad=0.02)
    cb.set_label("log2 enrichment in neighbourhood (±4 cap)", fontsize=pt("tick"))
    cb.ax.tick_params(labelsize=pt("tick"))
    fk.panel(axe, "e", "Niche identity")

    # f: neighbourhood genetic load per niche (bar labels = niche key)
    axf = fig.add_subplot(bot[1])
    nl = cells.groupby("niche_name").agg(load=("nb_load", "mean"), n=("nb_load", "size")).loc[niches]
    order = nl.sort_values("load", ascending=False).index
    y = np.arange(len(order))
    axf.barh(y, nl.loc[order, "load"], color=[ncol[n] for n in order], height=0.7)
    axf.axvline(0, color=C.grey_dark, lw=0.6)
    for yi, n in zip(y, order):
        v = nl.loc[n, "load"]
        axf.text(v + (0.08 if v >= 0 else -0.08), yi, f"{nl.loc[n, 'n'] / 1000:.0f}k", va="center",
                 ha="left" if v >= 0 else "right", fontsize=pt("annot"))
    axf.set_yticks(y, [f"{code[n]} {NICHE[code[n]][0]}" for n in order])
    axf.set_ylim(len(order) - 0.4, -0.6)
    axf.set_xlim(nl.load.min() - 0.7, nl.load.max() + 0.6)
    axf.set_xlabel("Mean neighbourhood genetic load (scDRS z)")
    fk.fit_ticklabels(axf, "y")
    fk.panel(axf, "f", "Neighbourhood genetic load by niche")

    saved = save(fig, "FigS4_spatial", provenance={
        "source": "db/ra_rx.duckdb::xenium_cells, xenium_cells_niche, xenium_niche_profile, xenium_patient_niche, "
                  "xenium_niche_response",
        "scripts": ["scripts/60_xenium_extract.py", "scripts/61_xenium_annotate.py", "scripts/62_xenium_niches.py",
                    "scripts/fig6_spatial.py"],
        "transformations": ["cells >= 20 transcripts; Unresolved/low-count cells excluded",
                            "kNN = 20 neighbours; MiniBatchKMeans k = 10 (seed 0)", "enrichment clipped at 16-fold",
                            "patient 16 section rotated 90 degrees for display"],
        "n": f"17 patients; {len(cells):,} annotated cells (Zenodo 15058262)"})
    print(saved.report)


if __name__ == "__main__":
    main()
