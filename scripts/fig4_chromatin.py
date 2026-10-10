"""fig4_chromatin.py — Figure 4: RA heritability in synovial chromatin (S-LDSC), with stromal positive controls.

a  S-LDSC matrix: rows = chromatin annotations (in vivo compartment-specific snATAC peaks of 11 RA donors; cultured
   FLS from Ge 2021: unstimulated ATAC, TNF-stimulated ATAC, TNF-gained ATAC, active histone regions), columns = traits
   (RA Ishigaki EUR, RA FinnGen, osteoarthritis, height, Dupuytren's contracture). Colour = coefficient (tau) z, i.e.
   enrichment beyond baselineLD v2.2 and beyond open chromatin in general; ring = BH-FDR < 0.05 within trait; number =
   heritability enrichment (proportion of h2 / proportion of SNPs).
b  enrichment (95% CI) of the five in vivo compartment annotations for RA (Ishigaki EUR), height and Dupuytren's
   contracture; * = coefficient FDR < 0.05.
c  FLS active regions (Ge 2021 definition; 15.7% of common SNPs): proportion of heritability per trait (+/- 1.96 SE);
   dashed line = the proportion of SNPs (no enrichment); dotted line = Ge 2021's "up to 24%" per FLS line; text =
   enrichment and one-sided coefficient P. A large annotation can hold a large share of heritability without being
   enriched beyond the baseline model (RA), while height and Dupuytren's contracture are enriched.
d  active regions built from the same three marks in Roadmap primary T and B cells and in each of the seven cultured
   FLS lines (3 RA, 3 OA, 1 healthy donor), for RA, height and Dupuytren's contracture: a double dissociation.
Source: DuckDB sldsc_chromatin, sldsc_annotations, sldsc_chromatin_r2b (scripts/27_sldsc_chromatin.py, 27b).
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402

IV = {"iv_TNK": "T/NK", "iv_Bplasma": "B/plasma", "iv_Myeloid": "Myeloid", "iv_Endothelial": "Endothelial",
      "iv_Stromal": "Stromal"}
FLS = {"fls_basal": "Unstimulated ATAC", "fls_stim": "TNF-stimulated ATAC", "fls_tnfgained": "TNF-gained ATAC",
       "fls_active": "Active regions (Ge 2021)"}
TRAITS = {"ra_eur": "RA\nEUR", "fg_ra": "RA\nFinnGen", "fg_oa": "OA", "fg_height": "Height",
          "fg_dupuytren": "Dupuytren"}
ZCLIP = 6
FLS_COLOR = C["vermilion@s0.35"]        # cultured fibroblasts: the stromal hue, darker


def have_tables():
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def panel_matrix(ax, D):
    rows = list(IV) + list(FLS)
    Z = D.pivot(index="annotation", columns="trait", values="coef_z").reindex(index=rows, columns=list(TRAITS))
    F = D.pivot(index="annotation", columns="trait", values="coef_fdr").reindex(index=rows, columns=list(TRAITS))
    E = D.pivot(index="annotation", columns="trait", values="enrichment").reindex(index=rows, columns=list(TRAITS))
    ypos = np.arange(len(IV)).tolist() + (np.arange(len(FLS)) + len(IV) + 0.6).tolist()
    norm = mpl.colors.TwoSlopeNorm(vmin=-ZCLIP, vcenter=0, vmax=ZCLIP)
    cmap = fk.cmap(kind="diverging")
    for i, (r, y) in enumerate(zip(rows, ypos)):
        for j, t in enumerate(TRAITS):
            z = Z.loc[r, t]
            ax.add_patch(mpl.patches.Rectangle((j - 0.46, y - 0.44), 0.92, 0.88, color=cmap(norm(np.clip(z, -ZCLIP, ZCLIP))),
                                               lw=0))
            if F.loc[r, t] < 0.05:
                ax.add_patch(mpl.patches.Rectangle((j - 0.46, y - 0.44), 0.92, 0.88, fill=False, ec=C.ink, lw=0.9))
            e = E.loc[r, t]
            ax.text(j, y, f"{e:.1f}" if abs(e) < 99.5 else f"{e:.0f}", ha="center", va="center", fontsize=pt("fine"),
                    color=C.white if abs(z) > 3.5 else C.ink)
    ax.set_xlim(-0.55, len(TRAITS) - 0.45)
    ax.set_ylim(max(ypos) + 0.55, -0.55)
    ax.set_xticks(range(len(TRAITS)), list(TRAITS.values()))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks(ypos, [IV.get(r, FLS.get(r)) for r in rows])
    ax.tick_params(axis="y", length=0)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(COMP_COLOR[IV[r]] if r in IV else FLS_COLOR)
    for s in ax.spines.values():
        s.set_visible(False)
    # row-group labels
    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, (ypos[0] + ypos[len(IV) - 1]) / 2, "In vivo synovium\n(snATAC, 11 RA donors)", transform=tr,
            rotation=270, ha="left", va="center", fontsize=pt("annot"), color=C.grey_dark)
    ax.text(1.02, (ypos[len(IV)] + ypos[-1]) / 2, "Cultured FLS\n(Ge 2021)", transform=tr, rotation=270,
            ha="left", va="center", fontsize=pt("annot"), color=C.grey_dark)
    cax = ax.inset_axes([0.1, -0.09, 0.8, 0.025])
    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                            ticks=[-6, -3, 0, 3, 6])
    cb.ax.tick_params(labelsize=pt("tick"), length=2)
    cb.set_label("Coefficient z (beyond baseline and open chromatin); number = enrichment; box = FDR < 0.05",
                 fontsize=pt("annot"))
    fk.panel(ax, "a", "RA heritability in synovial chromatin")


TSTYLE = {"ra_eur": ("RA (EUR)", C.ink, "o", True), "fg_ra": ("RA (FinnGen)", C.ink, "o", False),
          "fg_oa": ("OA", C.grey, "s", False), "fg_height": ("Height", C.grey_dark, "^", False),
          "fg_dupuytren": ("Dupuytren", C.grey_dark, "D", True)}


def panel_forest(ax, D):
    show = ["ra_eur", "fg_height", "fg_dupuytren"]
    off = {t: (k - 1) * 0.26 for k, t in enumerate(show)}
    for t in show:
        lab, col, mk, filled = TSTYLE[t]
        d = D[D.trait == t].set_index("annotation")
        for i, a in enumerate(IV):
            r = d.loc[a]
            y = i + off[t]
            lo, hi = r.enrichment - 1.96 * r.enrichment_se, r.enrichment + 1.96 * r.enrichment_se
            ax.plot([lo, hi], [y, y], color=col, lw=0.7, zorder=2)
            ax.scatter(r.enrichment, y, s=14, marker=mk, facecolor=col if filled else C.white, edgecolor=col,
                       linewidth=0.7, zorder=3, label=lab if i == 0 else None)
            if r.coef_fdr < 0.05:
                ax.text(hi + 4, y, "*", ha="left", va="center", fontsize=pt("annot"), color=col)
    ax.axvline(0, color=C.grey, lw=0.5, zorder=1)
    ax.axvline(1, color=C.grey, lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(IV)), list(IV.values()))
    for lab, a in zip(ax.get_yticklabels(), IV):
        lab.set_color(COMP_COLOR[IV[a]])
    ax.set_ylim(len(IV) - 0.45, -0.55)
    ax.set_xlabel("Heritability enrichment (95% CI)")
    ax.tick_params(axis="y", length=0)
    fk.legend_top(ax, ncol=3)
    fk.panel(ax, "b", "In vivo compartment chromatin: RA vs stromal traits")


def panel_prop(ax, D):
    """FLS active regions (Ge 2021 definition): proportion of heritability per trait; the dashed line is the
    proportion of SNPs (what no enrichment would give)."""
    d = D[D.annotation == "fls_active"].set_index("trait")
    snp = 100 * d.prop_snps.iloc[0]
    order = ["ra_eur", "fg_ra", "fg_oa", "fg_height", "fg_dupuytren"]
    col = {"ra_eur": C.ink, "fg_ra": C.ink, "fg_oa": C.grey, "fg_height": C.grey, "fg_dupuytren": C.grey}
    for i, t in enumerate(order):
        r = d.loc[t]
        v, e = 100 * r.prop_h2, 196 * r.prop_h2_se
        ax.barh(i, v, height=0.62, color=col[t], lw=0, zorder=2)
        ax.errorbar(v, i, xerr=e, fmt="none", ecolor=C.grey_dark, elinewidth=0.6, capsize=1.5, zorder=3)
        ax.text(v + e + 2, i, f"{r.enrichment:.1f}×, P = {fp(r.coef_p)}", ha="left", va="center",
                fontsize=pt("annot"))
    ax.axvline(snp, color=C.grey_dark, lw=0.7, ls="--", zorder=4, label=f"SNP share {snp:.0f}% (no enrichment)")
    ax.axvline(24, color=C.vermilion, lw=0.7, ls=":", zorder=4, label="Ge 2021: up to 24% per FLS line")
    ax.set_yticks(range(len(order)), [TSTYLE[t][0] for t in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Heritability in FLS active regions (%)")
    ax.tick_params(axis="y", length=0)
    fk.legend_top(ax, ncol=1)
    fk.panel(ax, "c", "Cultured FLS active chromatin: share is not enrichment")


def panel_lines(ax, R):
    """Same construction, different cells: active regions (H3K27ac + H3K4me1 + H3K4me3 union) of Roadmap primary T and
    B cells and of each cultured FLS line (Ge 2021), each on baselineLD + open chromatin, for RA, height and
    Dupuytren's contracture. Colour = coefficient z; number = enrichment; box = BH-FDR < 0.05 (sensitivity family)."""
    d = R[(R.model == "AllPeaks control") &
          (R.annotation.str.startswith("fls_active_") | R.annotation.str.startswith("rm_"))].copy()
    joint = {"KN": "knee", "SH": "shoulder", "MCP": "MCP"}

    def name(a):                                   # fls_active_<donor>_<RA|OA|healthy>_<joint>, as labelled in GEO
        if a == "rm_T_active":
            return "T cells (Roadmap)"
        if a == "rm_B_active":
            return "B cells (Roadmap)"
        donor, dx, j = a.replace("fls_active_", "").split("_")
        return f"{'Healthy-donor' if dx == 'healthy' else dx} FLS, {joint[j]} ({donor})"
    dx_rank = {"RA": 0, "OA": 1, "healthy": 2}
    fls = sorted(set(d.annotation[d.annotation.str.startswith("fls_")]),
                 key=lambda a: (dx_rank[a.split("_")[3]], a))
    rows = ["rm_T_active", "rm_B_active"] + fls
    traits = {"ra_eur": "RA EUR", "fg_height": "Height", "fg_dupuytren": "Dupuytren"}
    Z = d.pivot(index="annotation", columns="trait", values="coef_z").reindex(index=rows, columns=list(traits))
    F = d.pivot(index="annotation", columns="trait", values="coef_fdr").reindex(index=rows, columns=list(traits))
    E = d.pivot(index="annotation", columns="trait", values="enrichment").reindex(index=rows, columns=list(traits))
    ypos = [0, 1] + [k + 2.5 for k in range(len(fls))]
    norm = mpl.colors.TwoSlopeNorm(vmin=-ZCLIP, vcenter=0, vmax=ZCLIP)
    cmap = fk.cmap(kind="diverging")
    for r, y in zip(rows, ypos):
        for j, t in enumerate(traits):
            z = Z.loc[r, t]
            ax.add_patch(mpl.patches.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84,
                                               color=cmap(norm(np.clip(z, -ZCLIP, ZCLIP))), lw=0))
            if F.loc[r, t] < 0.05:
                ax.add_patch(mpl.patches.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84, fill=False, ec=C.ink, lw=0.9))
            ax.text(j, y, f"{E.loc[r, t]:.1f}", ha="center", va="center", fontsize=pt("fine"),
                    color=C.white if abs(z) > 3.5 else C.ink)
    ax.set_xlim(-0.55, len(traits) - 0.45)
    ax.set_ylim(max(ypos) + 0.55, -0.55)
    ax.set_xticks(range(len(traits)), list(traits.values()))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks(ypos, [name(r) for r in rows])
    ax.tick_params(axis="y", length=0)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(COMP_COLOR["T/NK"] if r == "rm_T_active" else COMP_COLOR["B/plasma"] if r == "rm_B_active"
                      else FLS_COLOR)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fk.panel(ax, "d", "Active regions built the same way: T and B cells vs each FLS line")


def fp(p):
    """P value as in the other figures: mathtext power of ten (Arial has no superscript 4-9 glyphs)."""
    if p < 1e-3:
        m, e = f"{p:.0e}".split("e")
        return f"{m} × 10$^{{{int(e)}}}$"
    return f"{p:.2f}"


def main():
    D = read("sldsc_chromatin")
    A = read("sldsc_annotations")
    R2 = read("sldsc_chromatin_r2b") if "sldsc_chromatin_r2b" in have_tables() else None
    fig = figure("double", 150 if R2 is not None else 105)
    nr = 3 if R2 is not None else 2
    gs = fig.add_gridspec(nr, 2, width_ratios=[1.15, 1], height_ratios=[1, 1.0, 1.15][:nr])
    panel_matrix(fig.add_subplot(gs[:, 0]), D)
    panel_forest(fig.add_subplot(gs[0, 1]), D)
    panel_prop(fig.add_subplot(gs[1, 1]), D)
    if R2 is not None:
        panel_lines(fig.add_subplot(gs[2, 1]), R2)
    if "--draft" in sys.argv:
        print(fk.save(fig, Path(sys.argv[-1]), strict=False).report)
        return
    saved = save(fig, "Fig4_chromatin", provenance={
        "source": "db/ra_rx.duckdb::sldsc_chromatin, sldsc_annotations, sldsc_chromatin_r2b",
        "scripts": ["scripts/27_sldsc_chromatin.py", "scripts/27b_sldsc_sensitivity.py", "scripts/fig4_chromatin.py"],
        "transformations": ["a: colour = tau z clipped at |z| = 6; box = BH-FDR < 0.05 within trait; number = enrichment",
                            "b: enrichment +/- 1.96 jackknife SE", "c: proportion of h2 +/- 1.96 SE vs proportion of SNPs"]})
    print(saved.report)


if __name__ == "__main__":
    main()
