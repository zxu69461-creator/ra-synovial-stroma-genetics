"""figS6_chromatin_sensitivity.py — Supplementary Fig. S6: sensitivity analyses of the chromatin-level result.

a  accessible chromatin (half of all peaks with the highest pooled CPM per compartment; absolute, not specificity) and
   lining-fibroblast, sublining-fibroblast and mural-cell specific chromatin, on baselineLD v2.2 + open chromatin.
b  the nine primary annotations of Fig. 4 and the five accessible halves (RA, height) refitted on baselineLD v2.2
   alone (no open-chromatin control).
c  in vivo compartment-specific peaks extended by 500 bp (control extended likewise), five traits.
   a-c: colour = coefficient z; number = heritability enrichment; box = BH-FDR < 0.05 within model x trait.
d  fine-mapped RA variants (Ishigaki 2022, PIP > 0.1): PIP-weighted share in each in vivo compartment annotation
   divided by its share of SNPs (fold); dashed line = 1.
Sources: sldsc_chromatin_r2b, sldsc_chromatin, finemap_overlap.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import C, COMP_COLOR, figure, fk, pt, save  # noqa: E402

ZCLIP = 6
COMP = {"TNK": "T/NK", "Bplasma": "B/plasma", "Myeloid": "Myeloid", "Endothelial": "Endothelial", "Stromal": "Stromal"}
FLS_COLOR = C["vermilion@s0.35"]
TR3 = {"ra_eur": "RA EUR", "fg_height": "Height", "fg_dupuytren": "Dupuytren"}
TR5 = {"ra_eur": "RA\nEUR", "fg_ra": "RA\nFinnGen", "fg_oa": "OA", "fg_height": "Height", "fg_dupuytren": "Dupuytren"}


def label(a):
    if a.startswith("iv_") and a.endswith("_acc"):
        return f"{COMP[a[3:-4]]}, accessible half", COMP_COLOR[COMP[a[3:-4]]]
    if a.startswith("iv_") and a.endswith("_ext"):
        return f"{COMP[a[3:-4]]}, specific +500 bp", COMP_COLOR[COMP[a[3:-4]]]
    if a in ("iv_fib_lining", "iv_fib_sublining", "iv_mural"):
        return {"iv_fib_lining": "Lining fibroblasts", "iv_fib_sublining": "Sublining fibroblasts",
                "iv_mural": "Mural cells"}[a], COMP_COLOR["Stromal"]
    if a.startswith("iv_"):
        return f"{COMP[a[3:]]}, specific", COMP_COLOR[COMP[a[3:]]]
    if a in ("rm_T_active", "rm_B_active"):
        return {"rm_T_active": "T-cell active regions", "rm_B_active": "B-cell active regions"}[a], \
            COMP_COLOR["T/NK" if a == "rm_T_active" else "B/plasma"]
    return {"fls_basal": "FLS ATAC, unstimulated", "fls_stim": "FLS ATAC, TNF", "fls_tnfgained": "FLS ATAC, TNF-gained",
            "fls_active": "FLS active regions"}.get(a, a), FLS_COLOR


def matrix(ax, D, rows, traits, letter, title):
    Z = D.pivot(index="annotation", columns="trait", values="coef_z").reindex(index=rows, columns=list(traits))
    F = D.pivot(index="annotation", columns="trait", values="coef_fdr").reindex(index=rows, columns=list(traits))
    E = D.pivot(index="annotation", columns="trait", values="enrichment").reindex(index=rows, columns=list(traits))
    norm = mpl.colors.TwoSlopeNorm(vmin=-ZCLIP, vcenter=0, vmax=ZCLIP)
    cmap = fk.cmap(kind="diverging")
    for i, r in enumerate(rows):
        for j, t in enumerate(traits):
            z = Z.loc[r, t]
            if np.isnan(z):
                continue
            ax.add_patch(mpl.patches.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84,
                                               color=cmap(norm(np.clip(z, -ZCLIP, ZCLIP))), lw=0))
            if F.loc[r, t] < 0.05:
                ax.add_patch(mpl.patches.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False, ec=C.ink, lw=0.8))
            ax.text(j, i, f"{E.loc[r, t]:.1f}", ha="center", va="center", fontsize=pt("fine"),
                    color=C.white if abs(z) > 3.5 else C.ink)
    ax.set_xlim(-0.55, len(traits) - 0.45)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(traits)), list(traits.values()))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1)
    labs = [label(r) for r in rows]
    ax.set_yticks(range(len(rows)), [lab for lab, _ in labs])
    for t, (_, col) in zip(ax.get_yticklabels(), labs):
        t.set_color(col)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fk.panel(ax, letter, title)
    return norm, cmap


def finemap(ax, F):
    rows = ["iv_TNK", "iv_Bplasma", "iv_Myeloid", "iv_Endothelial", "iv_Stromal",
            "iv_TNK_acc", "iv_Bplasma_acc", "iv_Myeloid_acc", "iv_Endothelial_acc", "iv_Stromal_acc",
            "fls_active", "rm_T_active", "rm_B_active"]
    d = F.set_index("annotation").reindex(rows).dropna(subset=["fold"])
    for i, (a, r) in enumerate(d.iterrows()):
        lab, col = label(a)
        ax.barh(i, r.fold, height=0.62, color=col, lw=0)
        ax.text(max(r.fold, 1) + 0.4, i, f"{100 * r.share_pip:.1f}% of PIP", va="center", ha="left", fontsize=pt("annot"))
    ax.axvline(1, color=C.grey_dark, lw=0.6, ls="--")
    ax.set_yticks(range(len(d)), [label(a)[0] for a in d.index])
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlim(0, max(d.fold) * 1.45)
    ax.set_xlabel("Fold (PIP share / SNP share)")
    ax.tick_params(axis="y", length=0)
    n = int(F.n_variants.iloc[0])
    fk.panel(ax, "d", f"Fine-mapped RA variants (PIP > 0.1; {n} in 1000G)")


def main():
    R2, R1, F = read("sldsc_chromatin_r2b"), read("sldsc_chromatin"), read("finemap_overlap")
    fig = figure("double", 150, kind="supp")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    acc = ["iv_TNK_acc", "iv_Bplasma_acc", "iv_Myeloid_acc", "iv_Endothelial_acc", "iv_Stromal_acc",
           "iv_fib_lining", "iv_fib_sublining", "iv_mural"]
    norm, cmap = matrix(fig.add_subplot(gs[0, 0]), R2[R2.model == "AllPeaks control"], acc, TR3, "a",
                        "Accessible chromatin and fibroblast sub-compartments")
    prim = ["iv_TNK", "iv_Bplasma", "iv_Myeloid", "iv_Endothelial", "iv_Stromal", "fls_basal", "fls_stim",
            "fls_tnfgained", "fls_active"]
    base = R2[R2.model == "baseline only"]
    prim += [a for a in acc[:5] if a in set(base.annotation)]       # accessible halves (RA, height): Gupta 2023 check
    matrix(fig.add_subplot(gs[0, 1]), R2[R2.model == "baseline only"], prim, TR3, "b",
           "Baseline model only (no open-chromatin control)")
    ext = ["iv_TNK_ext", "iv_Bplasma_ext", "iv_Myeloid_ext", "iv_Endothelial_ext", "iv_Stromal_ext"]
    axc = fig.add_subplot(gs[1, 0])
    matrix(axc, R1, ext, TR5, "c", "Compartment-specific peaks extended by 500 bp")
    cax = axc.inset_axes([0.1, -0.22, 0.8, 0.05])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[-6, -3, 0, 3, 6])
    cb.ax.tick_params(labelsize=pt("tick"), length=2)
    cb.set_label("Coefficient z; number = enrichment; box = FDR < 0.05", fontsize=pt("annot"))
    finemap(fig.add_subplot(gs[1, 1]), F)
    saved = save(fig, "FigS6_chromatin_sensitivity", provenance={
        "source": "db/ra_rx.duckdb::sldsc_chromatin_r2b, sldsc_chromatin, finemap_overlap",
        "scripts": ["scripts/27_sldsc_chromatin.py", "scripts/27b_sldsc_sensitivity.py", "scripts/figS6_chromatin_sensitivity.py"],
        "transformations": ["colour = coefficient z clipped at |z| = 6", "box = BH-FDR < 0.05 within model x trait",
                            "d: fold = PIP-weighted share / share of 1000G SNPs"]})
    print(saved.report)


if __name__ == "__main__":
    main()
