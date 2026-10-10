"""figS3_replication_atlas.py — Supplementary Figure 3: state-level genetic maps in the independent RA-only
replication atlas (GSE243917, 76 author-annotated states, 11 RA donors).

a  scDRS association z (76 states x GWAS), states grouped by compartment (colour strip) and ordered by RA z
   (Ishigaki EUR, MHC removed) within compartment; dot = BH-FDR < 0.05 within trait over the 76 states. Colour scale
   clipped at |z| = 8 (exact values in scdrs_group_gupta_state). Traits: RA (Ishigaki EUR, FinnGen, Ishigaki EAS when
   available; all MHC removed); contrast traits osteoarthritis and height (FinnGen).
b  MAGMA gene-property ("cell-type") analysis for the same states and traits: signed -log10 P
   (sign of beta; P of the one-sided test in that direction, from beta/SE); dot = FDR < 0.05 of the one-sided
   enrichment test (BH within trait); clipped at |6|.
c  cells per state (log scale), to read the small states (e.g. 2-7 cells) with care.
Sources: scdrs_group_gupta_state, scdrs_group_gupta_eas_state, magma_celltype, gupta_state_summary.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.transforms import ScaledTranslation
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_ORDER, figure, fk, pt, save  # noqa: E402

TRAITS = ["ra_eur_noMHC", "fg_ra_noMHC", "ra_eas_noMHC", "fg_oa", "fg_height"]
COL = {"ra_eur_noMHC": "RA EUR", "fg_ra_noMHC": "RA FinnGen", "ra_eas_noMHC": "RA EAS", "fg_oa": "OA",
       "fg_height": "Height"}
ZCLIP, PCLIP = 8, 6


def have_tables():
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def heat(ax, Z, F, traits, clip, label, ticks, order, grp, letter, title, labels=True):
    norm = mpl.colors.TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
    n = len(order)
    im = ax.imshow(np.clip(Z, -clip, clip), aspect="auto", cmap=fk.cmap(kind="diverging"), norm=norm,
                   interpolation="nearest", extent=(-0.5, len(traits) - 0.5, n - 0.5, -0.5))
    yy, xx = np.where(F < 0.05)
    ax.scatter(xx, yy, s=1.6, c=C.ink, marker="o", linewidths=0)
    for comp in COMP_ORDER:
        idx = np.where(grp == comp)[0]
        if labels:
            ax.add_patch(mpl.patches.Rectangle((-1.05, idx.min() - 0.5), 0.4, len(idx), color=COMP_COLOR[comp],
                                               lw=0))
        if idx.max() < n - 1:
            ax.axhline(idx.max() + 0.5, color=C.white, lw=1.0)
    nra = sum(t.startswith(("ra_", "fg_ra")) for t in traits)
    ax.axvline(nra - 0.5, color=C.ink, lw=0.7)
    ax.set_xlim(-1.1 if labels else -0.5, len(traits) - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(traits)), [COL[t] for t in traits], rotation=90)
    if labels:
        ax.set_yticks(range(n), order, fontsize=pt("fine"))
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([0.08, -0.03, 0.84, 0.008])
    cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels([f"≤ −{clip}" if t == -clip else f"≥ {clip}" if t == clip else f"{t:g}".replace("-", "−")
                           for t in ticks])
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=pt("base"))
    fk.panel(ax, letter, title)


def panel_cells(ax, cells, grp):
    y = np.arange(len(cells))
    ax.barh(y, cells, color=[COMP_COLOR[g] for g in grp], height=0.72, lw=0)
    ax.set_xscale("log")
    top = 10 ** np.ceil(np.log10(cells.max()))
    ax.set_xlim(1, top)
    ticks = [10 ** k for k in range(int(np.log10(top)) + 1)]
    ax.set_xticks(ticks, [f"{t:,}" if k % 2 == 0 else "" for k, t in enumerate(ticks)])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(len(cells) - 0.5, -0.5)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Nuclei per state")
    fk.panel(ax, "c", "State size")


def main():
    have = have_tables()
    meta = read("gupta_state_summary")
    meta = meta.assign(state=meta.state.astype(str), state_group=meta.state_group.astype(str))
    S = read("scdrs_group_gupta_state")
    if "scdrs_group_gupta_eas_state" in have:
        S = S._append(read("scdrs_group_gupta_eas_state"), ignore_index=True)
    S = S[S.trait.isin(TRAITS)].drop_duplicates(["trait", "state"], keep="last")
    S["fdr"] = S.groupby("trait").assoc_mcp.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    M = read("magma_celltype")
    M = M[(M.atlas == "gupta") & M.trait.isin(TRAITS)].copy()
    z = M.beta / M.se
    M["slp"] = np.sign(z) * -np.log10(stats.norm.sf(np.abs(z)))
    traits = [t for t in TRAITS if t in set(S.trait)]
    mtraits = [t for t in traits if t in set(M.trait)]
    eas = "ra_eas_noMHC" in traits

    zw = S.pivot_table(index="state", columns="trait", values="assoc_mcz")[traits]
    fw = S.pivot_table(index="state", columns="trait", values="fdr")[traits]
    mw = M.pivot_table(index="state", columns="trait", values="slp").reindex(columns=traits)
    mf = M.pivot_table(index="state", columns="trait", values="fdr").reindex(columns=traits)
    d = meta.set_index("state").join(zw)
    d["_c"] = d.state_group.map({c: i for i, c in enumerate(COMP_ORDER)})
    d = d.sort_values(["_c", "ra_eur_noMHC"], ascending=[True, False])
    order = d.index.tolist()
    grp = d.state_group.to_numpy()

    fig = figure("mid", 225, kind="supp")
    gs = fig.add_gridspec(1, 3, width_ratios=[len(traits) + 0.6, len(traits), 3])
    axa = fig.add_subplot(gs[0])
    heat(axa, zw.loc[order].to_numpy(), fw.loc[order].to_numpy(), traits, ZCLIP, "scDRS z (dot: FDR < 0.05)",
         [-8, -4, 0, 4, 8], order, grp, "a", "scDRS")
    axb = fig.add_subplot(gs[1], sharey=axa)
    heat(axb, mw.loc[order].to_numpy(), mf.loc[order].fillna(1).to_numpy(), traits, PCLIP,
         "MAGMA signed −log10 P\n(dot: FDR < 0.05)", [-6, -3, 0, 3, 6], order, grp, "b", "MAGMA cell-type",
         labels=False)
    axc = fig.add_subplot(gs[2], sharey=axa)
    panel_cells(axc, d.n_cells.to_numpy(), grp)

    if "--draft" in sys.argv:
        print(fk.save(fig, Path(sys.argv[-1]), strict=False).report)
        return
    saved = save(fig, "FigS3_replication_atlas", sub="", provenance={
        "source": "db/ra_rx.duckdb::scdrs_group_gupta_state, scdrs_group_gupta_eas_state (if present), "
                  "magma_celltype (atlas = gupta), gupta_state_summary",
        "scripts": ["scripts/14_atlas_gupta.py", "scripts/21_scdrs.py", "scripts/23_magma_celltype.py",
                    "scripts/figS3_replication_atlas.py"],
        "transformations": ["scDRS: BH-FDR within trait over 76 states; colour clipped at |z| = 8",
                            "MAGMA: signed -log10 P = sign(beta) x -log10 one-sided normal P of |beta/SE|; "
                            "clipped at |6|; dots from magma_celltype.fdr (one-sided enrichment, BH within trait)",
                            "states ordered by compartment, then RA EUR (MHC removed) scDRS z"],
        "eas_included": bool(eas), "traits": traits, "magma_traits": mtraits})
    print(saved.report)
    print("EAS included:", eas, "| traits:", traits, "| MAGMA traits:", mtraits)


if __name__ == "__main__":
    main()
