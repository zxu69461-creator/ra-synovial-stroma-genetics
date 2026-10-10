"""fig3_replication.py — Figure 3: RA genetic risk maps to lymphocyte states and to no stromal state, across methods,
atlases and GWAS; contrast traits (height, osteoarthritis) map to stromal states instead.

a  replication matrix (genetic_map_matrix): rows = analyses, grouped by atlas (discovery E-MTAB-11791, 54 states;
   replication GSE243917, 76 author-annotated states, RA only) and method (scDRS; MAGMA gene-property "cell-type";
   enrichment of likely causal RA genes, three gene sets); columns = compartments; dot area = fraction of the
   compartment's states significant (FDR < 0.05, enrichment direction), text = k/n states. Shaded rows: contrast
   traits (osteoarthritis, height and the fibroblast disease Dupuytren's contracture; FinnGen). RA GWAS: Ishigaki EUR, FinnGen, Ishigaki EAS (all MHC removed; EAS rows
   appear automatically once the EAS pipeline has written them).
b  replication atlas, scDRS z per state for RA (Ishigaki EUR, MHC removed), osteoarthritis and height, by compartment;
   filled = BH-FDR < 0.05 within trait over the 76 states; bar = compartment median.
c  causal-gene enrichment per compartment (mean expression-specificity z of the causal genes vs 10,000
   expression-matched random sets, z of that comparison), both atlases, three gene sets; band = |z| < 1.96;
   filled = one-sided P < 0.05.
d  MAGMA cell-type double dissociation: per state -log10 P (one-sided) for RA (Ishigaki EUR, MHC removed) vs height,
   both atlases (filled = discovery, open = replication).
e  cross-ancestry: discovery-atlas scDRS z, Ishigaki EUR vs EAS RA (both MHC removed), Spearman rho. Until the EAS
   tables exist: FinnGen vs Ishigaki EUR in the replication atlas.
Sources: genetic_map_matrix, scdrs_group_gupta_state, gupta_state_summary, scdrs_state_load, scdrs_group_eas_state,
causal_gene_state_enrichment, ra_causal_genes, magma_celltype.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read  # noqa: E402
from figstyle import fmt_p, C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402

TR = {"ra_eur_noMHC": "RA EUR", "fg_ra_noMHC": "RA FinnGen", "ra_eas_noMHC": "RA EAS", "fg_oa": "Osteoarthritis",
      "fg_height": "Height", "fg_dupuytren": "Dupuytren"}
CONTRAST = {"fg_oa": "osteoarthritis", "fg_height": "height", "fg_dupuytren": "Dupuytren"}   # stromal controls
RA_TRAITS = ["ra_eur_noMHC", "fg_ra_noMHC", "ra_eas_noMHC"]
GSETS = ["all", "no_blueprint_only", "coding_or_finemap"]
ATLAS = {"micheroli": ("Discovery", "E-MTAB-11791"), "gupta": ("Replication", "GSE243917, RA only")}
SMAX = 50          # marker area (pt^2) of a fraction of 1 in panel a


def have_tables():
    con = connect(read_only=True)
    try:
        return set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def gene_set_sizes():
    g = read("ra_causal_genes")
    ev = g.evidence.astype(str)
    return {"all": len(g), "no_blueprint_only": int((~g.blueprint_only.astype(bool)).sum()),
            "coding_or_finemap": int(ev.str.contains("coding|finemap").sum())}


def gs_label(gs, n):
    return {"all": f"All ({n})", "no_blueprint_only": f"No Blueprint-only ({n})",
            "coding_or_finemap": f"Coding/fine-map ({n})"}[gs]


def group_brackets(ax, groups, rotate=True):
    """Row-group labels left of the y tick labels: a thin bracket and a (rotated) label per group of rows.
    Offsets are in points from the axes edge, so they follow the axes through figkit's re-layout.
    Returns the x offset (inches) of the outer edge of the labels."""
    fig = ax.figure
    fig.canvas.draw()
    wlab = max(t.get_window_extent().width for t in ax.get_yticklabels()) * 72 / fig.dpi
    off = wlab + ax.yaxis.get_tick_padding() + 3
    base = blended_transform_factory(ax.transAxes, ax.transData)
    tr = base + ScaledTranslation(-(off + 2) / 72, 0, fig.dpi_scale_trans)
    lx = base + ScaledTranslation(-off / 72, 0, fig.dpi_scale_trans)
    texts = []
    for grp, yy in groups:
        texts.append(ax.text(0, (min(yy) + max(yy)) / 2, grp, transform=tr, ha="right", va="center",
                             rotation=90 if rotate else 0, fontsize=pt("tick"), color=C.grey_dark))
        ax.plot([0, 0], [min(yy) - 0.38, max(yy) + 0.38], transform=lx, color=C.grey, lw=0.5, clip_on=False)
    fig.canvas.draw()
    w = max(t.get_window_extent().width for t in texts) * 72 / fig.dpi
    return -(off + 2 + w) / 72


# ---------------------------------------------------------------------------------------------------- panel a
def matrix_layout(G, nset):
    """Row list for the matrix: ('head', atlas) or ('row', atlas, method, trait, label, group, contrast)."""
    have = set(zip(G.method, G.atlas, G.trait))
    rows = []
    for atlas in ATLAS:
        rows.append(("head", atlas))
        for method, mlab in [("scDRS", "scDRS"), ("MAGMA cell-type", "MAGMA")]:
            for t in RA_TRAITS:
                if (method, atlas, t) in have:
                    rows.append(("row", atlas, method, t, TR[t], mlab, False))
        for gs in GSETS:
            m = f"Causal genes ({gs})"
            if (m, atlas, "ra_causal") in have:
                rows.append(("row", atlas, m, "ra_causal", gs_label(gs, nset[gs]), "Causal genes", False))
        for method, mlab in [("scDRS", "scDRS"), ("MAGMA cell-type", "MAGMA")]:
            for t in CONTRAST:
                if (method, atlas, t) in have:
                    rows.append(("row", atlas, method, t, f"{mlab}, {CONTRAST[t]}", "Contrast", True))
    # y positions: a header takes 1.25 rows, method groups are separated by 0.35 rows
    y, ys, prev = 0.0, [], None
    for r in rows:
        if r[0] == "head":
            y += 0.55 if ys else 0.0
            ys.append(y)
            y += 1.15
            prev = None
            continue
        if prev is not None and r[5] != prev:
            y += 0.35
        ys.append(y)
        y += 1
        prev = r[5]
    return rows, np.array(ys), y


def panel_matrix(ax, G, nset):
    rows, ys, ymax = matrix_layout(G, nset)
    Gi = G.set_index(["method", "atlas", "trait", "compartment"])
    nc = len(COMP_ORDER)
    dx = 0.2                                # dot left of centre, k/n right of it
    XL = (-0.6, nc - 0.2)
    lab_y, lab_t = [], []
    groups = {}
    for r, y in zip(rows, ys):
        if r[0] == "head":
            continue
        _, atlas, method, trait, label, grp, contrast = r
        lab_y.append(y)
        lab_t.append(label)
        groups.setdefault((atlas, grp), []).append(y)
        if contrast:
            ax.add_patch(mpl.patches.Rectangle((XL[0], y - 0.5), XL[1] - XL[0], 1, color=C.grey_faint, lw=0,
                                               zorder=0))
        for j, comp in enumerate(COMP_ORDER):
            q = Gi.loc[(method, atlas, trait, comp)]
            f = float(q.frac_sig)
            if f > 0:
                ax.scatter(j - dx, y, s=SMAX * f, color=COMP_COLOR[comp], linewidths=0, zorder=3)
            else:
                ax.scatter(j - dx, y, s=5, facecolor="none", edgecolor=C.grey, linewidths=0.4, zorder=3)
            ax.text(j + 0.02, y, f"{int(q.n_sig)}/{int(q.n_states)}", ha="left", va="center",
                    fontsize=pt("fine"), color=C.ink if f > 0 else C.grey_dark)
    ax.set_yticks(lab_y, lab_t)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(*XL)
    ax.set_ylim(ymax + 1.6, -0.75)
    ax.xaxis.tick_top()
    ax.set_xticks(range(nc), COMP_ORDER, rotation=90)
    ax.tick_params(axis="x", length=0, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j, comp in enumerate(COMP_ORDER):                    # compartment key under each column name
        ax.scatter(j, -0.75, s=14, marker=COMP_MARKER[comp], color=COMP_COLOR[comp], linewidths=0, clip_on=False,
                   zorder=4)
    # method group labels: rotated, left of the row labels, with a bracket (offsets in points follow the layout)
    fig = ax.figure
    x_head = group_brackets(ax, [(g, yy) for (_, g), yy in groups.items()])
    base = blended_transform_factory(ax.transAxes, ax.transData)
    tr_head = base + ScaledTranslation(x_head, 0, fig.dpi_scale_trans)
    for r, y in zip(rows, ys):
        if r[0] != "head":
            continue
        name, acc = ATLAS[r[1]]
        n = int(G[G.atlas == r[1]].drop_duplicates("compartment").n_states.sum())
        ax.text(0, y - 0.05, f"{name} atlas  ({acc}; {n} states)", transform=tr_head, ha="left", va="center",
                fontsize=pt("base"), fontweight="bold")
    # size key, inside the axes below the last row
    yk = ymax + 0.9
    for j, f in enumerate([0, 0.25, 0.5, 1.0]):
        xx = j + 1
        if f:
            ax.scatter(xx - dx, yk, s=SMAX * f, color=C.grey_dark, linewidths=0, zorder=3)
        else:
            ax.scatter(xx - dx, yk, s=5, facecolor="none", edgecolor=C.grey, linewidths=0.4, zorder=3)
        ax.text(xx + 0.06, yk, f"{int(f * 100)}%", ha="left", va="center", fontsize=pt("fine"))
    ax.text(0.4, yk, "States with\nFDR < 0.05", ha="right", va="center", fontsize=pt("annot"), linespacing=1.0)
    fk.panel(ax, "a", "Replication across methods, atlases and GWAS")


# ---------------------------------------------------------------------------------------------------- panel b
def panel_strip(ax, S):
    rng = np.random.default_rng(1)
    show = [("ra_eur_noMHC", "RA EUR\n(MHC removed)"), ("fg_oa", "Osteoarthritis"), ("fg_height", "Height"),
            ("fg_dupuytren", "Dupuytren")]
    show = [s for s in show if s[0] in set(S.trait)]
    step = len(COMP_ORDER) + 1.2
    for k, (t, lab) in enumerate(show):
        d = S[S.trait == t]
        for j, comp in enumerate(COMP_ORDER):
            s = d[d.state_group == comp]
            x = k * step + j + rng.uniform(-0.22, 0.22, len(s))
            sig = (s.fdr < 0.05).to_numpy()
            col = COMP_COLOR[comp]
            ax.scatter(x[sig], s.assoc_mcz[sig], s=9, color=col, marker=COMP_MARKER[comp], edgecolor=C.white,
                       linewidth=0.2, zorder=3, label=comp if k == 0 else None)
            ax.scatter(x[~sig], s.assoc_mcz[~sig], s=8, facecolor=C.white, marker=COMP_MARKER[comp], edgecolor=col,
                       linewidth=0.5, zorder=3)
            ax.plot([k * step + j - 0.32, k * step + j + 0.32], [s.assoc_mcz.median()] * 2, color=C.ink, lw=0.9,
                    zorder=4)
    ax.axhline(0, color=C.grey, lw=0.5)
    ax.set_xticks([k * step + 2 for k in range(len(show))], [s[1] for s in show])
    ax.set_xlim(-0.8, (len(show) - 1) * step + len(COMP_ORDER) - 0.2)
    ax.set_ylabel("scDRS z per state")
    ax.tick_params(axis="x", length=0)
    n = S.state.nunique()
    ax.text(0.99, 0.98, f"{n} states; filled: FDR < 0.05", transform=ax.transAxes, ha="right", va="top",
            fontsize=pt("annot"))
    fk.legend_top(ax)
    fk.panel(ax, "b", "Replication atlas (scDRS): lymphoid RA load; stromal OA, height and Dupuytren load")


# ---------------------------------------------------------------------------------------------------- panel c
def panel_causal(ax, E, nset):
    comp_rows = E[E.state.str.startswith("[compartment] ")].copy()
    comp_rows["compartment"] = comp_rows.state.str.replace("[compartment] ", "", regex=False)
    ax.axvspan(-1.96, 1.96, color=C.grey_faint, lw=0, zorder=0)
    y, ys, labs, grp = 0.0, [], [], {}
    for atlas in ATLAS:
        for gs in GSETS:
            d = comp_rows[(comp_rows.atlas == atlas) & (comp_rows.gene_set == gs)].set_index("compartment")
            if d.empty:
                continue
            for comp in COMP_ORDER:
                r = d.loc[comp]
                col = COMP_COLOR[comp]
                filled = r.p < 0.05
                ax.scatter(r.enrich_z, y, s=20, marker=COMP_MARKER[comp], facecolor=col if filled else C.white,
                           edgecolor=C.white if filled else col, linewidth=0.3 if filled else 0.7, zorder=3)
            ys.append(y)
            labs.append(gs_label(gs, nset[gs]))
            grp.setdefault(atlas, []).append(y)
            y += 1
        y += 0.5
    ax.set_yticks(ys, labs)
    ax.set_ylim(y - 0.4, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("Compartment enrichment z")
    group_brackets(ax, [(ATLAS[a][0], yy) for a, yy in grp.items()])
    ax.spines["left"].set_visible(False)
    ax.text(0.5, 0.02, "|z| < 1.96", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=pt("annot"), color=C.grey_dark)
    fk.panel(ax, "c", f"{nset['all']} likely causal RA genes: lymphoid, not stromal")


# ---------------------------------------------------------------------------------------------------- panel d
def panel_magma(ax, M):
    w = M.pivot_table(index=["atlas", "state", "state_group"], columns="trait", values=["p", "fdr"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    counts = {}
    for atlas, filled in [("gupta", False), ("micheroli", True)]:
        d = w[w.atlas == atlas]
        for comp in COMP_ORDER:
            s = d[d.state_group == comp]
            col = COMP_COLOR[comp]
            ax.scatter(-np.log10(s.p_ra_eur_noMHC), -np.log10(s.p_fg_height), s=12, marker=COMP_MARKER[comp],
                       facecolor=col if filled else C.white, edgecolor=C.white if filled else col,
                       linewidth=0.25 if filled else 0.6, zorder=3 if filled else 2)
        ra, h = d.fdr_ra_eur_noMHC < 0.05, d.fdr_fg_height < 0.05
        counts[atlas] = (int((ra & ~h).sum()), int((h & ~ra).sum()), int((ra & h).sum()))
    ax.set_xlabel("RA EUR, −log10 P")
    ax.set_ylabel("Height, −log10 P")
    ax.set_xlim(-0.3, None)
    ax.set_ylim(-0.4, None)
    cd, cr = counts["micheroli"], counts["gupta"]
    ax.text(0.97, 0.97, "FDR < 0.05 (disc. / rep.)\n"
            f"RA only: {cd[0]} / {cr[0]}\nheight only: {cd[1]} / {cr[1]}\nboth: {cd[2]} / {cr[2]}",
            transform=ax.transAxes, ha="right", va="top", fontsize=pt("annot"))
    hs = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=C.grey_dark, mec=C.grey_dark, label="Discovery"),
          Line2D([], [], ls="", marker="o", ms=3.6, mfc=C.white, mec=C.grey_dark, mew=0.6, label="Replication")]
    ax.legend(handles=hs, loc="center right", frameon=False, fontsize=pt("annot"), handletextpad=0.1,
              borderaxespad=0.2, labelspacing=0.3)
    fk.panel(ax, "d", "MAGMA cell-type: RA vs height")
    return counts


# ---------------------------------------------------------------------------------------------------- panel e
def panel_ancestry(ax, w, x, y, xlabel, ylabel, title):
    for comp in COMP_ORDER:
        s = w[w.state_group == comp]
        ax.scatter(s[x], s[y], s=12, color=COMP_COLOR[comp], marker=COMP_MARKER[comp], edgecolor=C.white,
                   linewidth=0.3, zorder=3)
    ax.axhline(0, color=C.grey, lw=0.5, zorder=1)
    ax.axvline(0, color=C.grey, lw=0.5, zorder=1)
    r, p = stats.spearmanr(w[x], w[y])
    ax.text(0.97, 0.03, f"ρ = {r:.2f}\nP = {fmt_p(p)}\n{len(w)} states", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=pt("annot"))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fk.panel(ax, "e", title)
    return r, p


def main():
    have = have_tables()
    G = read("genetic_map_matrix")
    nset = gene_set_sizes()
    grp_rep = read("gupta_state_summary")
    grp_rep = grp_rep.assign(state=grp_rep.state.astype(str), state_group=grp_rep.state_group.astype(str))
    grp_rep = grp_rep.set_index("state").state_group
    S = read("scdrs_group_gupta_state")
    if "scdrs_group_gupta_dup_state" in have:      # Dupuytren positive control tagged table
        S = pd.concat([S, read("scdrs_group_gupta_dup_state")], ignore_index=True)
    S["state_group"] = S.state.map(grp_rep)
    S["fdr"] = S.groupby("trait").assoc_mcp.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    E = read("causal_gene_state_enrichment")
    M = read("magma_celltype")
    M = M[M.trait.isin(["ra_eur_noMHC", "fg_height"])]
    eas = "scdrs_group_eas_state" in have and \
        "ra_eas_noMHC" in set(read("SELECT DISTINCT trait FROM scdrs_group_eas_state").trait)

    fig = figure("double", 205)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.95], height_ratios=[1.05, 0.8, 1])
    bottom = gs[2, 1].subgridspec(1, 2)
    panel_matrix(fig.add_subplot(gs[:, 0]), G, nset)
    panel_strip(fig.add_subplot(gs[0, 1]), S)
    panel_causal(fig.add_subplot(gs[1, 1]), E, nset)
    counts = panel_magma(fig.add_subplot(bottom[0]), M)
    axe = fig.add_subplot(bottom[1])
    if eas:
        L = read("scdrs_state_load")
        L = L[L.trait == "ra_eur_noMHC"][["state", "state_group", "assoc_mcz"]].rename(columns={"assoc_mcz": "eur"})
        X = read("scdrs_group_eas_state")
        X = X[X.trait == "ra_eas_noMHC"][["state", "assoc_mcz"]].rename(columns={"assoc_mcz": "eas"})
        w = L.merge(X, on="state")
        w["state_group"] = w.state_group.astype(str)
        rho = panel_ancestry(axe, w, "eur", "eas", "scDRS z, RA EUR", "scDRS z, RA EAS",
                             "Cross-ancestry (discovery)")
        e_src = "scdrs_state_load + scdrs_group_eas_state (discovery atlas)"
    else:
        w = S.pivot_table(index=["state", "state_group"], columns="trait", values="assoc_mcz").reset_index()
        rho = panel_ancestry(axe, w, "ra_eur_noMHC", "fg_ra_noMHC", "scDRS z, RA EUR (Ishigaki)",
                             "scDRS z, RA FinnGen", "Two European GWAS (replication)")
        e_src = "scdrs_group_gupta_state (replication atlas; EAS not yet available)"

    if "--draft" in sys.argv:
        print(fk.save(fig, Path(sys.argv[-1]), strict=False).report)
        return
    saved = save(fig, "Fig3_replication", provenance={
        "source": "db/ra_rx.duckdb::genetic_map_matrix, scdrs_group_gupta_state, gupta_state_summary, "
                  "causal_gene_state_enrichment, ra_causal_genes, magma_celltype; e: " + e_src,
        "scripts": ["scripts/21_scdrs.py", "scripts/23_magma_celltype.py", "scripts/24_causal_genes.py",
                    "scripts/25_genetic_map_summary.py", "scripts/fig3_replication.py"],
        "transformations": ["a: dot area proportional to fraction of compartment states with FDR < 0.05",
                            "b: BH-FDR within trait over replication-atlas states; strip x-jitter seed 1",
                            "c: compartment enrichment z, filled = one-sided P < 0.05",
                            "d: MAGMA one-sided P; counts use BH-FDR within atlas x trait",
                            "e: Spearman rho over states"],
        "eas_included": bool(eas), "panel_d_counts": counts, "panel_e_rho": [float(rho[0]), float(rho[1])]})
    print(saved.report)
    print("EAS included:", eas, "| d counts (RA only, height only, both):", counts, "| e rho, P:", rho)


if __name__ == "__main__":
    main()
