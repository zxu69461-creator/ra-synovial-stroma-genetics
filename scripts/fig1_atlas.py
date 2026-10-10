"""fig1_atlas.py — Figure 1: study design (genetic blind spot) and the open RA synovial atlas with AMP2-named states.

a  study design, drawn with fk.art from real data, left to right. Two evidence streams: genetics (Ishigaki 2022 EUR,
   22,350 cases; FinnGen R13 RA; Ishigaki 2022 EAS, 11,025 cases / 162,608 controls; stromal control traits
   osteoarthritis, height and Dupuytren's contracture) -> gene level (Manhattan of the real MAGMA gene P values, RA EUR MHC removed, Bonferroni line; the
   154 likely causal genes of ra_causal_genes, six named) and synovial atlases (discovery E-MTAB-11791: 25
   inflammatory-arthritis biopsies, 15 RA, 83,503 cells, 54 states; RA-only replication GSE243917: 11 RA donors,
   27,856 nuclei, 76 author-annotated states; one dot per state, coloured by compartment). They converge on the
   genetic map: per compartment, the share of states with RA-risk enrichment at FDR < 0.05 (ra_eur_noMHC) by scDRS,
   MAGMA cell-type gene-property analysis and causal-gene enrichment, in both atlases (dot area = share; ring = none),
   plus one chromatin column: S-LDSC heritability enrichment of compartment-specific snATAC peaks of
   the replication nuclei (sldsc_chromatin; full dot = coefficient FDR < 0.05).
   Two consequences: inflammation (R4RA, STRAP, PEAC, GSE198520, GSE45867; n = 492 baseline biopsies; filled dot =
   compartment score vs DAS28 / CRP / ESR, pooled rho > 0 and P < 0.05) and drug targets (Open Targets, 8,420 RA
   targets; ChEMBL approved targeted-DMARD targets) -> "Stroma: a genetic blind spot". No GLI, no response readout,
   no spatial element (Xenium is supplementary). Art 170 x 61 mm: at 64 mm, panel d's 54 rows drop below the
   6.6-pt category pitch (figkit title pad 2 mm since 2026-09-30) and strict QA fails.
b  global UMAP (scVI) of 83,503 curated cells coloured by compartment
c  compartment composition of each of the 25 samples (ordered by stromal share)
d  top signature gene per state (54 rows; state numbers index e): expression in the state itself and, averaged
   over states, in the other states of each compartment; dot size = % cells detected, colour = per-gene min-max
   scaled mean (magma, light = low); full top-2 x 54-state version: Fig. S1
e  per-compartment UMAPs (lineage-specific scVI) coloured by state, numbered as in d; key = compartment sizes
Sources: atlas_cells_lineage, atlas_cells_fine, atlas_cells_state, state_signatures, state_pseudobulk_long,
magma_genes_ra_eur_noMHC, ra_causal_genes, scdrs_state_load, scdrs_group_gupta_state, magma_celltype,
causal_gene_state_enrichment, gupta_state_summary, activity_assoc_pooled.
Usage: python scripts/fig1_atlas.py            (strict fk.save -> figures/Fig1_atlas + figure_registry)
       python scripts/fig1_atlas.py --art PNG  (schematic preview + A.problems())
       python scripts/fig1_atlas.py --draft BASE  (non-strict exploration render, not archived)
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402

TRAIT = "ra_eur_noMHC"                      # primary GWAS for the genetic map (Ishigaki 2022 EUR, MHC removed)
STATE_COLORS = ["blue", "vermilion", "green", "purple", "sky", "amber", "indigo", "cyan", "teal", "olive", "sand",
                "rose", "wine", "magenta", "brown"]          # within-compartment state colours (numbers are the key)
ART_W, ART_H = 170, 61
CAUSAL_SHOWN = ["PTPN22", "CTLA4", "CCR6", "IL6R", "PADI4", "TYK2"]   # members of ra_causal_genes, shown by name
MAP_COLS = [("scDRS", "micheroli"), ("MAGMA", "micheroli"), ("Causal", "micheroli"),
            ("scDRS", "gupta"), ("MAGMA", "gupta"), ("Causal", "gupta"),
            ("ATAC", "gupta")]   # genetic map: method x atlas; ATAC = S-LDSC on compartment-specific snATAC (same nuclei)


# ------------------------------------------------------------------------------------------------ data
def atlas():
    cs = read("atlas_cells_state")
    cs["state_group"], cs["state"] = cs.state_group.astype(str), cs.state.astype(str)
    lin = read("SELECT cell, umap1, umap2 FROM atlas_cells_lineage")
    fine = read("SELECT cell, sub_umap1, sub_umap2 FROM atlas_cells_fine")
    d = cs.merge(lin, on="cell").merge(fine, on="cell")
    states = (d.groupby(["state_group", "state"]).size().rename("n").reset_index()
              .assign(_c=lambda x: x.state_group.map({c: i for i, c in enumerate(COMP_ORDER)}))
              .sort_values(["_c", "n"], ascending=[True, False]).reset_index(drop=True))
    states["num"] = np.arange(1, len(states) + 1)
    return d, states


def genetic_map():
    """Share of each compartment's states enriched for RA risk (FDR < 0.05, enrichment direction) per method x atlas;
    the definitions of scripts/25_genetic_map_summary.py (scDRS BH over the states of an atlas within trait; MAGMA
    cell-type FDR within atlas x trait; causal-gene state FDR, compartment pseudo-rows excluded)."""
    from statsmodels.stats.multitest import multipletests
    parts = []
    s = read(f"SELECT state, state_group, assoc_mcz, fdr FROM scdrs_state_load WHERE trait = '{TRAIT}'")
    parts.append(s.assign(sig=(s.fdr < 0.05) & (s.assoc_mcz > 0), method="scDRS", atlas="micheroli"))
    grp = read("gupta_state_summary").set_index("state").state_group.astype(str)
    g = read(f"SELECT state, assoc_mcz, assoc_mcp FROM scdrs_group_gupta_state WHERE trait = '{TRAIT}'")
    g["fdr"] = multipletests(g.assoc_mcp, method="fdr_bh")[1]
    parts.append(g.assign(state_group=g.state.map(grp), sig=(g.fdr < 0.05) & (g.assoc_mcz > 0), method="scDRS",
                          atlas="gupta"))
    m = read(f"SELECT atlas, state, state_group, fdr FROM magma_celltype WHERE trait = '{TRAIT}'")
    parts.append(m.assign(sig=m.fdr < 0.05, method="MAGMA"))
    c = read("SELECT atlas, state, state_group, fdr FROM causal_gene_state_enrichment WHERE gene_set = 'all'")
    c = c[~c.state.str.startswith("[")]                                   # drop the compartment pseudo-rows
    parts.append(c.assign(sig=c.fdr < 0.05, method="Causal"))
    # chromatin: one test per compartment, S-LDSC coefficient FDR < 0.05 (RA Ishigaki EUR)
    iv = {"iv_TNK": "T/NK", "iv_Bplasma": "B/plasma", "iv_Myeloid": "Myeloid", "iv_Endothelial": "Endothelial",
          "iv_Stromal": "Stromal"}
    h = read("SELECT annotation, coef_fdr FROM sldsc_chromatin WHERE trait = 'ra_eur'")
    h = h[h.annotation.isin(iv)]
    parts.append(pd.DataFrame(dict(state=h.annotation, state_group=h.annotation.map(iv), sig=h.coef_fdr < 0.05,
                                   method="ATAC", atlas="gupta")))
    R = pd.concat(parts, ignore_index=True)
    R["state_group"] = R.state_group.astype(str)
    out = R.groupby(["method", "atlas", "state_group"]).sig.agg(n_sig="sum", n="size")
    return out.assign(frac=out.n_sig / out.n)


# ------------------------------------------------------------------------------------------------ panel a (art)
def state_dots(A, x, y, groups, ncol=10, pitch=1.22, r=0.43):
    """One dot per cell state, coloured by compartment, filled row by row in compartment order."""
    order = sorted(groups, key=COMP_ORDER.index)
    for k, g in enumerate(order):
        A.circle(x + r + (k % ncol) * pitch, y + r + (k // ncol) * pitch, r, fill=COMP_COLOR[g])
    return y + 2 * r + (len(order) - 1) // ncol * pitch


def schematic(states):
    A = fk.art.Art(ART_W, ART_H)
    fine, leg, base, annot = "fine", "legend", "base", "annot"
    rng = np.random.default_rng(1)
    LX, LW = 1.0, 49.5                          # left column: the two evidence streams
    MX, MW = 56.0, 57.5                         # centre: the genetic map
    RX, RW = 119.0, 50.0                        # right column: the two consequences
    TOP, BOT, CH = 1.0, 32.0, 28.0              # top / bottom card y and card height (left and right columns)
    YC = (TOP + BOT + CH) / 2                   # vertical centre line
    ring = dict(fill="white", stroke="grey", sw=A.lw * .6)
    dy = 2.25                                   # line pitch of 5-pt text

    # backbone: two streams converge on the genetic map, which fans out to the two consequences
    A.ribbon([(10, YC), (40, YC), (78, YC)], 60, 44, c0="grey_light", c1="grey_faint", opacity=0.5)
    A.ribbon([(92, YC), (130, YC), (160, YC)], 44, 60, c0="grey_faint", c1="grey_light", opacity=0.5)

    # ---------------- stream 1: genetics (three GWAS -> gene level)
    A.card(LX, TOP, LW, CH)
    tx = LX + 2.2
    A.text(tx, TOP + 2.9, "Genetics", role=leg, bold=True)
    gwas = [("Ishigaki 2022 EUR", "22,350 cases"), ("FinnGen R13", "RA"),
            ("Ishigaki 2022 EAS", "11,025 cases / 162,608 controls"), ("Stromal controls", "OA, height, Dupuytren")]
    for k, (name, n) in enumerate(gwas):
        col = "grey_dark" if k == 3 else "ink"
        A.text(tx, TOP + 6.2 + k * dy, name, role=fine, color=col)
        A.text(tx + 16.2, TOP + 6.2 + k * dy, n, role=fine, color=col)
    ysep = TOP + 6.2 + 3 * dy + 1.8
    A.line(tx, ysep, LX + LW - 2.2, ysep, color="grey_light", sw=A.lw * .6)
    yg = ysep + 2.2                             # gene-level row
    # gene level, left: Manhattan of the real MAGMA gene P values (RA EUR, MHC removed)
    A.text(tx, yg, "Gene level (MAGMA)", role=fine, bold=True)
    m = read(f"SELECT CHR, START, P FROM magma_genes_{TRAIT}")
    m["CHR"] = pd.to_numeric(m.CHR, errors="coerce")
    m = m.dropna().sort_values(["CHR", "START"])
    off = m.groupby("CHR").START.max().cumsum().shift(fill_value=0)
    m["pos"] = m.START + m.CHR.map(off)
    m["lp"] = -np.log10(m.P)
    thr = -np.log10(0.05 / len(m))                                      # Bonferroni over all tested genes
    m = m[(m.lp > 2) | (rng.random(len(m)) < 0.10)]
    mx, my0, mw, mh, lpmax = tx + 0.3, yg + 1.8, 20.5, 5.9, 16.5
    for chrom, g in m.groupby("CHR"):
        col = "grey_dark" if int(chrom) % 2 else "grey"
        for p, lp in zip(g.pos, g.lp):
            A.circle(mx + p / m.pos.max() * mw, my0 + mh - min(lp, lpmax) / lpmax * mh, 0.15, fill=col)
    yt = my0 + mh - thr / lpmax * mh
    A.line(mx, yt, mx + mw, yt, color="vermilion", sw=A.lw * .7, dash="0.8,0.5")
    A.line(mx, my0 + mh + .25, mx + mw, my0 + mh + .25, color="ink", sw=A.lw * .6)
    # gene level, right: the likely causal genes (Ishigaki 2022)
    cg = read("SELECT gene FROM ra_causal_genes").gene
    shown = [g for g in CAUSAL_SHOWN if g in set(cg)]
    cx = tx + 24.0
    A.text(cx, yg, f"{len(cg)} likely causal genes", role=fine, bold=True)
    A.text(cx, yg + 2.4, " · ".join(shown[:3]), role=fine, italic=True)
    A.text(cx, yg + 2.4 + dy, " · ".join(shown[3:]) + " …", role=fine, italic=True)
    A.text(cx, yg + 2.4 + 2 * dy, "fine-mapped or colocalised", role=fine, color="grey_dark")

    # ---------------- stream 2: atlases (discovery + RA-only replication)
    A.card(LX, BOT, LW, CH)
    A.text(tx, BOT + 2.9, "Synovial atlases", role=leg, bold=True)
    gup = read("gupta_state_summary")
    rows = [(states.state_group.tolist(), BOT + 5.4, "Discovery · E-MTAB-11791",
             ["25 inflammatory-arthritis biopsies", f"15 RA · 83,503 cells · {len(states)} states"]),
            (gup.state_group.astype(str).tolist(), BOT + 15.4, "Replication · GSE243917",
             ["RA only: 11 RA donors · 27,856 nuclei", f"{len(gup)} author-annotated states"])]
    for groups, y0, head, lines in rows:
        state_dots(A, tx + 0.2, y0, groups)
        A.text(tx + 14.6, y0 + 0.9, head, role=fine, bold=True)
        for k, s in enumerate(lines):
            A.text(tx + 14.6, y0 + 0.9 + (k + 1) * dy, s, role=fine)
    A.text(tx + 14.6, BOT + 15.4 + 0.9 + 3 * dy + 0.5, "1 dot = 1 cell state", role=fine, color="grey_dark")

    # streams converge: genetics x atlases
    jx = (LX + LW + MX) / 2
    A.arrow(LX + LW + 0.5, YC - 15.0, MX - 0.7, YC - 8.5, width=1.5)
    A.arrow(LX + LW + 0.5, YC + 15.0, MX - 0.7, YC + 8.5, width=1.5)
    A.circle(jx, YC, 1.9, fill="white", stroke="grey", sw=A.lw * .7)
    A.text(jx, YC + 0.05, "×", role=base, anchor="middle", color="grey_dark", bold=True)

    # ---------------- centre: genetic map (compartment x method x atlas, real)
    G = genetic_map()
    MH = 49.5
    Y0 = YC - MH / 2
    A.card(MX, Y0, MW, MH, stroke="grey")
    cx0 = MX + 2.2
    A.text(cx0, Y0 + 3.1, "Genetic map", role=leg, bold=True)
    A.text(cx0, Y0 + 5.8, "share of each compartment's states with RA risk enrichment", role=fine, color="grey_dark")
    A.text(cx0, Y0 + 5.8 + dy, "(FDR < 0.05; Ishigaki EUR; ATAC = chromatin heritability)", role=fine, color="grey_dark")
    lab_r = cx0 + 9.4                                   # right edge of the compartment names
    pitch, gap = 6.4, 1.0
    xs = [lab_r + 3.8 + k * pitch + (gap if k >= 3 else 0) for k in range(len(MAP_COLS))]
    for k0, k1, name in ((0, 2, "Discovery"), (3, 6, "Replication")):
        A.text((xs[k0] + xs[k1]) / 2, Y0 + 11.9, name, role=fine, bold=True, anchor="middle")
        A.line(xs[k0] - 2.8, Y0 + 13.2, xs[k1] + 2.8, Y0 + 13.2, color="grey", sw=A.lw * .6)
    for x, (meth, _) in zip(xs, MAP_COLS):
        A.text(x, Y0 + 15.3, meth, role=fine, anchor="middle")
    RMAX, y_row, rp = 1.6, Y0 + 19.3, 3.9
    for i, comp in enumerate(COMP_ORDER):
        y = y_row + i * rp
        A.text(lab_r, y, comp, role=fine, anchor="end", color=COMP_COLOR[comp], bold=True)
        for x, (meth, atl) in zip(xs, MAP_COLS):
            f = float(G.loc[(meth, atl, comp), "frac"])
            if f > 0:
                A.circle(x, y, RMAX * np.sqrt(f), fill=COMP_COLOR[comp])
            else:
                A.circle(x, y, 0.75, **ring)
    ky = y_row + 4 * rp + 4.8                           # size key
    A.text(cx0, ky, "states enriched:", role=fine, color="grey_dark")
    kx = cx0 + 15.5
    for f, s in ((0, "none"), (0.5, "half"), (1.0, "all")):
        if f:
            A.circle(kx, ky, RMAX * np.sqrt(f), fill="grey_dark")
        else:
            A.circle(kx, ky, 0.75, **ring)
        A.text(kx + 2.2, ky, s, role=fine, color="grey_dark")
        kx += 9.0
    A.text(MX + MW / 2, ky + 5.0, "Lymphocytes loaded, stroma silent", role=base, anchor="middle")

    # genetic map -> two consequences
    A.arrow(MX + MW + 0.5, YC - 8.5, RX - 0.7, YC - 15.0, width=1.5)
    A.arrow(MX + MW + 0.5, YC + 8.5, RX - 0.7, YC + 15.0, width=1.5)

    # ---------------- consequence 1: inflammation (five bulk cohorts; compartment scores vs activity, real)
    A.card(RX, TOP, RW, CH)
    rx = RX + 2.2
    A.text(rx, TOP + 2.9, "Inflammation", role=leg, bold=True)
    for k in range(5):
        A.cylinder(rx + k * 5.0, TOP + 5.0, 3.8, 4.3, color="grey_dark")
    for k, s in enumerate(("R4RA · STRAP · PEAC", "GSE198520 · GSE45867", "n = 492 baseline biopsies")):
        A.text(rx, TOP + 11.7 + k * dy, s, role=fine)
    act = read("SELECT variable, clinical, rho, p FROM activity_assoc_pooled "
               "WHERE clinical IN ('das28', 'crp', 'esr')").set_index(["variable", "clinical"])
    gx = [RX + RW - 4.4 - k * 6.0 for k in range(3)][::-1]
    for x, (cl, lab) in zip(gx, (("das28", "DAS28"), ("crp", "CRP"), ("esr", "ESR"))):
        A.text(x, TOP + 5.3, lab, role=fine, anchor="middle")
        for i, comp in enumerate(COMP_ORDER):
            r_ = act.loc[(comp, cl)]
            y = TOP + 7.8 + i * 2.15
            if r_.rho > 0 and r_.p < 0.05:
                A.circle(x, y, 0.8, fill=COMP_COLOR[comp])
            else:
                A.circle(x, y, 0.55, **ring)
    yk = TOP + 7.8 + 4 * 2.15 + 2.3
    A.circle(gx[0] - 2.4, yk, 0.8, fill="grey_dark")
    A.text(gx[0] - 1.2, yk, "ρ > 0, P < 0.05", role=fine, color="grey_dark")
    A.text(rx, TOP + 21.6, "Lymphocytes track inflammation; stroma inversely", role=annot)
    A.text(rx, TOP + 21.6 + 2.4, "genetic risk and inflammation share compartments", role=fine, color="grey_dark")

    # ---------------- consequence 2: drug targets
    A.card(RX, BOT, RW, CH)
    A.text(rx, BOT + 2.9, "Drug targets", role=leg, bold=True)
    A.cylinder(rx + 0.2, BOT + 4.8, 4.0, 4.4, color="grey_dark")
    A.text(rx + 6.2, BOT + 5.8, "Open Targets", role=fine, bold=True)
    A.text(rx + 6.2, BOT + 5.8 + dy, "8,420 RA-associated targets", role=fine)
    A.pill(rx + 2.2, BOT + 12.4, 5.2, angle=0, color="grey_dark")
    A.text(rx + 6.2, BOT + 11.4, "ChEMBL", role=fine, bold=True)
    A.text(rx + 6.2, BOT + 11.4 + dy, "approved targeted-DMARD targets", role=fine)
    A.text(rx, BOT + 17.2, "Genetically supported and approved targets", role=annot)
    A.text(rx, BOT + 17.2 + 2.4, "concentrate in lymphocyte / immune states", role=annot)
    A.text(rx, BOT + 23.0, "Stroma: a genetic blind spot", role=base, color="vermilion", bold=True)
    A.text(rx, BOT + 25.5, "little polygenic risk, implicated in refractory RA", role=fine, color="grey_dark")
    return A


# ------------------------------------------------------------------------------------------------ data panels
def umap_frame(ax, xlab, ylab):
    """UMAP convention: no ticks, a short L of the axes at the lower left, axis names at its start."""
    ax.set(xticks=[], yticks=[])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.spines["bottom"].set_bounds(x0, x0 + 0.22 * (x1 - x0))
    ax.spines["left"].set_bounds(y0, y0 + 0.22 * (y1 - y0))
    ax.set_xlabel(xlab, loc="left", fontsize=pt("annot"), labelpad=1.5)
    ax.set_ylabel(ylab, loc="bottom", fontsize=pt("annot"), labelpad=1.5)


def panel_umap(ax, d, states):
    rng = np.random.default_rng(0)
    dd = d.iloc[rng.permutation(len(d))]
    ax.scatter(dd.umap1, dd.umap2, s=0.12, c=dd.state_group.map(COMP_COLOR), linewidths=0, rasterized=True)
    for comp in COMP_ORDER:
        big = states[states.state_group == comp].state.iloc[0]          # anchor the name on the largest state
        m = d[d.state == big][["umap1", "umap2"]].median()
        ax.text(m.umap1, m.umap2, comp, fontsize=pt("annot"), fontweight="bold", ha="center", va="center",
                color=COMP_COLOR[comp], bbox=dict(boxstyle="round,pad=0.15", fc=C.white, ec="none", alpha=0.85))
    umap_frame(ax, "UMAP1", "UMAP2")
    fk.panel(ax, "b", f"{len(d):,} cells, {d['sample'].nunique()} samples")


def panel_key(ax, d, states):
    handles = [mpl.lines.Line2D([], [], marker="o", ls="", color=COMP_COLOR[c], ms=4.5,
                                label=f"{c}\n{(states.state_group == c).sum()} states, {(d.state_group == c).sum():,} cells")
               for c in COMP_ORDER]
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=pt("legend"), labelspacing=0.8,
              handletextpad=0.3, borderaxespad=0)


def panel_substates(axes, d, states):
    rng = np.random.default_rng(0)
    for k, (ax, comp) in enumerate(zip(axes, COMP_ORDER)):
        s = d[d.state_group == comp]
        st = states[states.state_group == comp]
        pal = {n: fk.c[STATE_COLORS[i]] for i, n in enumerate(st.state)}
        so = s.iloc[rng.permutation(len(s))]
        ax.scatter(so.sub_umap1, so.sub_umap2, s=0.25 if len(s) < 5000 else 0.12, c=so.state.map(pal),
                   linewidths=0, rasterized=True)
        texts, xs, ys = [], [], []
        for _, r in st.iterrows():
            m = s[s.state == r.state][["sub_umap1", "sub_umap2"]].median()
            texts.append(ax.text(m.sub_umap1, m.sub_umap2, str(r.num), fontsize=pt("fine"), ha="center",
                                 va="center", fontweight="bold",
                                 bbox=dict(boxstyle="circle,pad=0.12", fc=C.white, ec=C.grey_dark, lw=0.4,
                                           alpha=0.9)))
            xs.append(m.sub_umap1)
            ys.append(m.sub_umap2)
        fk.repel(ax, texts, x=xs, y=ys)
        title = f"{comp} ({len(s):,})"
        if k == 0:
            fk.panel(ax, "e", title)
        else:
            ax.set_title(title, loc="left", fontsize=pt("title"), fontweight="bold")
        ax._left_title.set_color(COMP_COLOR[comp])
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)


def panel_composition(ax, d):
    share = pd.crosstab(d["sample"], d.state_group, normalize="index")[COMP_ORDER].sort_values("Stromal")
    bottom = np.zeros(len(share))
    for comp in COMP_ORDER:
        ax.bar(np.arange(len(share)), share[comp], bottom=bottom, width=0.86, color=COMP_COLOR[comp], linewidth=0)
        bottom += share[comp].to_numpy()
    ax.set(xlim=(-0.7, len(share) - 0.3), ylim=(0, 1), xticks=[], yticks=[0, 0.5])
    ax.set_xlabel(f"Samples (n = {len(share)})")
    ax.set_ylabel("Fraction of cells")
    fk.panel(ax, "c", "Composition")


def marker_matrix(states):
    """Top signature gene per state (first not already used) and its expression in the state itself and in the
    other states of each compartment (unweighted mean over states)."""
    sig = read("state_signatures")
    genes = []
    for stn in states.state:
        for g in sig[sig.state == stn].sort_values("rank").gene:
            if g not in genes:
                genes.append(g)
                break
    pb = read("state_pseudobulk_long")
    pb = pb[pb.gene.isin(genes) & pb.state.isin(states.state)]
    mean = pb.pivot(index="state", columns="gene", values="mean_lognorm").loc[states.state, genes]
    pct = pb.pivot(index="state", columns="gene", values="pct_detected").loc[states.state, genes]
    scaled = (mean - mean.min()) / (mean.max() - mean.min() + 1e-9)      # per gene, over the 54 states
    grp = states.set_index("state").state_group
    V, P = np.zeros((len(states), 6)), np.zeros((len(states), 6))
    for i, (stn, g) in enumerate(zip(states.state, genes)):
        V[i, 0], P[i, 0] = scaled.loc[stn, g], pct.loc[stn, g]
        for j, comp in enumerate(COMP_ORDER, start=1):
            others = [x for x in grp.index[grp == comp] if x != stn]
            V[i, j], P[i, j] = scaled.loc[others, g].mean(), pct.loc[others, g].mean()
    return genes, V, P


def panel_markers(ax, fig, states):
    genes, V, P = marker_matrix(states)
    n = len(states)
    cm = mpl.colors.ListedColormap(fk.cmap()(np.linspace(0.97, 0.2, 256)))   # magma, light = low
    yy, xx = np.meshgrid(np.arange(n), np.arange(6), indexing="ij")
    SMAX = 34
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=P.ravel() * SMAX, c=V.ravel(), cmap=cm, vmin=0, vmax=1,
                    linewidths=0)
    ax.set_yticks(range(n), [f"{k}  {s}" for k, s in zip(states.num, states.state)], fontsize=pt("annot"))
    for lab, g in zip(ax.get_yticklabels(), states.state_group):
        lab.set_color(COMP_COLOR[g])
    ax.set_xticks(range(6), ["Own state"] + COMP_ORDER)
    fk.tilt(ax)
    for lab, g in zip(ax.get_xticklabels()[1:], COMP_ORDER):
        lab.set_color(COMP_COLOR[g])
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(-0.6, 5.6)
    ax.tick_params(length=0)
    ax.axvline(0.5, color=C.grey, lw=0.5)
    for comp in COMP_ORDER[:-1]:
        idx = np.where(states.state_group == comp)[0]
        ax.axhline(idx.max() + 0.5, color=C.grey_light, lw=0.5)
    for i, g in enumerate(genes):
        ax.text(5.75, i, g, fontsize=pt("annot"), fontstyle="italic", va="center", ha="left", clip_on=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [ax.scatter([], [], s=p * SMAX, c=C.grey_dark, linewidths=0, label=f"{int(p * 100)}%")
               for p in (0.25, 0.5, 1.0)]
    fk.legend_top(ax, handles=handles, align="right", title=None)
    cb = fig.colorbar(sc, ax=ax, location="bottom", shrink=0.9, aspect=18, pad=0.005)
    cb.set_label("Scaled mean expression", fontsize=pt("annot"))
    cb.set_ticks([0, 0.5, 1])
    fk.panel(ax, "d", "Top markers")
    return genes


def main():
    d, states = atlas()
    fig = figure("double", 230)
    gs = fig.add_gridspec(2, 1, height_ratios=[ART_H + 6, 230 - ART_H - 6])
    axa = fig.add_subplot(gs[0])
    fk.art.place(axa, schematic(states))
    fk.panel(axa, "a", "Study design: RA genetics × synovial atlases → genetic map → inflammation and drug targets")
    bot = gs[1].subgridspec(1, 2, width_ratios=[100, 21])
    left = bot[0].subgridspec(3, 3, height_ratios=[1.25, 1, 1], width_ratios=[1, 1, 1])
    top = left[0, :].subgridspec(1, 2, width_ratios=[1.55, 1])
    panel_umap(fig.add_subplot(top[0]), d, states)
    panel_composition(fig.add_subplot(top[1]), d)
    genes = panel_markers(fig.add_subplot(bot[1]), fig, states)
    axe = [fig.add_subplot(left[1, 0]), fig.add_subplot(left[1, 1]), fig.add_subplot(left[1, 2]),
           fig.add_subplot(left[2, 0]), fig.add_subplot(left[2, 1])]
    panel_substates(axe, d, states)
    panel_key(fig.add_subplot(left[2, 2]), d, states)
    if len(sys.argv) > 2 and sys.argv[1] == "--draft":            # throw-away exploration, never archived
        print(fk.save(fig, Path(sys.argv[2]), strict=False).report)
        return
    saved = save(fig, "Fig1_atlas", provenance={
        "source": "db/ra_rx.duckdb::atlas_cells_state, atlas_cells_lineage, atlas_cells_fine, state_signatures, "
                  "state_pseudobulk_long, magma_genes_ra_eur_noMHC, ra_causal_genes, scdrs_state_load, "
                  "scdrs_group_gupta_state, magma_celltype, causal_gene_state_enrichment, gupta_state_summary, "
                  "activity_assoc_pooled",
        "scripts": ["scripts/10_atlas_micheroli.py", "scripts/11_fine_states.py", "scripts/12_curate_states.py",
                    "scripts/13_state_pseudobulk.py", "scripts/14_atlas_gupta.py", "scripts/20_magma.py",
                    "scripts/21_scdrs.py", "scripts/22_scdrs_summary.py", "scripts/23_magma_celltype.py",
                    "scripts/24_causal_genes.py", "scripts/30_state_signatures.py", "scripts/31_score_bulk.py",
                    "scripts/34_disease_activity.py", "scripts/fig1_atlas.py"],
        "transformations": ["QC 5-MAD per sample, mt<20%", "scVI (global + per lineage)", "manual curation",
                            "d: dot colour = per-gene min-max of state means; compartment columns = unweighted mean "
                            "over the other states of that compartment",
                            "a: Manhattan = MAGMA gene P (RA EUR noMHC; all genes with P<0.01 + 10% of the rest; "
                            "line = Bonferroni); atlas dots = one per state; genetic map = share of a compartment's "
                            "states with FDR<0.05 enrichment (scDRS z>0, BH within atlas; MAGMA gene-property FDR; "
                            "causal-gene state FDR) for ra_eur_noMHC; inflammation dots = pooled rho>0 and P<0.05 "
                            "(compartment score vs DAS28/CRP/ESR, activity_assoc_pooled)"],
        "n": f"{len(d)} cells, {d['sample'].nunique()} samples, {len(states)} states (E-MTAB-11791); "
             "27,856 nuclei, 76 states (GSE243917)"})
    print(saved.report)


def preview():
    d, states = atlas()
    A = schematic(states)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "logs" / "fig1a_preview.png"
    A.png(out, dpi=400)
    print(out, A.problems())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--art":
        preview()
    else:
        main()
