"""fig7_targets.py — Figure 6: RA drug-target discovery is concentrated in genetically loaded compartments.

a  per state (n = 54): genetic load (scDRS z, RA MHC removed) vs % of its specific genes that are genetically
   prioritised RA targets (Open Targets genetic_association >= 0.5); Spearman rho (state-level, descriptive; the
   two axes partly share GWAS input); the three extreme states are named
b  % of compartment-specific genes with genetic support, immune- vs stromal-specific genes (95% Wilson CI);
   Fisher exact P on the 2 x 2 counts and one-sided Mann-Whitney P on the continuous genetic score
c  clinically pursued RA targets (OT clinical >= 0.5) by the compartment that specifically expresses them,
   split by genetic support; genetically supported targets named
d  targets of approved / late-stage targeted DMARD mechanisms (grouped by mechanism, right) across the 54 states
   (grouped by compartment, top): dot size = % of cells detected, colour = specificity z (clipped -1 to 3)
Sources: ot_target_state_summary, ot_ra_targets, state_pseudobulk_long.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import fmt_p, C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402

IMMUNE = {"T/NK", "B/plasma", "Myeloid"}
# targets of biologic / targeted-synthetic DMARDs approved for RA (ChEMBL max phase 4 for RA; table
# approved_target_compartment, scripts/73_target_revision.py), grouped by mechanism. Abatacept targets CD80/CD86.
MECH = [("anti-TNF", ["TNF"]), ("anti-IL-6R", ["IL6R"]), ("anti-CD20", ["MS4A1"]),
        ("CTLA4-Ig", ["CD80", "CD86"]), ("JAKi", ["JAK1", "JAK2", "JAK3", "TYK2"]), ("IL-1Ra", ["IL1R1"])]
IMM_COL, STRO_COL = C.grey_dark, COMP_COLOR["Stromal"]
SUPPORT_COL, NOSUPPORT_COL = C.amber, C.grey_light


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def load():
    summ = read("ot_target_state_summary")
    summ["state_group"] = summ.state_group.astype(str)
    ot = read("ot_ra_targets")
    pb = read("state_pseudobulk_long")
    pb["state_group"] = pb.state_group.astype(str)
    spec = pb[(pb.spec_z >= 2) & (pb.pct_detected >= 0.10)].merge(ot, on="gene", how="left").fillna(
        {"genetic_association": 0.0, "clinical": 0.0})
    per_gene = spec.groupby("gene").agg(immune=("state_group", lambda s: s.isin(IMMUNE).any()),
                                        stromal=("state_group", lambda s: (s == "Stromal").any()),
                                        ga=("genetic_association", "first"), clin=("clinical", "first"))
    imm, stro = per_gene[per_gene.immune & ~per_gene.stromal], per_gene[per_gene.stromal & ~per_gene.immune]
    return summ, pb, imm, stro


def panel_a(ax, summ):
    for comp in COMP_ORDER:
        s = summ[summ.state_group == comp]
        ax.scatter(s.assoc_mcz, 100 * s.frac_genetic, s=14, c=COMP_COLOR[comp], marker=COMP_MARKER[comp],
                   edgecolor=C.white, linewidth=0.3, label=comp, zorder=3)
    r, p = stats.spearmanr(summ.assoc_mcz, summ.frac_genetic)
    ax.text(0.97, 0.04, f"ρ = {r:.2f}, P = {fmt_p(p)}\n{len(summ)} states (descriptive)", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=pt("annot"))
    top_frac, top_load = summ.frac_genetic.idxmax(), summ.assoc_mcz.idxmax()
    top_mye = summ[summ.state_group == "Myeloid"].frac_genetic.idxmax()
    for i, off, ha, va in [(top_frac, (-4, 5), "right", "bottom"), (top_load, (-2, -9), "right", "top"),
                           (top_mye, (5, 4), "left", "bottom")]:
        row = summ.loc[i]
        ax.annotate(row.state, (row.assoc_mcz, 100 * row.frac_genetic), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=pt("annot"),
                    arrowprops=dict(arrowstyle="-", color=C.grey, lw=0.4, shrinkA=0, shrinkB=2))
    ax.set_ylim(0, 100 * summ.frac_genetic.max() * 1.18)
    ax.set_xlim(summ.assoc_mcz.min() - 0.6, 10.8)
    ax.set(xlabel="Genetic load (scDRS z, RA MHC removed)",
           ylabel="State-specific genes that are\ngenetic RA targets (%)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.1, borderaxespad=0.2, fontsize=pt("legend"))
    fk.panel(ax, "a", "Genetic load vs genetic targets")


def panel_b(ax, T):
    """Share of immune- vs stromal-specific genes with Open Targets genetic support, at three score thresholds
    (exclusive assignment; descriptive: the genetic score derives partly from the same GWAS)."""
    d = T[(T.assignment == "exclusive") & (T.evidence == "genetic")].sort_values("threshold")
    w = 0.36
    for j, r in enumerate(d.itertuples()):
        for i, (hit, n, col) in enumerate([(r.immune_hit, r.immune_n, IMM_COL), (r.stromal_hit, r.stromal_n, STRO_COL)]):
            pr, lo, hi = wilson(hit, n)
            x = j + (i - 0.5) * w
            ax.bar(x, 100 * pr, width=w * 0.92, color=col, label=(["Immune-specific", "Stromal-specific"][i] if j == 0 else None))
            ax.errorbar(x, 100 * pr, yerr=[[100 * (pr - lo)], [100 * (hi - pr)]], color=C.ink, lw=0.6, capsize=1.5)
        top = 100 * max(wilson(r.immune_hit, r.immune_n)[2], wilson(r.stromal_hit, r.stromal_n)[2])
        ax.text(j, top + 0.6, f"OR {r.odds_ratio:.1f}\nP = {r.p:.1g}", ha="center", va="bottom", fontsize=pt("annot"))
    ax.set_xticks(range(len(d)), [f"≥ {t:g}" for t in d.threshold])
    ax.set_xlabel("Open Targets genetic score")
    ax.set_ylabel("Genes with genetic support (%)")
    ax.set_ylim(0, 13)
    fk.legend_top(ax, ncol=1)
    fk.panel(ax, "b", "Genetic share (descriptive)")
    return d


def panel_c(ax, T, imm, stro):
    """Clinically pursued RA targets (Open Targets clinical >= 0.5), normalised to the gene denominators."""
    r = T[(T.assignment == "exclusive") & (T.evidence == "clinical")].iloc[0]
    for i, (hit, n, col, d) in enumerate([(r.immune_hit, r.immune_n, IMM_COL, imm), (r.stromal_hit, r.stromal_n, STRO_COL, stro)]):
        pr, lo, hi = wilson(hit, n)
        ax.bar(i, 100 * pr, width=0.62, color=col)
        ax.errorbar(i, 100 * pr, yerr=[[100 * (pr - lo)], [100 * (hi - pr)]], color=C.ink, lw=0.7, capsize=2)
        ax.text(i, 100 * hi + 0.03, f"{hit}/{n:,}", ha="center", va="bottom", fontsize=pt("annot"))
    fk.pbracket(ax, 0, 1, 1.05, f"OR {r.odds_ratio:.1f}, P = {r.p:.2f}")
    ax.set(xticks=[0, 1], xlim=(-0.6, 1.6), ylim=(0, 1.45), ylabel="Clinically pursued targets (%)")
    ax.set_xticklabels(["Immune-\nspecific", "Stromal-\nspecific"])
    fk.panel(ax, "c", "Clinical targets")


def panel_d(ax, summ, pb):
    genes = [g for _, gs in MECH for g in gs if g in set(pb.gene)]
    states = (summ.assign(_c=summ.state_group.map({c: i for i, c in enumerate(COMP_ORDER)}))
              .sort_values(["_c", "assoc_mcz"], ascending=[True, False]))
    P = pb[pb.gene.isin(genes)]
    Zm = P.pivot(index="gene", columns="state", values="spec_z").loc[genes, states.state]
    Pm = P.pivot(index="gene", columns="state", values="pct_detected").loc[genes, states.state]
    yy, xx = np.meshgrid(np.arange(len(genes)), np.arange(len(states)), indexing="ij")
    smax, zlo, zhi = 20, -1, 3
    cmap = fk.cmap("viridis")
    ax.scatter(xx.ravel(), yy.ravel(), s=Pm.to_numpy().ravel() * smax,
               c=np.clip(Zm.to_numpy().ravel(), zlo, zhi), cmap=cmap, vmin=zlo, vmax=zhi, linewidths=0)
    ax.set_xticks(range(len(states)), states.state, rotation=90, fontsize=pt("fine"))
    ax.set_yticks(range(len(genes)), genes, fontsize=pt("tick"), fontstyle="italic")
    ax.set_xlim(-0.7, len(states) - 0.3)
    ax.set_ylim(len(genes) - 0.5, -1.75)
    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for comp in COMP_ORDER:                               # compartment band inside the axes, above the genes
        idx = np.where(states.state_group.to_numpy() == comp)[0]
        ax.add_patch(mpl.patches.Rectangle((idx.min() - 0.45, -1.6), idx.max() - idx.min() + 0.9, 0.85,
                                           color=COMP_COLOR[comp], lw=0))
        ax.text((idx.min() + idx.max()) / 2, -1.175, comp, ha="center", va="center", fontsize=pt("annot"),
                color=C.white)
        if idx.max() < len(states) - 1:
            ax.plot([idx.max() + 0.5] * 2, [-0.5, len(genes) - 0.5], color=C.grey_light, lw=0.4, zorder=0)
    side = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for mech, gs in MECH:
        rows = [genes.index(g) for g in gs if g in genes]
        if not rows:
            continue
        lo, hi = min(rows), max(rows)
        ax.plot([1.006, 1.006], [lo - 0.3, hi + 0.3], color=C.grey_dark, lw=0.6, transform=side, clip_on=False)
        ax.text(1.012, (lo + hi) / 2, mech, transform=side, va="center", ha="left", fontsize=pt("annot"))
        if hi < len(genes) - 1:
            ax.axhline(hi + 0.5, color=C.grey_faint, lw=0.4, zorder=0)
    norm = mpl.colors.Normalize(zlo, zhi)
    L = mpl.lines.Line2D
    hs = [L([], [], ls="", marker="", label="Specificity z")]
    hs += [L([], [], ls="", marker="o", ms=4, color=cmap(norm(v)), label=(f"≥{v}" if v == zhi else f"{v}".replace("-", "−")))
           for v in range(zlo, zhi + 1)]
    hs += [L([], [], ls="", marker="", label="   Detected")]
    hs += [L([], [], ls="", marker="o", color=C.grey_dark, ms=np.sqrt(v * smax), label=f"{int(100 * v)}%")
           for v in (0.25, 0.5, 1.0)]
    fk.legend_top(ax, handles=hs, align="right", handletextpad=0.1, columnspacing=0.7)
    ap = read("approved_target_compartment")
    tt = read("target_tests")
    bt = tt[tt.evidence.str.startswith("approved")].iloc[0]
    fk.panel(ax, "d", f"Approved b/tsDMARD targets (ChEMBL): {int(bt.immune_hit)} of {int(bt.immune_n)} most specific "
                      f"to an immune state (base rate {100 * bt.base_rate:.0f}%, P = {bt.p:.3f})")


def main():
    summ, pb, imm, stro = load()
    T = read("target_tests")
    fig = figure("double", 165)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35])
    top = gs[0].subgridspec(1, 3, width_ratios=[1.1, 1.0, 0.6], wspace=0.08)
    panel_a(fig.add_subplot(top[0]), summ)
    bt = panel_b(fig.add_subplot(top[1]), T)
    panel_c(fig.add_subplot(top[2]), T, imm, stro)
    panel_d(fig.add_subplot(gs[1]), summ, pb)
    saved = save(fig, "Fig6_targets", provenance={
        "source": "db/ra_rx.duckdb::ot_target_state_summary, ot_ra_targets, state_pseudobulk_long",
        "scripts": ["scripts/13_state_pseudobulk.py", "scripts/72_ot_targets.py", "scripts/73_target_revision.py",
                    "scripts/fig7_targets.py"],
        "transformations": ["state-specific gene: spec_z >= 2 and detected >= 10%",
                            "genetic support: Open Targets genetic_association >= 0.5 (MONDO_0008383)",
                            "clinically pursued: Open Targets clinical >= 0.5",
                            "b/c: Fisher exact tests from target_tests (exclusive assignment)",
                            "d: ChEMBL phase-4 RA b/tsDMARD targets; base rate = share of specific genes whose most "
                            "specific state is immune"],
        "note": "Open Targets (release 26.09 served) and ChEMBL (ChEMBL_37) queried on 2026-09-24"})
    print(saved.report)


if __name__ == "__main__":
    main()
