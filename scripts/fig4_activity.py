"""fig4_activity.py — Figure 5: bulk synovial cohorts; pathotype positive control; the genetically loaded lymphocyte
compartments track baseline disease activity, and the coupling is between compartments, not within them.

a  cohorts: baseline biopsies per cohort (treatment context under the name); columns on the right give the number
   with DAS28 and with CRP available (clin_activity_harmonised: DAS28-ESR computed for R4RA / STRAP; deposited DAS28
   for PEAC and GSE198520; deposited DAS28-CRP for GSE45867)
b  pathotype positive control: number of the 52 bulk-scorable states whose score differs between histological
   pathotypes (Kruskal-Wallis, BH-FDR < 0.05 within cohort), stacked by compartment
c  pathotype heatmap: mean within-cohort state score (z) by pathotype (L lymphoid, M myeloid, F fibroid) in R4RA,
   STRAP and PEAC; dot = state with pathotype FDR < 0.05 in that cohort, placed on its highest pathotype; states
   grouped by compartment (colour strip) and ordered by genetic load (scDRS z) within compartment
d  compartment-mean score vs baseline activity: pooled Spearman rho (Fisher-z DerSimonian-Laird, 95% CI) with
   DAS28, log CRP and ESR; ticks = single cohorts; outlined = pooled P < 0.05
e  state-level map: genetic load (scDRS z, RA MHC removed) vs pooled Spearman rho of the state score with DAS28;
   all-state and within-compartment Spearman from activity_state_test (descriptive: states are correlated);
   lines = least-squares guides (dashed: all states; solid: within compartment)
Sources: bulk_samples, clin_activity_harmonised, pathotype_assoc_state, activity_assoc(_pooled), activity_state,
activity_state_test.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.transforms import ScaledTranslation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figkit.rules import rules as fk_rules  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_MARKER, COMP_ORDER, figure, fk, pt, save  # noqa: E402

PATHO = ["Lymphoid", "Myeloid", "Fibroid"]
PATHO_COHORTS = ["R4RA", "STRAP", "PEAC"]
COHORTS = ["R4RA", "STRAP", "PEAC", "GSE198520", "GSE45867"]
CONTEXT = {"R4RA": "RTX vs TCZ trial", "STRAP": "ETN/TCZ/RTX trial", "PEAC": "early RA, untreated",
           "GSE198520": "anti-TNF", "GSE45867": "TCZ or MTX"}
CLIN = [("das28", "DAS28"), ("crp", "log CRP"), ("esr", "ESR")]   # SJC and CDAI: text + Supplementary Table S11
TITLE_PAD = fk_rules()["layout"]["title_pad_mm"] * 72 / 25.4      # facet titles on the panel-title line


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10$^{{{int(e)}}}$"


def panel_cohorts(ax, M, H):
    base = M[M.timepoint == "baseline"][["cohort", "sample"]].merge(H, on=["cohort", "sample"], how="left")
    t = base.groupby("cohort").agg(n=("sample", "size"), das28=("das28", "count"), crp=("crp", "count")).loc[COHORTS]
    y = np.arange(len(t))
    ax.barh(y, t.n, color=C.grey_dark, height=0.62)
    tr = ax.get_yaxis_transform()
    for x, col, head in [(1.08, "n", "n"), (1.37, "das28", "DAS28"), (1.66, "crp", "CRP")]:
        ax.text(x, -0.95, head, transform=tr, ha="center", va="center", fontsize=pt("annot"), fontweight="bold")
        for i, v in enumerate(t[col]):
            ax.text(x, i, str(int(v)) if v else "–", transform=tr, ha="center", va="center", fontsize=pt("annot"))
    ax.set_yticks(y, [f"{c}\n{CONTEXT[c]}" for c in t.index])
    ax.set_xlim(0, t.n.max() * 1.04)
    ax.set_ylim(len(t) - 0.45, -1.35)
    ax.spines["left"].set_bounds(len(t) - 0.45, -0.5)
    ax.set_xlabel("Baseline biopsies")
    fk.panel(ax, "a", "Bulk RNA-seq cohorts")
    return t


def panel_counts(ax, P):
    coh = COHORTS[:4]
    sig = P[P.fdr < 0.05]
    left = np.zeros(len(coh))
    for comp in COMP_ORDER:
        v = np.array([((sig.cohort == c) & (sig.state_group == comp)).sum() for c in coh])
        ax.barh(range(len(coh)), v, left=left, color=COMP_COLOR[comp], height=0.62, label=comp)
        left += v
    for i, v in enumerate(left):
        ax.text(v + 1, i, f"{int(v)}", va="center", fontsize=pt("annot"))
    n = P.groupby("cohort").state.nunique().iloc[0]
    ax.set_yticks(range(len(coh)), coh)
    ax.set_xlim(0, n)
    ax.set_ylim(len(coh) - 0.4, -0.6)
    ax.set_xlabel(f"States with FDR < 0.05\n(of {n})")
    fk.panel(ax, "b", "Pathotype control")


def panel_heat(ax, P, M, order, grp):
    cols = [(c, p) for c in PATHO_COHORTS for p in PATHO]
    Pi = P.set_index(["cohort", "state"])
    H = np.array([[Pi.loc[(c, s), f"mean_{p}"] for c, p in cols] for s in order])
    im = ax.imshow(np.clip(H, -1, 1), aspect="auto", cmap=fk.cmap(kind="diverging"), vmin=-1, vmax=1,
                   interpolation="nearest", extent=(-0.5, len(cols) - 0.5, len(order) - 0.5, -0.5))
    for j0, c in enumerate(PATHO_COHORTS):
        for i, s in enumerate(order):
            r = Pi.loc[(c, s)]
            if r.fdr < 0.05:
                ax.scatter(j0 * 3 + int(np.argmax([r[f"mean_{p}"] for p in PATHO])), i, s=1.6, c=C.ink,
                           linewidths=0, zorder=3)
        if j0:
            ax.axvline(j0 * 3 - 0.5, color=C.white, lw=1.2)
    for comp in COMP_ORDER:
        idx = [i for i, s in enumerate(order) if grp[s] == comp]
        ax.add_patch(mpl.patches.Rectangle((-1.15, min(idx) - 0.5), 0.45, len(idx), color=COMP_COLOR[comp], lw=0))
        if max(idx) < len(order) - 1:
            ax.axhline(max(idx) + 0.5, color=C.white, lw=1.0)
    ax.set_xlim(-1.2, len(cols) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(cols)), [p[0] for _, p in cols])
    ax.set_yticks(range(len(order)), order, fontsize=pt("fine"))
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig = ax.figure
    tr = ax.get_xaxis_transform() + ScaledTranslation(0, -2 / 72, fig.dpi_scale_trans)
    base = M[M.timepoint == "baseline"]
    for j0, c in enumerate(PATHO_COHORTS):
        n = int(base[(base.cohort == c) & base.pathotype.isin(PATHO)].shape[0])
        ax.plot([j0 * 3 - 0.42, j0 * 3 + 2.42], [0, 0], color=C.ink, lw=0.6, transform=tr, clip_on=False)
        ax.text(j0 * 3 + 1, 0, f"{c} ({n})", transform=tr + ScaledTranslation(0, -1 / 72, fig.dpi_scale_trans),
                ha="center", va="top", fontsize=pt("annot"))
    cax = ax.inset_axes([0.1, -0.05, 0.8, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["≤ −1", "0", "≥ 1"])
    cb.outline.set_linewidth(0.4)
    cb.set_label("Mean state score (z)", fontsize=pt("base"))
    fk.panel(ax, "c", "Score by pathotype")


def panel_activity(axes, AP, AC):
    y = np.arange(len(COMP_ORDER))
    for ax, (clin, lab) in zip(axes, CLIN):
        ap = AP[AP.clinical == clin].set_index("variable")
        for i, comp in enumerate(COMP_ORDER):
            r = ap.loc[comp]
            col = COMP_COLOR[comp]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=0.9, solid_capstyle="butt")
            cc = AC[(AC.variable == comp) & (AC.clinical == clin)]
            ax.scatter(cc.rho, [i] * len(cc), s=9, marker="|", color=col, linewidth=0.6, zorder=2)
            ax.scatter(r.rho, i, s=20, marker=COMP_MARKER[comp], color=col, zorder=3,
                       edgecolor=C.ink if r.p < 0.05 else "none", linewidth=0.6)
        ax.plot([0, 0], [len(COMP_ORDER) - 0.5, -0.5], color=C.grey, lw=0.5, ls="--", zorder=1)
        ax.set_xlim(-0.45, 0.5)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4], ["−0.4", "", "0", "", "0.4"])
        title = f"{lab} (n = {int(ap.n_total.max())})"
        if ax is axes[0]:
            fk.panel(ax, "d", title)
        else:
            ax.set_title(title, loc="left", fontsize=pt("title"), fontweight="bold", pad=TITLE_PAD)
    axes[1].set_xlabel("Pooled Spearman ρ (95% CI)")
    axes[0].set_yticks(y, COMP_ORDER)
    axes[0].set_ylabel("Compartment-mean score")
    axes[0].set_ylim(len(COMP_ORDER) - 0.5, -0.6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)


def panel_map(ax, A, T):
    a = A[A.clinical == "das28"].copy()
    t = T[T.clinical == "das28"].set_index("scope")
    ax.errorbar(a.genetic_load, a.rho, yerr=[a.rho - a.ci_lo, a.ci_hi - a.rho], fmt="none", ecolor=C.grey_faint,
                lw=0.5, zorder=1)
    b = np.polyfit(a.genetic_load, a.rho, 1)
    xs = np.array([a.genetic_load.min(), a.genetic_load.max()])
    ax.plot(xs, np.polyval(b, xs), color=C.grey, lw=0.8, ls="--", zorder=2)
    hs = []
    for comp in COMP_ORDER:
        s = a[a.state_group == comp]
        key = f"within {comp}"
        if key in t.index and len(s) >= 5:
            bb = np.polyfit(s.genetic_load, s.rho, 1)
            xx = np.array([s.genetic_load.min(), s.genetic_load.max()])
            ax.plot(xx, np.polyval(bb, xx), color=COMP_COLOR[comp], lw=1.1, zorder=2, solid_capstyle="round")
            lab = f"{comp} ({len(s)}): ρ = {t.loc[key, 'rho']:+.2f}, P = {t.loc[key, 'p']:.2f}".replace("-", "−")
        else:
            lab = f"{comp} ({len(s)}): too few"
        ax.scatter(s.genetic_load, s.rho, s=16, color=COMP_COLOR[comp], marker=COMP_MARKER[comp],
                   edgecolor=C.white, linewidth=0.3, zorder=3)
        hs.append(mpl.lines.Line2D([], [], ls="", marker=COMP_MARKER[comp], ms=3.8, color=COMP_COLOR[comp],
                                   label=lab))
    ax.axhline(0, color=C.grey, lw=0.5, zorder=0)
    al = t.loc["all states"]
    ax.text(0.02, 0.98, f"All {int(al.n_states)} states: ρ = {al.rho:.2f}, P = {fmt_p(al.p)}\n"
            "coupling is between compartments;\nwithin T/NK and myeloid it is negative",
            transform=ax.transAxes, ha="left", va="top", fontsize=pt("annot"),
            bbox=dict(fc=C.white, ec="none", alpha=0.85, pad=0.5))
    ax.legend(handles=hs, loc="lower right", frameon=False, title="Within compartment (n states)",
              title_fontsize=pt("annot"), fontsize=pt("annot"), handletextpad=0.1, labelspacing=0.3,
              borderaxespad=0.2, alignment="left")
    pick = {"T CD4+ Treg": a.genetic_load.idxmax(), "Mono IL1B+ FCN1+": a.rho.idxmax(),
            "Fib SCARA5+ TNXB+ sublining": a.rho.idxmin(),
            "Mac MERTK+ LYVE1+": a[a.state_group == "Myeloid"].genetic_load.idxmax()}
    texts = [ax.text(a.loc[i, "genetic_load"], a.loc[i, "rho"], a.loc[i, "state"], fontsize=pt("annot"),
                     color=C.grey_dark, bbox=dict(fc=C.white, ec="none", alpha=0.85, pad=0.3))
             for i in pick.values()]
    fk.repel(ax, texts, x=a.genetic_load.to_numpy(), y=a.rho.to_numpy())
    ax.set_ylim(-0.36, 0.46)
    ax.set_xlabel("Genetic load of state (scDRS z, RA MHC removed)")
    ax.set_ylabel("Spearman ρ of state score with DAS28\n(pooled, 5 cohorts, 95% CI)")
    fk.panel(ax, "e", f"State level: genetic load vs DAS28 ({len(a)} states)")


def main():
    M = read("bulk_samples")
    H = read("clin_activity_harmonised")
    P = read("pathotype_assoc_state")
    P["state_group"] = P.state_group.astype(str)
    AP, AC = read("activity_assoc_pooled"), read("activity_assoc")
    A = read("activity_state")
    A["state_group"] = A.state_group.astype(str)
    T = read("activity_state_test")
    a = A[A.clinical == "das28"].copy()
    grp = a.set_index("state").state_group
    a["_c"] = a.state_group.map({c: i for i, c in enumerate(COMP_ORDER)})
    order = a.sort_values(["_c", "genetic_load"], ascending=[True, False]).state.tolist()

    fig = figure("double", 205)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.85, 1], height_ratios=[0.62, 0.62, 1.15])
    top = gs[0, 0].subgridspec(1, 2, width_ratios=[1.5, 1])
    mid = gs[1, 0].subgridspec(1, 3, width_ratios=[1.12, 1, 1])
    panel_cohorts(fig.add_subplot(top[0]), M, H)
    panel_counts(fig.add_subplot(top[1]), P)
    panel_heat(fig.add_subplot(gs[:, 1]), P, M, order, grp)
    axd = [fig.add_subplot(mid[0])]
    axd += [fig.add_subplot(mid[i], sharey=axd[0]) for i in (1, 2)]
    panel_activity(axd, AP, AC)
    panel_map(fig.add_subplot(gs[2, 0]), A, T)

    saved = save(fig, "Fig5_cohorts_activity", provenance={
        "source": "db/ra_rx.duckdb::bulk_samples, clin_activity_harmonised, pathotype_assoc_state, activity_assoc, "
                  "activity_assoc_pooled, activity_state, activity_state_test",
        "scripts": ["scripts/31_score_bulk.py", "scripts/32_response_association.py",
                    "scripts/34_disease_activity.py", "scripts/fig4_activity.py"],
        "transformations": ["compartment mean = mean of its states' scores", "pathotype heatmap clipped at |z| = 1",
                            "Fisher-z DerSimonian-Laird pooling of per-cohort Spearman rho",
                            "least-squares lines in e are display guides only"]})
    print(saved.report)


if __name__ == "__main__":
    main()
