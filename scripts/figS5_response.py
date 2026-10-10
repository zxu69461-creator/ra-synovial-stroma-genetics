"""figS5_response.py — Supplementary Figure 5: treatment response is not predicted by the genetically loaded
programme (all treatment-response analyses in one place).

a  primary pre-specified test: odds ratio of non-response per SD of the Genetic Load Index (GLI, RA-weighted, MHC
   removed) by cohort (95% CI, log scale) and DerSimonian-Laird pooled over the randomised trials (R4RA + STRAP) and
   over all cohorts with >= 8 patients per response class; OA-weighted contrast index in grey
b  compartment-mean state score (mean of its states' scores) in responders vs non-responders at baseline, R4RA and
   STRAP (two-sided Mann-Whitney P per compartment above the boxes)
c  STRAP, exploratory drug specificity (confidently linked biopsies only, FigS3 linkage): OR of ACR20 non-response
   per SD of GLI and of each compartment-mean score within each randomised arm (95% CI, log scale); arm x score
   interaction likelihood-ratio P (compartments: BH-FDR over 5 tests) on the right
d  within-patient change: Spearman rho of the post-treatment change in GLI vs change in DAS28 per cohort (95% CI by
   Fisher z) and Fisher-z DerSimonian-Laird pooled; weight-permutation P = one-sided P over 2,000 random
   permutations of the state weights (activity_longitudinal_pooled.p_perm_weights)
e  STRAP linkage QC: distance margin to the second-best published patient vs 18-subset profile correlation with the
   partner; used = margin >= 0.2 and leave-one-subset-out stability >= 0.9
Sources: gli_response, bulk_samples, bulk_state_scores, response_assoc_state, strap_drug_specific,
activity_longitudinal(_pooled), strap_sample_match.
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
from figkit.rules import rules as fk_rules  # noqa: E402
from figstyle import (C, COMP_COLOR, COMP_MARKER, COMP_ORDER, RESP_COLOR, WEIGHT_COLOR, figure, fk,  # noqa: E402
                      pt, save)

MIN_PER_ARM = 8
MARGIN_MIN, STAB_MIN = 0.2, 0.9
ARMS = ["etanercept", "tocilizumab", "rituximab"]
VARS = [("gli", "GLI (RA)"), ("T_NK", "T/NK"), ("B_plasma", "B/plasma"), ("Myeloid", "Myeloid"),
        ("Endothelial", "Endothelial"), ("Stromal", "Stromal")]
TITLE_PAD = fk_rules()["layout"]["title_pad_mm"] * 72 / 25.4      # facet titles on the panel-title line
OR_TICKS = [0.25, 0.5, 1, 2, 4, 8]


def dl(b, se):
    w = 1 / se ** 2
    fixed = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - fixed) ** 2)
    df = len(b) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    beta = np.sum(wr * b) / np.sum(wr)
    s = np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - df) / q) if q > 0 and df > 0 else 0.0
    return beta, s, 2 * stats.norm.sf(abs(beta / s)), i2


def log_or_axis(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    t = [v for v in OR_TICKS if lo <= v <= hi]
    ax.set_xticks(t, [f"{v:g}" for v in t])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def panel_gli(ax):
    gli = read("gli_response")
    gli = gli[(gli.n_nonresp >= MIN_PER_ARM) & (gli.n - gli.n_nonresp >= MIN_PER_ARM)]
    rows = []
    for trait, key in [("ra_eur_noMHC", "RA"), ("fg_oa", "OA")]:
        g = gli[gli.trait == trait].set_index("cohort").loc[["R4RA", "STRAP", "GSE198520"]]
        for coh, r in g.iterrows():
            rows.append((f"{coh} ({r.n}, {r.n_nonresp} NR)", key, r.log_or, r.se, "o", f"{r.p:.2f}"))
        tr = g.loc[["R4RA", "STRAP"]]
        b, s, p, i2 = dl(tr.log_or.to_numpy(), tr.se.to_numpy())
        rows.append((f"Pooled trials (I² {100 * i2:.0f}%)", key, b, s, "D", f"{p:.2f}"))
        b, s, p, i2 = dl(g.log_or.to_numpy(), g.se.to_numpy())
        rows.append((f"Pooled all (I² {100 * i2:.0f}%)", key, b, s, "D", f"{p:.2f}"))
    y = np.arange(len(rows), dtype=float)
    y[len(rows) // 2:] += 0.8                                    # gap between RA and OA blocks
    dpi = ax.figure.dpi_scale_trans
    col = [ax.get_yaxis_transform() + ScaledTranslation(mm / 25.4, 0, dpi) for mm in (1.5, 19.5)]
    for yi, (name, key, b, s, mk, txt) in zip(y, rows):
        lo, hi = np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)
        ax.plot([lo, hi], [yi, yi], color=WEIGHT_COLOR[key], lw=0.9)
        ax.scatter(np.exp(b), yi, s=16 if mk == "o" else 20, marker=mk, color=WEIGHT_COLOR[key], zorder=3)
        ax.text(1, yi, f"{np.exp(b):.2f} ({lo:.2f}–{hi:.2f})", transform=col[0], va="center",
                fontsize=pt("annot"))
        ax.text(1, yi, txt, transform=col[1], va="center", fontsize=pt("annot"))
    for c, head in zip(col, ["OR (95% CI)", "P"]):
        ax.text(1, -1.0, head, transform=c, va="center", fontsize=pt("annot"), fontweight="bold")
    ax.axvline(1, color=C.grey, lw=0.5, ls="--", zorder=0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(y[-1] + 0.6, -1.5)
    ax.spines["left"].set_bounds(y[-1] + 0.6, -0.6)
    log_or_axis(ax, 0.3, 14)
    ax.set_xlabel("OR of non-response per SD of index (95% CI)")
    hs = [mpl.lines.Line2D([], [], ls="", marker="o", ms=3.6, color=WEIGHT_COLOR["RA"], label="RA-weighted GLI"),
          mpl.lines.Line2D([], [], ls="", marker="o", ms=3.6, color=WEIGHT_COLOR["OA"],
                           label="OA-weighted (contrast)")]
    fk.legend_top(ax, handles=hs, align="right")
    fk.panel(ax, "a", "GLI vs non-response")


def comp_means(M, S, grp, coh):
    rb = M[(M.cohort == coh) & (M.timepoint == "baseline") & M.response_binary.notna()].set_index("sample")
    W = S[S.cohort == coh].pivot(index="sample", columns="state", values="score").loc[rb.index]
    cm = W.T.groupby(W.columns.map(grp)).mean().T[COMP_ORDER]
    return cm, rb.response_binary


def panel_compmeans(ax, M, S, grp):
    """R4RA and STRAP side by side: 5 compartments x (responder, non-responder) per cohort."""
    step = len(COMP_ORDER) + 0.9
    ticks, labels, tops = [], [], []
    res = {}
    for k, coh in enumerate(["R4RA", "STRAP"]):
        cm, y = comp_means(M, S, grp, coh)
        res[coh] = (cm, y)
        pos = k * step + np.arange(len(COMP_ORDER))
        for m, (key, mask) in enumerate([("responder", y == 1), ("non_responder", y == 0)]):
            bp = ax.boxplot([cm.loc[mask, c] for c in COMP_ORDER], positions=pos + (m - 0.5) * 0.4, widths=0.34,
                            patch_artist=True, showfliers=False, medianprops=dict(color=C.ink, lw=0.8),
                            boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
            for b in bp["boxes"]:
                b.set_facecolor(RESP_COLOR[key])
                b.set_edgecolor(C.ink)
            tops += [ln.get_ydata().max() for ln in bp["caps"]]
        ticks += list(pos)
        labels += COMP_ORDER
    top = max(tops)
    for k, coh in enumerate(["R4RA", "STRAP"]):
        cm, y = res[coh]
        for j, c in enumerate(COMP_ORDER):
            p = stats.mannwhitneyu(cm.loc[y == 1, c], cm.loc[y == 0, c])[1]
            ax.text(k * step + j, top + 0.1, f"{p:.2f}", ha="center", va="bottom", fontsize=pt("annot"),
                    fontweight="bold" if p < 0.05 else "normal")
        ax.text(k * step + 2, top + 0.55, f"{coh} (n = {len(y)}, {int((y == 0).sum())} NR)", ha="center",
                va="bottom", fontsize=pt("annot"))
    ax.set_ylim(top=top + 0.95)
    ax.set_xticks(ticks, labels)
    fk.tilt(ax)
    ax.set_xlim(-0.6, step + len(COMP_ORDER) - 0.4)
    ax.set_ylabel("Compartment-mean score (z)")
    fk.legend_top(ax, [mpl.patches.Patch(color=RESP_COLOR["responder"]),
                       mpl.patches.Patch(color=RESP_COLOR["non_responder"])], ["responder", "non-responder"],
                  align="right")
    fk.panel(ax, "b", "Baseline scores by response")


def panel_drug(axes):
    D = read("strap_drug_specific")
    y = np.arange(len(VARS))
    col = {"gli": WEIGHT_COLOR["RA"], **{k: COMP_COLOR[v] for k, v in VARS[1:]}}
    mk = {"gli": "D", **{k: COMP_MARKER[v] for k, v in VARS[1:]}}
    for ax, arm in zip(axes, ARMS):
        s = D[D.arm == arm].set_index("variable")
        for yi, (k, _) in enumerate(VARS):
            row = s.loc[k]
            ax.plot([row.ci_lo, row.ci_hi], [yi, yi], color=col[k], lw=0.9)
            ax.scatter(row.or_per_sd, yi, s=16, color=col[k], marker=mk[k], zorder=3,
                       edgecolor=C.ink if row.p < 0.05 else "none", linewidth=0.5)
        ax.plot([1, 1], [len(VARS) - 0.5, -0.5], color=C.grey, lw=0.5, ls="--")
        log_or_axis(ax, 0.15, 3.5)
        n, nn = int(s.n.iloc[0]), int(s.n_nonresp.iloc[0])
        title = f"STRAP {arm} (n = {n}, {nn} NR)" if ax is axes[0] else f"{arm.capitalize()} (n = {n}, {nn} NR)"
        if ax is axes[0]:
            fk.panel(ax, "c", title)
        else:
            ax.set_title(title, loc="left", fontsize=pt("title"), fontweight="bold", pad=TITLE_PAD)
    axes[1].set_xlabel("OR of ACR20 non-response per SD (95% CI)")
    axes[0].set_yticks(y, [v for _, v in VARS])
    axes[0].set_ylim(len(VARS) - 0.5, -0.6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    s = D.drop_duplicates("variable").set_index("variable")
    last = axes[-1]
    tr = last.get_yaxis_transform()
    last.text(1.05, -0.9, "Arm × score interaction", transform=tr, fontsize=pt("annot"), va="center",
              fontweight="bold")
    for yi, (k, _) in enumerate(VARS):
        txt = f"P = {s.loc[k, 'p_interaction']:.2f}"
        if not np.isnan(s.loc[k, "fdr_interaction"]):
            txt += f", FDR = {s.loc[k, 'fdr_interaction']:.2f}"
        last.text(1.05, yi, txt, transform=tr, fontsize=pt("annot"), va="center")


def panel_long(ax):
    order = ["R4RA", "GSE198520", "GSE45867"]
    weeks = {"R4RA": "wk 16", "GSE198520": "wk 12", "GSE45867": "wk 12"}
    rows = []
    for var, key in [("GLI (RA)", "RA"), ("GLI (OA contrast)", "OA")]:
        L = read("activity_longitudinal").query("variable == @var and clinical == 'das28'").set_index("cohort")
        LP = read("activity_longitudinal_pooled").query("variable == @var and clinical == 'das28'").iloc[0]
        for coh, r in L.loc[order].iterrows():
            z, se = np.arctanh(r.rho), 1 / np.sqrt(r.n - 3)
            rows.append((f"{coh} ({weeks[coh]}, {r.n})", key, r.rho, np.tanh(z - 1.96 * se),
                         np.tanh(z + 1.96 * se), "o", f"P = {r.p:.2f}"))
        txt = f"P = {LP.p:.2f}, I² = {100 * LP.i2:.0f}%"
        if not np.isnan(LP.p_perm_weights):
            txt += f"; weight-permutation P = {LP.p_perm_weights:.2f}"
        rows.append((f"Pooled ({int(LP.n_total)})", key, LP.rho, LP.ci_lo, LP.ci_hi, "D", txt))
    y = np.arange(len(rows), dtype=float)
    y[len(rows) // 2:] += 0.8
    for yi, (name, key, rho, lo, hi, mk, txt) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=WEIGHT_COLOR[key], lw=0.9)
        ax.scatter(rho, yi, s=16 if mk == "o" else 20, marker=mk, color=WEIGHT_COLOR[key], zorder=3)
        ax.text(1.03, yi, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=pt("annot"))
    ax.axvline(0, color=C.grey, lw=0.5, ls="--", zorder=0)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(y[-1] + 0.6, -0.6)
    ax.set_xlim(-0.65, 0.65)
    ax.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"])
    ax.set_xlabel("Spearman ρ, Δindex vs ΔDAS28 (95% CI)")
    fk.panel(ax, "d", "Within-patient change")


def panel_linkage(ax):
    m = read("strap_sample_match")
    d = m[m.patient.notna()]
    for flag, col, lab in [(False, C.grey, "Not used"), (True, C.grey_dark, "Used")]:
        s = d[d.used == flag]
        ax.scatter(s.margin, s.profile_r, s=9, facecolor=col if flag else "none", edgecolor=col, linewidth=0.5,
                   label=f"{lab} ({len(s)})", zorder=3)
    ax.axvline(MARGIN_MIN, color=C.vermilion, lw=0.7, ls="--", label=f"Threshold ({MARGIN_MIN})")
    n_stab = int((d.stability >= STAB_MIN).sum())
    ax.text(0.97, 0.04, f"Stability ≥ {STAB_MIN}: {n_stab}/{len(d)}\n{len(m) - len(d)} samples unmatched",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=pt("annot"))
    ax.set(xlabel="Distance margin to 2nd-best patient", ylabel="18-subset profile r with partner")
    ax.legend(loc="upper right", frameon=False, fontsize=pt("legend"), handletextpad=0.2, borderaxespad=0.2)
    ax.set_ylim(d.profile_r.min() - 0.03, 1.17)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.spines["left"].set_bounds(ax.get_ylim()[0], 1.0)
    fk.panel(ax, "e", f"STRAP linkage QC ({int(d.used.sum())}/{len(m)} used)")


def main():
    M = read("bulk_samples")
    S = read("bulk_state_scores")
    R = read("response_assoc_state")
    R["state_group"] = R.state_group.astype(str)
    grp = R.drop_duplicates("state").set_index("state").state_group

    fig = figure("double", 175, kind="supp")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.8, 1.0])
    top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.05])
    mid = gs[1].subgridspec(1, 3)
    bot = gs[2].subgridspec(1, 2, width_ratios=[1, 0.9])
    panel_gli(fig.add_subplot(top[0]))
    panel_compmeans(fig.add_subplot(top[1]), M, S, grp)
    axc = [fig.add_subplot(mid[0])]
    axc += [fig.add_subplot(mid[i], sharey=axc[0]) for i in (1, 2)]
    panel_drug(axc)
    panel_long(fig.add_subplot(bot[0]))
    panel_linkage(fig.add_subplot(bot[1]))

    saved = save(fig, "FigS5_response", provenance={
        "source": "db/ra_rx.duckdb::gli_response, bulk_samples, bulk_state_scores, response_assoc_state, "
                  "strap_drug_specific, activity_longitudinal, activity_longitudinal_pooled, strap_sample_match",
        "scripts": ["scripts/32_response_association.py", "scripts/34_disease_activity.py",
                    "scripts/36_genetic_load_index.py", "scripts/37_strap_match.py",
                    "scripts/39_strap_drug_specific.py", "scripts/figS5_response.py"],
        "transformations": [f"cohorts with >= {MIN_PER_ARM} patients per response class", "DerSimonian-Laird",
                            "Mann-Whitney two-sided per compartment", "single-cohort rho CI by Fisher z",
                            f"linkage used = margin >= {MARGIN_MIN} and stability >= {STAB_MIN}"]})
    print(saved.report)


if __name__ == "__main__":
    main()
