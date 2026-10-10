"""figS1_markers.py — Figure S1: full marker dot plot of the 54 atlas states.

Rows: the top-2 signature genes of every state (state_signatures rank 1-2; genes shared by two states are shown
once, at their first state: 108 state-gene pairs -> 98 genes), ordered by compartment and state as in Fig. 1d, gene
symbols in italics coloured by the compartment of the state they mark. Columns: the 54 states, numbered as in
Fig. 1d/e (numbers coloured by compartment; the key on the right names each number). Dot size = % of the state's
cells in which the gene is detected; colour = mean log-normalised expression min-max scaled per gene across the 54
states (magma, light = low). Light lines separate compartments.
Sources: atlas_cells_state (state order), state_signatures, state_pseudobulk_long.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read  # noqa: E402
from figstyle import C, COMP_COLOR, COMP_ORDER, figure, fk, pt, save  # noqa: E402

SMAX = 30                                   # marker area (pt^2) at 100% detected


def states_in_order():
    """Same numbering as Fig. 1: compartments in COMP_ORDER, states by cell count (descending)."""
    st = read("SELECT state_group, state, count(*) AS n FROM atlas_cells_state GROUP BY 1, 2")
    st["state_group"], st["state"] = st.state_group.astype(str), st.state.astype(str)
    st["_c"] = st.state_group.map({c: i for i, c in enumerate(COMP_ORDER)})
    st = st.sort_values(["_c", "n"], ascending=[True, False]).reset_index(drop=True)
    st["num"] = np.arange(1, len(st) + 1)
    return st


def main():
    st = states_in_order()
    sig = read("SELECT state, gene, rank FROM state_signatures WHERE rank <= 2")
    genes, owner = [], []
    for stn, grp in zip(st.state, st.state_group):
        for g in sig[sig.state == stn].sort_values("rank").gene:
            if g not in genes:
                genes.append(g)
                owner.append(grp)
    pb = read("state_pseudobulk_long")
    pb = pb[pb.gene.isin(genes) & pb.state.isin(st.state)]
    mean = pb.pivot(index="gene", columns="state", values="mean_lognorm").loc[genes, st.state]
    pct = pb.pivot(index="gene", columns="state", values="pct_detected").loc[genes, st.state]
    lo, hi = mean.min(axis=1), mean.max(axis=1)
    scaled = mean.sub(lo, axis=0).div(hi - lo + 1e-9, axis=0)

    fig = figure("double", 230, kind="supp")
    gs = fig.add_gridspec(1, 2, width_ratios=[118, 36])
    ax = fig.add_subplot(gs[0])
    ng, ns = len(genes), len(st)
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ns), indexing="ij")
    cm = mpl.colors.ListedColormap(fk.cmap()(np.linspace(0.97, 0.2, 256)))      # magma, light = low
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=pct.to_numpy().ravel() * SMAX, c=scaled.to_numpy().ravel(), cmap=cm,
                    vmin=0, vmax=1, linewidths=0)
    ax.set_yticks(range(ng), genes, fontsize=pt("fine"), fontstyle="italic")
    for lab, g in zip(ax.get_yticklabels(), owner):
        lab.set_color(COMP_COLOR[g])
    ax.set_xticks(range(ns), [str(k) for k in st.num], fontsize=pt("fine"), rotation=90)
    ax.tick_params(axis="x", labeltop=True, labelbottom=True, top=False, bottom=False)
    ax.tick_params(axis="y", length=0)
    for t, g in zip(ax.xaxis.get_major_ticks(), st.state_group):
        t.label1.set_color(COMP_COLOR[g])
        t.label2.set_color(COMP_COLOR[g])
    ax.set_xlim(-0.6, ns - 0.4)
    ax.set_ylim(ng - 0.4, -0.6)
    for comp in COMP_ORDER[:-1]:
        xi = np.where(st.state_group == comp)[0].max() + 0.5
        yi = max(i for i, g in enumerate(owner) if g == comp) + 0.5
        ax.axvline(xi, color=C.grey_light, lw=0.5, zorder=0)
        ax.axhline(yi, color=C.grey_light, lw=0.5, zorder=0)
    ax.set_xlabel("State (number as in Fig. 1D, E)", fontsize=pt("annot"))

    # right column: number -> state key, then the two scales
    right = gs[1].subgridspec(2, 1, height_ratios=[1, 0.16])
    ak = fig.add_subplot(right[0])
    ak.axis("off")
    ak.set_xlim(0, 1)
    row = 0
    for comp in COMP_ORDER:
        s = st[st.state_group == comp]
        ak.text(0, row, comp, fontsize=pt("annot"), fontweight="bold", color=COMP_COLOR[comp], va="center")
        row += 1
        for k, name in zip(s.num, s.state):
            ak.text(0.085, row, f"{k}", fontsize=pt("fine"), va="center", ha="right", color=COMP_COLOR[comp])
            ak.text(0.13, row, name, fontsize=pt("fine"), va="center")
            row += 1
    ak.set_ylim(row - 0.5, -0.5)                                   # the key fills its column top to bottom
    al = fig.add_subplot(right[1])
    al.axis("off")
    handles = [al.scatter([], [], s=p * SMAX, c=C.grey_dark, linewidths=0, label=f"{int(p * 100)}%")
               for p in (0.25, 0.5, 0.75, 1.0)]
    al.legend(handles=handles, title="Cells detected", loc="upper left", ncol=2, frameon=False,
              fontsize=pt("annot"), title_fontsize=pt("annot"), handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0)
    cb = fig.colorbar(sc, ax=al, location="bottom", fraction=0.35, aspect=9, pad=0.02)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=pt("annot"))
    cb.set_label("Scaled mean expression", fontsize=pt("annot"))

    if len(sys.argv) > 2 and sys.argv[1] == "--draft":            # throw-away exploration, never archived
        print(fk.save(fig, Path(sys.argv[2]), strict=False).report)
        return
    saved = save(fig, "FigS1_markers", provenance={
        "source": "db/ra_rx.duckdb::atlas_cells_state, state_signatures, state_pseudobulk_long",
        "scripts": ["scripts/12_curate_states.py", "scripts/13_state_pseudobulk.py", "scripts/30_state_signatures.py",
                    "scripts/figS1_markers.py"],
        "transformations": ["top-2 signature genes per state (rank <= 2); shared genes shown once",
                            "colour = per-gene min-max of state mean log-normalised expression",
                            "size = fraction of the state's cells with the gene detected"],
        "n": f"{ns} states, {ng} genes (E-MTAB-11791)"})
    print(saved.report)


if __name__ == "__main__":
    main()
