"""62_xenium_niches.py — spatial niches of genetically loaded vs decoupled states (Fig 6).

Cells: Xenium cells with an assigned compartment (Unresolved / low_counts excluded as centres and neighbours).
Neighbourhood: 20 nearest cells within the same patient (Euclidean on centroids, µm); composition over the
54 atlas states + Mast. Niches: MiniBatchKMeans (k=10) on neighbourhood compositions (rows sum to 1).
Genetic-load map: each state carries its scDRS z (ra_eur_noMHC); a cell's "neighbourhood genetic load" = mean z
of its 20 neighbours.

Tables:
  xenium_niche_profile      niche x state: mean neighbourhood fraction (niche identity)
  xenium_state_niche        state x niche: fraction of the state's cells in each niche
  xenium_patient_niche      patient x niche fractions + clinical (remission status)
  xenium_niche_response     per niche: remission vs non-remission (Mann-Whitney on patient fractions)
  xenium_cells_niche        per cell: niche, neighbourhood genetic load (parquet view)
  xenium_state_load_niche   per state: scDRS z vs mean neighbourhood genetic load (all neighbours, and
                            cross-compartment neighbours only, which removes the trivial same-type effect)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_parquet_view, write_table  # noqa: E402

K_NN, K_NICHE = 20, 10


def main():
    c = read("SELECT * FROM xenium_cells WHERE compartment NOT IN ('Unresolved', 'low_counts')")
    c["state"] = np.where(c.compartment == "Mast", "Mast", c.state)
    states = sorted(c.state.unique())
    sidx = {s: i for i, s in enumerate(states)}
    load = read("scdrs_state_load").query("trait == 'ra_eur_noMHC'").set_index("state").assoc_mcz
    c["gl"] = c.state.map(load)

    comp = np.zeros((len(c), len(states)), dtype="float32")
    nb_load = np.full(len(c), np.nan, dtype="float32")
    nb_load_x = np.full(len(c), np.nan, dtype="float32")   # cross-compartment neighbours only (no self-type effect)
    for p, d in c.groupby("patient"):
        xy = d[["x_centroid", "y_centroid"]].to_numpy()
        nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(xy)
        _, ind = nn.kneighbors(xy)
        ind = ind[:, 1:]
        codes = d.state.map(sidx).to_numpy()[ind]
        rows = d.index.to_numpy()
        for j in range(K_NN):
            np.add.at(comp, (rows, codes[:, j]), 1.0 / K_NN)
        g = d.gl.to_numpy()[ind]
        nb_load[rows] = np.nanmean(g, axis=1)
        other = d.compartment.to_numpy()[ind] != d.compartment.to_numpy()[:, None]
        with np.errstate(invalid="ignore"):
            nb_load_x[rows] = np.nansum(np.where(other, g, 0), axis=1) / other.sum(axis=1)
    c["nb_load"], c["nb_load_x"] = nb_load, nb_load_x

    km = MiniBatchKMeans(n_clusters=K_NICHE, random_state=0, batch_size=20000, n_init=5).fit(comp)
    c["niche"] = km.labels_
    prof = pd.DataFrame(comp).groupby(c.niche.values).mean()
    prof.columns = states
    # name niches by their two most over-represented states relative to the tissue average
    enr = prof / (comp.mean(axis=0) + 1e-6)
    names = {n: f"N{n}: " + " + ".join(enr.loc[n].sort_values(ascending=False).index[:2]) for n in prof.index}
    c["niche_name"] = c.niche.map(names)
    P = prof.rename(index=names).rename_axis(index="niche", columns="state").stack().rename("mean_fraction").reset_index()
    P["enrichment"] = enr.rename(index=names).stack().values
    write_table(P, "xenium_niche_profile", script=__file__, datasets=["Zenodo 15058262"],
                note=f"kNN={K_NN}, MiniBatchKMeans k={K_NICHE} on neighbourhood state fractions")

    write_parquet_view(c[["cell", "patient", "niche_name", "nb_load", "nb_load_x"]], "xenium_cells_niche",
                       script=__file__, datasets=["Zenodo 15058262"], note="per-cell niche + neighbourhood genetic load")
    sn = pd.crosstab(c.state, c.niche_name, normalize="index").stack().rename("fraction").reset_index()
    write_table(sn, "xenium_state_niche", script=__file__, datasets=["Zenodo 15058262"])

    clin = read("clin_xenium")
    clin["patient"] = clin.patient_id.astype(str)
    pn = pd.crosstab(c.patient, c.niche_name, normalize="index")
    pnl = pn.stack().rename("fraction").reset_index().merge(clin[["patient", "response", "treatment", "pathotype"]],
                                                             on="patient")
    write_table(pnl, "xenium_patient_niche", script=__file__, datasets=["Zenodo 15058262"])
    rows = []
    for n, d in pnl.groupby("niche_name"):
        a, b = d[d.response == "remission"].fraction, d[d.response == "non-remission"].fraction
        u, p = stats.mannwhitneyu(a, b) if len(a) and len(b) else (np.nan, np.nan)
        rows.append(dict(niche=n, mean_remission=a.mean(), mean_nonremission=b.mean(), n_rem=len(a), n_non=len(b),
                         p=p))
    NR = pd.DataFrame(rows)
    write_table(NR, "xenium_niche_response", script=__file__, datasets=["Zenodo 15058262"],
                note="patient-level niche fractions, remission vs non-remission (borderline excluded), Mann-Whitney")

    sl = c.groupby("state").agg(n=("state", "size"), gl=("gl", "first"), nb_load=("nb_load", "mean"),
                                nb_load_x=("nb_load_x", "mean"), compartment=("compartment", "first")).reset_index()
    ok = sl.dropna()
    rho, p = stats.spearmanr(ok.gl, ok.nb_load)
    rho_x, p_x = stats.spearmanr(ok.gl, ok.nb_load_x)
    sl["rho"], sl["p"], sl["rho_xcomp"], sl["p_xcomp"] = rho, p, rho_x, p_x
    write_table(sl, "xenium_state_load_niche", script=__file__, datasets=["Zenodo 15058262", "Ishigaki 2022"],
                note="state genetic load (scDRS z) vs mean genetic load of its spatial neighbours")
    pat = c.groupby("patient").nb_load.mean().rename("mean_nb_load").reset_index().merge(clin, on="patient")
    u2, p2 = stats.mannwhitneyu(pat[pat.response == "remission"].mean_nb_load,
                                pat[pat.response == "non-remission"].mean_nb_load)

    pd.set_option("display.width", 220)
    print(pd.Series(names).to_string())
    print(pn.round(3).join(clin.set_index("patient").response).to_string())
    print(NR.round(3).to_string(index=False))
    print(f"spatial self-segregation of genetic load: rho(state z, neighbour z)={rho:.2f} p={p:.2g}; "
          f"cross-compartment neighbours only: rho={rho_x:.2f} p={p_x:.2g}")
    print(sl.groupby("compartment")[["gl", "nb_load", "nb_load_x"]].median().round(2).to_string())
    print(pat[["patient", "response", "pathotype", "mean_nb_load"]].sort_values("mean_nb_load").round(2).to_string(index=False))
    print(f"patient mean neighbourhood genetic load, remission vs non-remission: MW p={p2:.3f}")


if __name__ == "__main__":
    main()
