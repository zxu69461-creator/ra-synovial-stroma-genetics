"""24_causal_genes.py — where are the likely causal RA genes expressed? (MAGMA/scDRS-independent genetic evidence)

Likely causal genes from Ishigaki 2022 (Nat Genet; Supplementary Data MOESM4), following the AMP phase-2 definition
(Zhang 2023 Nature, Supplementary Methods): a gene is "likely causal" if
  (1) coding   : an RA risk allele (or variant in LD) is nonsynonymous in the gene          (ST-8)
  (2) finemap  : a fine-mapped variant with PIP > 0.5 lies in its exon / intron / UTR / splice site / upstream (ST-6)
  (3) gtex     : a GTEx eQTL colocalises with the RA signal (PP > 0.5, any tissue)             (ST-10)
  (4) blueprint: a Blueprint immune-cell eQTL/sQTL colocalises (PP > 0.5)                    (ST-9)
Blueprint covers immune cells only (possible bias towards immune expression) -> sensitivity set without
Blueprint-only genes. MHC genes (chr6:25-34 Mb) excluded.

Test per state (and per compartment = mean over its states): mean expression-specificity z (state_pseudobulk_long
spec_z) of causal genes vs 10,000 random gene sets matched on overall expression (deciles of mean expression across
states); empirical one-sided P (enrichment), BH-FDR over states.
Usage: python scripts/24_causal_genes.py [--atlas micheroli|gupta]
DuckDB: ra_causal_genes (gene, evidence...), causal_gene_state_enrichment (atlas, gene_set, state, ...)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import connect, read, write_table  # noqa: E402

XL = ROOT / "refs/ishigaki2022_supp/41588_2022_1213_MOESM4_ESM.xlsx"
N_PERM = 10000
GENIC = {"exonic", "intronic", "UTR5", "UTR3", "splicing", "upstream", "UTR5;UTR3", "exonic;splicing"}


def sheet(name, hdr_row):
    d = pd.read_excel(XL, name, header=None)
    cols = [f"{a}|{b}".replace("\n", " ") for a, b in zip(d.iloc[hdr_row - 1].ffill().astype(str),
                                                          d.iloc[hdr_row].astype(str))]
    d = d.iloc[hdr_row + 1:].copy()
    d.columns = cols
    return d.dropna(how="all")


def causal_genes():
    ev = {}

    def add(g, e):
        for x in re.split(r"[;,]", str(g)):
            x = re.sub(r"\(.*\)", "", x).strip()
            if x and x not in (".", "nan") and not x.startswith("LOC"):
                ev.setdefault(x, set()).add(e)
    s8 = sheet("ST-8", 4)
    for v in s8.filter(like="Amino acid").iloc[:, 0]:
        add(str(v).split(":")[0], "coding")
    s6 = sheet("ST-6", 5)
    pip = pd.to_numeric(s6.filter(like="|PIP").iloc[:, 0], errors="coerce")
    ann = s6.filter(like="Annotation|Gene").iloc[:, 1].astype(str)            # Annotation|Gene = region class
    name = s6.filter(like="Annotation|Gene name").iloc[:, 0]
    for g, a, p, chrom in zip(name, ann, pip, s6.iloc[:, 0]):
        if p > 0.5 and a in GENIC:
            add(g, "finemap")
    for sh, lab in [("ST-10", "gtex"), ("ST-9", "blueprint")]:
        d = sheet(sh, 4)
        pp = pd.to_numeric(d.filter(like="|PP").iloc[:, 0], errors="coerce")
        for g, p in zip(d.filter(like="Gene name").iloc[:, 0], pp):
            if p > 0.5:
                add(g, lab)
    G = pd.DataFrame([dict(gene=g, evidence=";".join(sorted(e)), n_evidence=len(e),
                           blueprint_only=(e == {"blueprint"})) for g, e in ev.items()])
    loc = pd.read_csv(ROOT / "tools/magma/NCBI38.gene.loc", sep="\t", header=None, usecols=[1, 2, 5],
                      names=["chr", "start", "gene"]).drop_duplicates("gene")
    G = G.merge(loc, on="gene", how="left")
    mhc = (G.chr.astype(str) == "6") & G.start.between(25_000_000, 34_000_000)
    return G[~mhc].reset_index(drop=True)


def enrichment(P, genes, rng):
    Z = P.pivot(index="gene", columns="state", values="spec_z")
    expr = P.pivot(index="gene", columns="state", values="mean_lognorm").mean(axis=1)
    expressed = P[P.pct_detected >= 0.10].gene.unique()
    Z, expr = Z.loc[expressed], expr.loc[expressed]
    bins = pd.qcut(expr.rank(method="first"), 10, labels=False)
    gs = [g for g in genes if g in Z.index]
    obs = Z.loc[gs].mean()
    pools = {b: Z.index[bins == b].to_numpy() for b in range(10)}
    need = bins.loc[gs].value_counts()
    null = np.empty((N_PERM, Z.shape[1]))
    Zv = Z.to_numpy()
    pos = {g: i for i, g in enumerate(Z.index)}
    for k in range(N_PERM):
        pick = np.concatenate([rng.choice(pools[b], n, replace=False) for b, n in need.items()])
        null[k] = Zv[[pos[g] for g in pick]].mean(axis=0)
    p = (np.sum(null >= obs.to_numpy(), axis=0) + 1) / (N_PERM + 1)
    zsc = (obs.to_numpy() - null.mean(axis=0)) / null.std(axis=0)
    return pd.DataFrame({"state": Z.columns, "n_genes": len(gs), "mean_spec_z": obs.to_numpy(), "enrich_z": zsc,
                         "p": p}), null, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="micheroli")
    a = ap.parse_args()
    G = causal_genes()
    write_table(G, "ra_causal_genes", script=__file__, datasets=["Ishigaki 2022 Nat Genet Supplementary Data"],
                note="AMP2-style likely causal genes: coding / fine-map PIP>0.5 genic / GTEx coloc / Blueprint coloc; MHC excluded")
    print(f"{len(G)} likely causal genes; evidence counts:",
          pd.Series([e for x in G.evidence for e in x.split(';')]).value_counts().to_dict(),
          f"blueprint-only {int(G.blueprint_only.sum())}")
    P = read("state_pseudobulk_long") if a.atlas == "micheroli" else read("gupta_state_pseudobulk_long")
    P["state_group"] = P.state_group.astype(str)
    grp = P.drop_duplicates("state").set_index("state").state_group
    rng = np.random.default_rng(20261002)
    rows = []
    for setname, genes in [("all", G.gene), ("no_blueprint_only", G[~G.blueprint_only].gene),
                           ("coding_or_finemap", G[G.evidence.str.contains("coding|finemap")].gene)]:
        R, null, obs = enrichment(P, set(genes), rng)
        R["fdr"] = multipletests(R.p, method="fdr_bh")[1]
        R["state_group"] = R.state.map(grp)
        R["gene_set"] = setname
        rows.append(R)
        # compartment level: mean over its states, same null draws
        for c, idx in R.groupby("state_group").groups.items():
            cols = [list(R.state).index(s) for s in R.loc[idx, "state"]]
            o = obs.iloc[cols].mean()
            nl = null[:, cols].mean(axis=1)
            rows.append(pd.DataFrame([dict(state=f"[compartment] {c}", state_group=c, gene_set=setname,
                                           n_genes=R.n_genes.iloc[0], mean_spec_z=o,
                                           enrich_z=(o - nl.mean()) / nl.std(),
                                           p=(np.sum(nl >= o) + 1) / (N_PERM + 1), fdr=np.nan)]))
    R = pd.concat(rows, ignore_index=True).assign(atlas=a.atlas)
    con = connect(read_only=True)
    try:
        have = "causal_gene_state_enrichment" in set(
            con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()
    if have:
        old = read("causal_gene_state_enrichment")
        R = pd.concat([old[old.atlas != a.atlas], R], ignore_index=True)
    write_table(R, "causal_gene_state_enrichment", script=__file__, datasets=["Ishigaki 2022", a.atlas],
                note=f"mean spec_z of causal genes vs {N_PERM} expression-matched random sets; one-sided")
    S = R[R.atlas == a.atlas]
    pd.set_option("display.width", 200)
    print(S[S.state.str.startswith("[compartment]")][["gene_set", "state", "n_genes", "enrich_z", "p"]]
          .round(4).to_string(index=False))
    print(S[~S.state.str.startswith("[")].groupby(["gene_set", "state_group"]).fdr.apply(
        lambda f: f"{(f < 0.05).sum()}/{len(f)}").unstack())


if __name__ == "__main__":
    main()
