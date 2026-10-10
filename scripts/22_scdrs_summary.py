"""22_scdrs_summary.py — Fig 2 summary tables from scdrs_group_state.

scdrs_state_load        per state x trait: assoc_mcz, assoc_mcp, BH-FDR within trait (+ compartment)
scdrs_compartment_load  per compartment x trait: median z, n states FDR<0.05
scdrs_trait_concordance Spearman correlation of state-level z between traits (replication / specificity)
"Genetic load" used downstream = assoc_mcz for ra_eur_noMHC (primary; MHC removed so a single locus cannot drive it).
"""
import sys
from pathlib import Path

from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import read, write_table  # noqa: E402

ORDER = ["ra_eur", "ra_eur_noMHC", "ra_eur_seropos", "fg_ra", "fg_ra_noMHC", "fg_ra_seropos", "fg_ra_seroneg",
         "fg_oa", "fg_height"]
g = read("scdrs_group_state").merge(read("SELECT DISTINCT state, state_group FROM atlas_cells_state"), on="state")
g["fdr"] = g.groupby("trait").assoc_mcp.transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_table(g, "scdrs_state_load", script=__file__, datasets=["Ishigaki 2022", "FinnGen R13", "E-MTAB-11791"],
            note="BH-FDR within trait over 54 states")
c = g.groupby(["trait", "state_group"]).agg(median_z=("assoc_mcz", "median"),
                                            n_fdr05=("fdr", lambda f: int((f < 0.05).sum())),
                                            n_states=("state", "size")).reset_index()
write_table(c, "scdrs_compartment_load", script=__file__, datasets=["Ishigaki 2022", "FinnGen R13"])
w = g.pivot_table(index="state", columns="trait", values="assoc_mcz")[ORDER]
cc = w.corr(method="spearman").rename_axis(index="trait_a", columns="trait_b").stack().rename("rho").reset_index()
write_table(cc, "scdrs_trait_concordance", script=__file__, datasets=["Ishigaki 2022", "FinnGen R13"])
print(c.pivot(index="state_group", columns="trait", values="median_z")[ORDER].round(2).to_string())
