"""12_curate_states.py — manual curation of fine states (Fig 1 labels) from 11_fine_states.py subclusters.

Each subcluster (lineage_group, leiden id) gets a final `state` label or is excluded, based on the AMP2 marker-set
match (atlas_fine_state_map) plus the top-12 Wilcoxon markers (reviewed 2026-09-23). Reasons are inline so the
labels can be audited/revised; re-run 21_scdrs.py etc. after any change.

Output: db/objects/atlas/micheroli_states.h5ad (obs: state, state_group, lineage_group; excluded cells removed)
DuckDB: atlas_state_curation (subcluster -> state / exclusion + reason), atlas_cells_state
"""
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from archive import OBJECTS, write_object, write_table  # noqa: E402

X = None  # exclude
# (group, leiden) -> (state label | None, reason)
CURATION = {
    ("T_NK", "0"): ("NK IFN-response", "AMP2 NK-8 match; ISG15/IFI6"),
    ("T_NK", "1"): ("T GZMK+ HLA-DR+ memory", "CCL5/NKG7/GZMK/HLA-DR; AMP2 CD4+ GZMK+ memory"),
    ("T_NK", "2"): ("NK CD56dim cytotoxic", "AMP2 NK-2; NKG7/CST7"),
    ("T_NK", "3"): ("T CD4+ IL7R+ memory", "IL7R/CD69 with FOS/FOSB stress genes; weak AMP2 match"),
    ("T_NK", "4"): ("NK CD56bright", "AMP2 NK-5; GZMK/GZMA"),
    ("T_NK", "5"): ("T CD8+ ZNF683+ resident memory", "CD8A/CD8B/ZNF683/HOPX"),
    ("T_NK", "6"): (X, "8 cells, mitochondrial RNA only"),
    ("T_NK", "7"): ("T CD4+ Treg", "AMP2 CD25-high Treg; FOXP3"),
    ("T_NK", "8"): ("T CD4+ naive", "AMP2 CD4+ naive"),
    ("T_NK", "9"): ("T CD4+ Tph CXCL13+OX40+", "CXCL13/MAF/TNFRSF4/NR3C1; AMP2 CD4+ OX40+NR3C1+"),
    ("T_NK", "10"): ("T CD4+ GPR183+ memory", "IL7R/GPR183; AMP2 IL7R+CD161+ ILC match rejected (T lineage)"),
    ("T_NK", "11"): ("T CD161+ innate-like", "AMP2 Innate-like; KLRB1/RORA"),
    ("T_NK", "12"): ("T CD4+ Tph", "AMP2 CD4+ Tph match"),
    ("T_NK", "13"): ("T gd/NK-like TRDC+", "TRDC/TYROBP/FCER1G/GNLY; AMP2 Vdelta2 (weak margin)"),
    ("T_NK", "14"): (X, "HBA/C1QB/LYZ: erythrocyte/myeloid contamination"),
    ("B_plasma", "0"): ("B unswitched memory", "AMP2 B-1; IGHD/IGHM"),
    ("B_plasma", "1"): ("B activated CD69+", "NR4A2/CD55/TNFRSF13C; merged with c2"),
    ("B_plasma", "2"): ("B activated CD69+", "FOS/CD69/CD83 immediate-early; merged with c1"),
    ("B_plasma", "3"): ("B ABC CD11c+", "AMP2 B-5 ABC (strong)"),
    ("B_plasma", "4"): ("B switched memory", "BANK1/AHNAK/POU2F2; AMP2 B-0 (weak)"),
    ("B_plasma", "5"): ("B switched memory", "ribosomal-high variant of c4"),
    ("B_plasma", "6"): ("pDC", "AMP2 B-4 GC-like match rejected: state signature PLD4/LILRA4/IRF8/CLIC3/JCHAIN = pDC"),
    ("B_plasma", "7"): ("Plasma cell", "MZB1/XBP1/PRDM1"),
    ("B_plasma", "8"): (X, "17 cells with COL4A1/COL4A2: endothelial contamination"),
    ("Myeloid", "0"): ("Mac TIMD4+ LYVE1+ resident", "TIMD4/LILRB5/MARCO"),
    ("Myeloid", "1"): ("Mac MERTK+ LYVE1+", "AMP2 M-0"),
    ("Myeloid", "2"): ("Mac APOE+ lipid", "APOE/APOC1/NR1H3"),
    ("Myeloid", "3"): ("DC3", "AMP2 M-9"),
    ("Myeloid", "4"): ("Mono IL1B+ FCN1+", "AMP2 M-7 (strong)"),
    ("Myeloid", "5"): ("Mac MERTK+ SELENOP+ FOLR2+", "AMP2 M-1; merged with c10"),
    ("Myeloid", "6"): ("Mac SPP1+", "AMP2 M-4; merged with c9"),
    ("Myeloid", "7"): (X, "cycling (MKI67/TOP2A/STMN1); excluded from state-level tests"),
    ("Myeloid", "8"): ("Mac GPNMB+ HLA-high", "GPNMB/IL4I1/HLA-DR"),
    ("Myeloid", "9"): ("Mac SPP1+", "SPP1/FABP5; merged with c6"),
    ("Myeloid", "10"): ("Mac MERTK+ SELENOP+ FOLR2+", "SELENOP/F13A1; merged with c5"),
    ("Myeloid", "11"): ("Mac hypoxic glycolytic", "MT2A/LDHA/HK2/ERO1A/PGK1"),
    ("Myeloid", "12"): ("Mac STAT1+ CXCL10+", "AMP2 M-6 (strong)"),
    ("Myeloid", "13"): ("DC2", "AMP2 M-10"),
    ("Myeloid", "14"): (X, "IL32/CD3D/TRAC: T-cell doublets"),
    ("Myeloid", "15"): (X, "AMP2 pDC score but no specific genes survive (DCN/COL3A1 ambient); true pDC = B_plasma c6"),
    ("Myeloid", "16"): ("DC4", "AMP2 M-11"),
    ("Myeloid", "17"): ("DC LAMP3+ mreg", "AMP2 M-14"),
    ("Stromal", "0"): ("Fib CXCL12+ HLA-DR+ sublining", "CXCL12/PTGDS/CD74/HLA-DRB1; AMP2 F-5/F-6"),
    ("Stromal", "1"): ("Fib CD34+ sublining", "AMP2 F-2"),
    ("Stromal", "2"): ("Fib MMP2+ THBS2+ sublining", "SERPINF1/MMP2/THBS2/COMP"),
    ("Stromal", "3"): ("Fib CTSK+ sublining", "CTSK/EPAS1/MDK; weak AMP2 match"),
    ("Stromal", "4"): ("Fib SCARA5+ TNXB+ sublining", "SCARA5/TNXB/MGP/CRTAC1"),
    ("Stromal", "5"): (X, "mitochondrial-high low-quality"),
    ("Stromal", "6"): ("Fib NOTCH3+ sublining", "AMP2 F-7 (strong)"),
    ("Stromal", "7"): ("Fib PRG4+ CLIC5+ lining", "AMP2 F-0"),
    ("Stromal", "8"): ("Fib LOXL2+ glycolytic", "LOXL2/TGFBI/PKM/ENO1"),
    ("Stromal", "9"): ("Fib POSTN+ sublining", "AMP2 F-3"),
    ("Stromal", "10"): ("Fib CHI3L2+ SOD2+ lining", "CHI3L2/SOD2/MT/TREM1"),
    ("Stromal", "11"): ("Fib COMP+ POSTN+ sublining", "COMP/FNDC1/POSTN (Bhamidipati 2026 treatment-resistant niche)"),
    ("Stromal", "12"): ("Fib PRG4+ lining", "AMP2 F-1; merged with c13"),
    ("Stromal", "13"): ("Fib PRG4+ lining", "55 cells, IFN-high lining; merged with c12"),
    ("Stromal", "14"): ("Mural", "AMP2 Mu-0 (strong)"),
    ("Stromal", "15"): ("Fib CXCL12-hi sublining", "AMP2 F-6"),
    ("Stromal", "16"): ("Fib CTHRC1+ LTBP2+ sublining", "LTBP2/CTHRC1; myofibroblast-like"),
    ("Endothelial", "0"): ("EC capillary SPARC+", "AMP2 E-0"),
    ("Endothelial", "1"): ("EC venous LIFR+", "AMP2 E-1; merged with c3"),
    ("Endothelial", "2"): ("EC capillary COL4A1+", "COL4A1/COL15A1/ADAMTS9"),
    ("Endothelial", "3"): ("EC venous LIFR+", "SOCS3/ribosomal; merged with c1"),
    ("Endothelial", "4"): ("EC venous HLA-DR+", "IGFBP7/HLA-DR/CD74/HAPLN3"),
    ("Endothelial", "5"): ("EC venous ICAM1+", "AMP2 E-2"),
    ("Endothelial", "6"): ("EC arterial", "AMP2 E-3 (strong)"),
    ("Endothelial", "7"): ("EC lymphatic", "AMP2 E-4 (strong)"),
    ("Endothelial", "8"): (X, "S100A8/PTPRC: immune doublets"),
    ("Endothelial", "9"): (X, "POSTN/TAGLN/MYL9 with VWF: endothelial-mural doublets"),
    ("Endothelial", "10"): (X, "cycling"),
    ("Endothelial", "11"): (X, "COL1A1/COL3A1/DCN: fibroblast doublets"),
    ("Endothelial", "12"): (X, "S100A8/S100A9/CXCL8/G0S2: neutrophil contamination"),
}
STATE_GROUP = {"T_NK": "T/NK", "B_plasma": "B/plasma", "Myeloid": "Myeloid", "Stromal": "Stromal",
               "Endothelial": "Endothelial"}


def main():
    fine = sc.read_h5ad(OBJECTS / "atlas" / "micheroli_fine.h5ad")
    key = list(zip(fine.obs.lineage_group.astype(str), fine.obs.fine_cluster.astype(str).str.split("-c").str[-1]))
    missing = sorted(set(key) - set(CURATION))
    if missing:
        raise SystemExit(f"uncurated subclusters: {missing}")
    fine.obs["state"] = [CURATION[k][0] for k in key]
    cur = pd.DataFrame([(g, c, s if s else "EXCLUDED", r) for (g, c), (s, r) in CURATION.items()],
                       columns=["lineage_group", "leiden", "state", "reason"])
    cnt = pd.Series(key).value_counts()
    cur["n_cells"] = [int(cnt.get((g, c), 0)) for g, c in zip(cur.lineage_group, cur.leiden)]
    write_table(cur, "atlas_state_curation", script=__file__, datasets=["E-MTAB-11791"],
                note="manual curation of 11_fine_states subclusters; EXCLUDED = doublet/contaminant/low-quality/cycling")

    keep = fine.obs.state.notna()
    s = fine[keep.values].copy()
    s.obs["state"] = s.obs.state.astype("category")
    s.obs["state_group"] = s.obs.lineage_group.astype(str).map(STATE_GROUP)
    s.obs.loc[s.obs.state == "pDC", "state_group"] = "Myeloid"  # pDC co-clustered with B/plasma (shared JCHAIN/IRF4)
    s.obs["state_group"] = s.obs.state_group.astype("category")
    o = s.obs[["sample", "lineage_group", "state_group", "state"]].reset_index(names="cell")
    write_table(o, "atlas_cells_state", script=__file__, datasets=["E-MTAB-11791"])
    f = OBJECTS / "atlas" / "micheroli_states.h5ad"
    s.write_h5ad(f, compression="gzip")
    write_object(f, "atlas_micheroli_states", script=__file__, datasets=["E-MTAB-11791"], n_rows=s.n_obs,
                 note=f"{s.obs.state.nunique()} curated states; {int((~keep).sum())} cells excluded")
    print(f"kept {s.n_obs} cells in {s.obs.state.nunique()} states; excluded {int((~keep).sum())}")
    print(s.obs.groupby(["state_group", "state"], observed=True).size().to_string())


if __name__ == "__main__":
    main()
