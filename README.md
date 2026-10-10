# The synovial stroma carries little of the polygenic risk of rheumatoid arthritis

Code and result tables for:

> Xu Z, Cheng J, Chen X, Luo J, Guo X, Zhu X. *The synovial stroma carries little of the polygenic risk of rheumatoid arthritis: implications for genetics-led drug discovery.* Manuscript submitted to *Rheumatology* (2026).

The study maps RA genome-wide association signals onto synovial cell states of two single-cell atlases. It partitions RA heritability across single-nucleus chromatin of RA synovium and across chromatin of cultured fibroblast-like synoviocytes (FLS), using height, OA and Dupuytren's contracture as stromal positive controls. It then relates the genetic map to 492 synovial biopsies and to approved drug targets. All inputs are public data.

## Checking the results without running anything

| Path | Contents |
|---|---|
| `results/key_numbers.csv` | Every number quoted in the manuscript, with the result table it comes from (`source`) |
| `results/supplementary_tables/` | Supplementary Tables S1–S24 as CSV. `README_contents.csv` lists their titles and `README_codes.csv` explains the trait, annotation and column codes |
| `results/Supplementary_Tables.xlsx` | Written by `scripts/91_supp_tables.py` when the pipeline is run |
| `results/figure_source/` | Additional tables plotted in the figures but not in the Supplementary Tables (Parquet): UMAP coordinates, state pseudobulk expression, gene-level MAGMA results, Open Targets associations, Xenium niche summaries |

The atlas codes used in the tables are `discovery` (E-MTAB-11791; 54 states) and `replication` (GSE243917, RA only; 76 states).

## Repository layout

```
scripts/   analysis pipeline, in run order (numbered), and one script per figure (fig*.py)
py/        shared helpers: archive.py (results database with provenance), cohorts.py (bulk cohort loaders),
           store.py (download and cache helpers), figstyle.py (figure conventions)
config/    supplementary-table numbering
results/   result tables (see above)
```

Each analysis script writes its result tables, with provenance (script, time, input datasets, input checksums), to a local DuckDB database (`db/ra_rx.duckdb`, created on first run). Downstream scripts and the figure scripts read from it.

## Where each result comes from

| Item | Script(s) |
|---|---|
| Fig. 1 (design, discovery atlas) | `fig1_atlas.py` |
| Fig. 2 (scDRS genetic load) | `fig2_genetic_load.py` |
| Fig. 3 (replication across methods, atlases, ancestries) | `fig3_replication.py` |
| Fig. 4 (chromatin heritability) | `fig4_chromatin.py` |
| Fig. 5 (biopsies and disease activity) | `fig4_activity.py` |
| Fig. 6 (drug targets) | `fig7_targets.py` |
| Supplementary Figs S1–S6 | `figS1_markers.py`, `figS2_scdrs_robustness.py`, `figS3_replication_atlas.py`, `fig6_spatial.py` (S4), `figS5_response.py`, `figS6_chromatin_sensitivity.py` |
| Table S1 (discovery-atlas states) | `10_atlas_micheroli.py`, `11_fine_states.py`, `12_curate_states.py`, `30_state_signatures.py` |
| Table S2 (replication-atlas states) | `14_atlas_gupta.py` |
| Tables S3, S9 (likely causal genes, enrichment) | `24_causal_genes.py` |
| Table S4 (cohorts) | `01_salmon_cohort.py`, `02_gene_matrix.py`, `31_score_bulk.py` |
| Table S5 (state signatures) | `30_state_signatures.py` |
| Tables S6, S17 (disease activity) | `34_disease_activity.py` |
| Table S7 (scDRS) | `20_magma.py`, `21_scdrs.py`, `22_scdrs_summary.py`, `26_eas_pipeline.sh`, `28_dupuytren_control.sh` |
| Table S8 (MAGMA gene-property analysis) | `23_magma_celltype.py` |
| Table S10 (genetic map matrix) | `25_genetic_map_summary.py` |
| Table S11 (power of the gene-level map) | `29_spikein_power.py` |
| Tables S12, S13 (S-LDSC, primary) | `27_sldsc_chromatin.py` (`27_run_all.sh`) |
| Tables S14, S15 (S-LDSC sensitivity, comparator, fine-mapping) | `27b_sldsc_sensitivity.py` (`27b_run_all.sh`) |
| Table S16 (pathotype) | `32_response_association.py` |
| Table S18 (partial correlations) | `34b_activity_composition.py` |
| Table S19 (treatment response) | `32_response_association.py`, `34_disease_activity.py`, `36_genetic_load_index.py`, `38_remodeling.py`, `39_strap_drug_specific.py` |
| Table S20 (STRAP biopsy linkage) | `37_strap_match.py` |
| Tables S21–S24 (targets) | `70_ra_drug_targets.py`, `72_ot_targets.py`, `73_target_revision.py` |
| Spatial niches (Supplementary Fig. S4) | `60_xenium_extract.py`, `61_xenium_annotate.py`, `62_xenium_niches.py` |
| Numbers in the text; Supplementary Tables workbook | `90_key_numbers.py`, `91_supp_tables.py` |

## Data

All inputs are public. Download scripts place them under `raw/`, `refs/` and `tools/`; none of these folders is part of the repository.

| Data | Accession / source | Obtained by |
|---|---|---|
| Discovery atlas (25 inflammatory-arthritis biopsies) | ArrayExpress E-MTAB-11791 | `00_download_sc_spatial.sh` |
| Replication atlas and snATAC (11 RA donors) | GEO GSE243917 | `00_download_upgrade.sh`, `00_download_sldsc.sh` |
| Cultured FLS chromatin (Ge et al. 2021) | GEO GSE163548 | `00_download_sldsc.sh` |
| Roadmap primary T and B cells (H3K27ac, H3K4me1, H3K4me3) | Roadmap Epigenomics E034, E032 | `27b_sldsc_sensitivity.py` |
| Bulk synovial RNA-seq: R4RA, STRAP, PEAC | ArrayExpress E-MTAB-11611, E-MTAB-13733, E-MTAB-6141 (FASTQ from ENA/SRA) | `01_salmon_cohort.py` |
| Bulk synovial cohorts used as deposited | GEO GSE198520, GSE45867, GSE24742 | `00_download_geo_cohorts.sh` |
| Xenium 5k, 17 pre-treatment sections | Zenodo 15058262 | `00_download_sc_spatial.sh` |
| RA GWAS (Ishigaki et al. 2022) | GWAS Catalog GCST90132223 (EUR), GCST90132224 (EAS), GCST90132226 (seropositive) | `00_download_refs.sh`, `00_download_upgrade.sh` |
| FinnGen release 13 (RA, serotypes, OA, height, Dupuytren's contracture) | FinnGen public summary statistics | `00_download_refs.sh`, `27_sldsc_chromatin.py` |
| LD score references, 1000G Phase 3 EUR | Zenodo 10515792 | `00_download_sldsc.sh` |
| HapMap3 SNP list; UCSC hg38→hg19 liftOver chain | LDSC resources; UCSC | `00_download_sldsc.sh` |
| scDRS gene annotation | figshare 19312583 | `00_download_refs.sh` |
| GENCODE v46 transcriptome | GENCODE | `00_download_refs.sh` |
| MAGMA v1.10 with 1000G EUR/EAS references and NCBI38 gene locations | https://cncr.nl/research/magma/ | manual, into `tools/magma/` and `tools/magma_eas/` |
| Open Targets Platform (release 26.09, accessed 24 September 2026) | GraphQL API | `72_ot_targets.py` |
| ChEMBL (release 37) | REST API | `70_ra_drug_targets.py` |

Supplementary files of published papers are read from `refs/`. They are not redistributed here, so download them from the publishers:

| Folder | Paper | Files |
|---|---|---|
| `refs/ishigaki2022_supp/` | Ishigaki et al., *Nat Genet* 2022;54:1640–51 | `41588_2022_1213_MOESM4_ESM.xlsx` (fetched by `00_download_upgrade.sh`) |
| `refs/amp1_natimmunol2019_supp/` | Zhang et al., *Nat Immunol* 2019;20:928–42 | `41590_2019_378_MOESM4_ESM.xlsx` |
| `refs/amp2_nature2023_supp/` | Zhang et al., *Nature* 2023;623:616–24 | `41586_2023_6708_MOESM5_ESM.xlsx` to `MOESM8` |
| `refs/r4ra_natmed2022_supp/` | Rivellese et al., *Nat Med* 2022;28:1256–68 | `41591_2022_1789_MOESM3_ESM.xls` to `MOESM7` |
| `refs/strap_natcommun2025_supp/` | Lewis et al., *Nat Commun* 2025;16:5374 | `41467_2025_60987_MOESM3_ESM.xlsx` to `MOESM10` |

Reference-file checksums (MD5) as used in the analysis: `w_hm3.snplist` e1372a59749eb1f92f7f6931c075f5ac (1,217,312 lines); `hg38ToHg19.over.chain.gz` ff3031d93792f4cbb86af44055efd903. The Zenodo 10515792 archives are MD5-verified by the download script.

## Software

- Main environment, `environment.yml`: Python 3.11 with scanpy 1.11, scvi-tools 1.4, scDRS 1.0.2, DuckDB 1.5, statsmodels 0.15, pandas 2.3.
- MAGMA v1.10 and Salmon v1.11.4.
- S-LDSC runs in a separate environment, `environment_ldsc.yml` (Python 3.11, numpy 1.26, pandas 2.1, bitarray 2.9), using the Python 3 implementation of LDSC in PolyFun (commit `8b17b4e`, cloned by `00_download_sldsc.sh`).
  - Set `PYTHON` to the main interpreter and `LDSC_PYTHON` to the LDSC interpreter.
  - Optional overrides: `LDSC_REF_DIR`, `HM3_SNPLIST`, `LIFTOVER_CHAIN`, `SCRATCH`, `H2_WORKERS`.

The figure scripts use figkit, the authors' in-house figure layout and QA package, which is not distributed. Every value they plot is in `results/`, so the figures can be redrawn with any plotting library.

## Running the pipeline

```bash
conda env create -f environment.yml          # main environment
conda env create -f environment_ldsc.yml     # S-LDSC environment
export PYTHON=$(conda run -n ra-rx which python) LDSC_PYTHON=$(conda run -n ra-rx-ldsc which python)

bash scripts/00_download_refs.sh && bash scripts/00_download_sc_spatial.sh && bash scripts/00_download_upgrade.sh
bash scripts/00_download_sldsc.sh && bash scripts/00_download_geo_cohorts.sh
$PYTHON scripts/00_register_clinical.py
for acc in E-MTAB-11611 E-MTAB-13733 E-MTAB-6141; do $PYTHON scripts/01_salmon_cohort.py $acc; done
$PYTHON scripts/02_gene_matrix.py
# discovery and replication atlases
for s in 10_atlas_micheroli 11_fine_states 12_curate_states 13_state_pseudobulk 14_atlas_gupta; do $PYTHON scripts/$s.py; done
# gene-level genetic mapping (see each script's header for arguments)
for s in 20_magma 21_scdrs 22_scdrs_summary 23_magma_celltype 24_causal_genes; do $PYTHON scripts/$s.py; done
bash scripts/26_eas_pipeline.sh && bash scripts/28_dupuytren_control.sh
$PYTHON scripts/25_genetic_map_summary.py && $PYTHON scripts/29_spikein_power.py
# chromatin-level heritability partitioning
bash scripts/27_run_all.sh && bash scripts/27b_run_all.sh
# biopsies, disease activity and treatment response
for s in 30_state_signatures 31_score_bulk 32_response_association 34_disease_activity 34b_activity_composition \
         36_genetic_load_index 37_strap_match 38_remodeling 39_strap_drug_specific; do $PYTHON scripts/$s.py; done
# spatial niches, drug targets, numbers and tables
for s in 60_xenium_extract 61_xenium_annotate 62_xenium_niches 70_ra_drug_targets 72_ot_targets 73_target_revision \
         90_key_numbers 91_supp_tables; do $PYTHON scripts/$s.py; done
```

Each script header gives its inputs, outputs and options. Notes on running:
- The full pipeline downloads several hundred GB of reads and reference data.
- Bulk re-quantification uses the first 20 million read pairs of each run, as stated in the Methods.
- The S-LDSC regressions need about 3.5 GB of memory each.

## Licence

- Code: MIT licence (`LICENSE`).
- Result tables in `results/`: CC BY 4.0.
- Third-party data remain under their original licences and terms of use.

## Contact

- Xingrong Guo (gxrdl@126.com)
- Xiaohu Zhu (zhuxh0831@163.com)

Taihe Hospital, Hubei University of Medicine, Shiyan, China.
