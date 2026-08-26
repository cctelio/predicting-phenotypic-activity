# Plan: Standardized Confound-Controlled Activity Prediction Study (BBBC036v1 + JUMP-CP)

Statistical methodology follows Ash, Wognum et al., "Practically Significant Method Comparison
Protocols for Machine Learning in Small Molecule Drug Discovery," J. Chem. Inf. Model. 2025, 65,
9398-9411 (`548496_Fulltext.pdf`), referred to below as **[Ash2025]**. Reference implementation:
https://github.com/polaris-hub/polaris-method-comparison.

## Folder structure

```
chemical_surrogate_study/
├── PLAN.md                              # this file
├── notebooks/
│   ├── 01_prepare_bbbc036v1.ipynb
│   ├── 02_prepare_jumpcp.ipynb
│   ├── 03_train_and_compare.ipynb       # parameterized: run once per dataset
│   └── 04_plots_and_tables.ipynb
├── data/
│   ├── bbbc036v1_standardized.parquet
│   └── jumpcp_standardized.parquet
├── results/
│   ├── bbbc036v1/
│   └── jumpcp/
└── figures/
```

## Standardized dataset schema (output of notebooks 1 and 2, input to notebook 3)

One row per compound:

| column | meaning |
|---|---|
| `compound_id` | BROAD_ID (BBBC) / JCP2022 ID (JUMP) |
| `SMILES` | for structure representations downstream |
| `is_active`, `is_active_and_toxic`, `is_active_not_toxic` | 3 binary targets (see note below) |
| `activity_map`, `activity_p`, `toxicity_map`, `toxicity_p` | continuous stats, for EDA. `toxicity_map`/`toxicity_p` are null for compounds where the toxicity test wasn't run (inactive compounds — see note below) |
| `morph_0 ... morph_N` | compound-level morphology profile (cell-count excluded for BBBC; already excluded for JUMP's `ALL` product) |
| `batch_id` | grouping column for the plate/batch split (plate-batch for BBBC, `Metadata_Source` for JUMP) |
| `cloome_split` | `train`/`val`/`test` for BBBC, null for JUMP |

Structure-derived representations (MorganFP, PhysChem, LogP, chemical k-medoids split) are
deterministic functions of `SMILES` and are computed on the fly in notebook 3, not stored here.

### Note: toxicity is only tested among active compounds

Realization that simplifies both notebooks: toxicity is only ever *meaningful* among active
compounds — BBBC already showed inactive-and-toxic is vanishingly rare (3/10,680 = 0.03%). So
rather than running the toxicity `copairs` test on the whole population, we run it **only on
active perturbations + sampled negative controls**, and default every inactive compound's toxicity
to `False` without testing it. The three targets become:

- `is_active` — over the whole population, as before.
- `is_active_and_toxic` = `is_active AND is_toxic` (inactive compounds are `False` by construction, untested).
- `is_active_not_toxic` = `is_active AND NOT is_toxic` (inactive compounds are `False` by construction).

Restricted to actives-only, `is_active_and_toxic` and `is_active_not_toxic` would be exact mirror
images of each other (redundant) — but evaluated over the *whole* population (as specified here)
they're genuinely different problems: `is_active_and_toxic` is a rare event embedded in a large
population dominated by easy true-negatives (inactives), while `is_active_not_toxic` is the common
case. This is why there's no longer a standalone `is_toxic` target — it's absorbed into these two
composites, and this is also exactly what makes the JUMP-CP toxicity fetch cheap: we only ever need
cell-count data for ~12,220 active compounds' wells (+ sampled negcons), not the full ~114,239.

---

## Notebook 1: `01_prepare_bbbc036v1.ipynb`

Direct port of the validated logic already run in `notebooks/08_activity_confound_study.ipynb`:

1. Load official CLOOME/CellCLIP split (`suinleelab/CellCLIP`, `datasplit1-{train,val,test}.csv`) → fixes population (10,680 compounds) and gives `cloome_split`.
2. Load `data/train_set_30kcpds_normalized_profiles.csv.gz`, restrict to official population + DMSO controls.
3. TVN preprocessing with `Cells_Number_Object_Number` **excluded from the feature set before whitening** (not dropped after — whitening would otherwise mix it into every other feature).
4. `is_active`: `copairs` permutation test, **cosine** distance, full (cell-count-excluded) TVN profile, `neg_sameby=["plate"]`.
5. **Toxicity — restricted to active compounds + negative controls** (revised for consistency with notebook 2's new design, see the schema note above): `copairs` permutation test, **euclidean** distance (cosine on a scalar collapses to `sign(x)` — degenerate, this was the bug caught earlier that gave a false 99.5% toxic rate), on the per-plate-DMSO-centered cell-count value alone, run only on wells belonging to `is_active == True` compounds + sampled DMSO controls. Compounds with `is_active == False` get `is_active_and_toxic = False` / `toxicity_map,toxicity_p = null` by construction, not tested. (This changes nothing about BBBC's already-computed, already-analyzed results in practice — only 3/10,680 compounds were previously found inactive-and-toxic, so restricting the test doesn't lose meaningful information — but it makes both notebooks methodologically identical, which is worth having for the paper's methods section.)
6. `is_active_and_toxic` = `is_active AND is_toxic`; `is_active_not_toxic` = `is_active AND NOT is_toxic`.
7. `batch_id`: connected components of co-occurring plates (compounds span 2-8 replicate plates, not confined to one — 95 batches).
8. Aggregate morphology to compound level (median across wells), write `data/bbbc036v1_standardized.parquet`.

No open questions — this is a direct, already-validated reuse, with the one small toxicity-scoping revision above.

## Notebook 2: `02_prepare_jumpcp.ipynb`

Revised to (a) pull activity p-values live from **JUMP-RR** rather than reusing the old cache, and
(b) exploit the "toxicity is only tested among actives" simplification above: the toxicity fetch
now only ever needs the ~12,220 active compounds' wells (+ sampled negcons), not the full ~114,239
— which is what actually keeps this notebook cheap, more so than any subsampling trick. Morphology
(genuinely expensive — hundreds of feature dimensions per compound) is still deferred until the
final population is fixed. This mirrors and extends the pattern already used in the earlier
`jump_data_prep.py` script (found and reused below), which already did a polars
`scan_parquet(...).join(ids_df, how="semi")` lazy semi-join against the S3-hosted harmonized
profile rather than downloading it in full.

**Step 1 — JUMP-RR activity p-values (live fetch, not cache).**
Fetch `https://zenodo.org/api/records/20496083/files/compound.parquet/content`
(columns `Perturbation`, `JCP2022`, `Corrected p-value`). Drop duplicate `JCP2022`, drop rows with
no p-value. Derive `inchikey14 = Perturbation[:14]`, `is_active = Corrected p-value <= 0.05`,
`Metadata_JCP2022 = str(JCP2022)`. This *is* "the provided p-values" — no `copairs` rerun.
Expect ~114,740 compounds, ~10.7% active (~12,220).

**Step 2 — SMILES.**
Fetch `https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz`, merge on
`Metadata_JCP2022`, extract the SMILES column, drop rows with no SMILES.

**Step 3 — CLOOME/CellCLIP dedup.**
Drop compounds whose `inchikey14` matches BBBC036v1's official population (recomputed directly
from notebook 1's official split — self-contained, not the external pickle the old script used).
**Confirmed 501 of 114,740 overlap** against the full official train+val+test population (the old
script only deduped against the narrower "train" partition — using the full official population
here is the more conservative, safer choice given uncertainty about exactly which partition
CLOOME/CellCLIP's own pretraining touched). ~114,239 compounds remain.

**Step 4 — toxicity, restricted to active compounds + negative controls only (~12,220 compounds,
not the full ~114,239).**
From the Interpretable product (`COMPOUND/v1.0/profiles_var_mad_int.parquet`, needed since the
harmonized `ALL` profile has no cell-count column at all). **Confirmed via direct schema/data
inspection**:
  - Schema has `Metadata_Source`, `Metadata_Plate`, `Metadata_Well`, `Metadata_JCP2022` + 3,180
    feature columns, including **`Nuclei_Number_Object_Number`** and `Cytoplasm_Number_Object_Number`
    (both confirmed present — using `Nuclei_Number_Object_Number` as planned).
  - This column is **already variance/MAD-normalized** (values like -1.18, 0.92, not raw counts —
    the product name itself, `profiles_var_mad_int`, says as much). We still apply our own
    per-plate DMSO-centering on top (harmless if redundant, guarantees exact methodological
    consistency with BBBC regardless of what normalization JUMP already applied).
  - **No `Metadata_pert_type` column exists anywhere in JUMP's metadata** (checked `well.csv.gz`,
    `plate.csv.gz`, this profile's own schema) — the plan's original assumption was wrong. DMSO
    control wells are instead identified by **`Metadata_JCP2022 == "JCP2022_033924"`**, confirmed
    via its `compound.csv.gz` entry having SMILES `C[S+](C)[O-]` (DMSO). 93,552 DMSO wells exist
    in this product across all sources/plates.
  - Semi-join scan restricted to the `is_active == True` subset of the deduped `Metadata_JCP2022`
    IDs (~12,220), selecting only `Metadata_JCP2022`, `Metadata_Plate`, `Metadata_Well`,
    `Metadata_Source`, `Nuclei_Number_Object_Number`. Never touches the other ~3,000+ feature
    columns.
  - Also pull DMSO wells (`Metadata_JCP2022 == "JCP2022_033924"`) on the same plates as these
    active-compound wells (a plate-list semi-join, small since it's scoped to only the plates
    touched by ~12,220 compounds rather than all of JUMP).
  - Downsample DMSO controls to **≤190 per plate** via the same `sample_controls_per_batch` trick
    already used for BBBC, keeping the permutation test itself fast (BBBC's equivalent step ran in
    ~2s at full BBBC scale, and this is a comparable or smaller row count).
  - Run the toxicity `copairs` test: **euclidean** distance (singular-value distance, not an
    embedding distance — cosine would be degenerate here, same fix as BBBC), per-**plate**-DMSO-
    centered cell count, `neg_sameby=["Metadata_Plate"]` — **plate**, not source (see reasoning
    below). Produces `is_toxic` / `toxicity_map` / `toxicity_p` for the ~12,220 active compounds
    only. All inactive compounds get `is_toxic = False` (untested) / `toxicity_map,toxicity_p = null`.

**Step 5 — define the final population: all active ∪ inactive padding.**
No separate "all toxic" clause needed anymore — toxicity is now scoped to actives by construction,
so "all active" already includes every toxicity-tested compound.
- Include **every** compound with `is_active == True` (~12,220, all toxicity-tested in step 4).
- Pad with a sample of inactive compounds so the classifiers have a real negative class for the
  `is_active` task — **defaulting to matching the active count 1:1 (~12,000)**, giving roughly
  balanced classes overall (final population ≈ 24-25k). Stratify this padding sample
  proportionally across `Metadata_Source` (from the lightweight lookup below) so the source/batch
  split stays meaningful. **Flag if you want a different ratio.**
- A lightweight `Metadata_Source`-only lookup (2 narrow columns, no feature data) is still needed
  for the *inactive* compounds to enable this stratification, since step 4 only touched actives:
  `polars.scan_parquet(JUMP_ALL_HARMONY_PROFILE_URL).select(["Metadata_JCP2022", "Metadata_Source"]).unique(subset=["Metadata_JCP2022"], keep="first").collect()`,
  filtered to the inactive subset.

**Step 6 — fetch morphology, restricted to the final chosen population (~24-25k IDs).**
Same lazy semi-join + per-compound median-aggregation pattern as the original script:
`pl.scan_parquet(JUMP_ALL_HARMONY_PROFILE_URL).join(ids_df, on="Metadata_JCP2022", how="semi").group_by("Metadata_JCP2022").agg([median of each feature column])`,
`ids_df` now containing only the step-5 population instead of the full ~114,740 — this becomes
`morph_0...morph_N`. This is the one genuinely expensive fetch (hundreds of feature dimensions),
which is exactly why it's deferred until after the population is fixed.

**Step 7 — assemble and write.** Join activity + toxicity (null for inactives) + morphology +
`Metadata_Source` (as `batch_id`) + SMILES into the standard schema, write
`data/jumpcp_standardized.parquet`. No `cloome_split` column.

### Plate vs. Source — resolved, used for two different purposes
- **Toxicity `copairs` test's `neg_sameby`: `Metadata_Plate`.** Matching each well against DMSO
  controls on its *own physical plate* is the statistically correct, standard practice — mirrors
  BBBC exactly, and is almost certainly what JUMP-RR itself did internally to compute the activity
  p-values we're reusing in Step 1.
- **Downstream ML `batch_id` (the plate/batch-held-out split in notebook 3): `Metadata_Source`.**
  Coarser, lab-level — tests generalization across entirely different screening sites, a harder and
  more interesting test than holding out plates within the same lab, and matches your instruction
  and this project's prior convention (the earlier JUMP work already captured `primary_source_raw`
  this same way).

**Note for during build**: verify `jumpcp_cloome_bioactivity_cache.parquet` / `jumpcp_cellclip_cache.parquet` (already cached, 26.8MB each) are keyed by SMILES and cover the subsampled population before reusing — recompute only the gap if some SMILES are missing.

## Notebook 3: `03_train_and_compare.ipynb` — run once per dataset (parameterized by dataset path/name)

### 3.1 Representations
- `Morphology`: `morph_0...morph_N` from the standardized dataset, used as-is.
- `MorganFP`: radius=2, 2048 bits, non-chiral (matches the original BBBC study's definition) — from `SMILES`.
- `PhysChem`: full RDKit `Descriptors.CalcMolDescriptors` set — from `SMILES`.
- `LogP`: Crippen LogP, standalone — from `SMILES`.
- `CLOOME`, `CellCLIP`: reuse cached embeddings (BBBC: `bbbc036v1_cloome_bioactivity_cache.parquet` / `bbbc036v1_cellclip_cache.parquet`; JUMP: the two files above), keyed by SMILES, dedup cache index before reindexing (known duplicate-SMILES issue, already solved once).
- `MLP_CLOOME_arch`: **new**, per your request. Input = Morgan fingerprint matching CLOOME's own encoder spec (radius=3, 1024 bits, **chiral=True** — deliberately different from the plain `MorganFP` condition above, since the point is testing CLOOME's architecture minus its contrastive pretraining, not re-testing the same fingerprint). Architecture: 4 hidden layers (matching `CLOOME_MLP_HIDDEN_LAYERS=4`), dropout 0.4, AdamW weight_decay=1e-4, early stopping on validation AUC, `pos_weight` for class imbalance — the exact regularization recipe already validated via the 9-combo sensitivity sweep (all 9 combos landed within a tight 0.028 AUC band, so this choice is not a lucky pick).
- `CellCount` excluded throughout (defines the toxic label — circular).

No concatenation ablations (per earlier decision).

### 3.2 Splits
- **Chemical**: k-medoids (manual PAM, k=80 for BBBC; scale k proportionally for JUMP's ~24-25k, e.g. k≈185-190) on Tanimoto distance over `MorganFP`, clusters bin-packed into 5 folds.
- **Plate/batch**: GroupKFold on `batch_id`, bin-packed into 5 folds.
- **CLOOME official**: only for BBBC (`cloome_split` populated).

### 3.3 Statistics — corrected per [Ash2025]

- **5×5 repeated CV** (not 1×5) for chemical and plate splits on both datasets — 5 repeats × 5 folds = 25 samples per (representation, target, split), matching Guidelines 1's explicit recommendation for datasets in the 500-100,000 range (both ours qualify). Same fold assignments across all representations within each repeat (required for the paired/repeated-measures design).
- Per-fold hyperparameter selection: inner train/val split of the training fold for C-selection (matches [Ash2025]'s "comparable to one iteration of the inner loop of nested CV" recommendation) — already implemented this way, just needs 5x more repeats.
- **Repeated-measures ANOVA + Tukey HSD**, not vanilla independent-samples Tukey HSD: run `statsmodels.stats.anova.AnovaRM` (subject=repeat×fold identifier, within=representation, dv=auc) to get the correct within-subject error term, then feed that error term into the Tukey HSD pairwise comparisons rather than treating the 25 scores per representation as independent samples. Will consult the paper's reference implementation (`polaris-hub/polaris-method-comparison` on GitHub) during build to match their exact procedure rather than approximate it.
- **Cohen's d, computed pairwise** (fix from earlier): for each pairwise comparison, `d = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)` using only the two groups being compared — not a global pooled SD across all representations (which is what the earlier BBBC-only run used).
- **BBBC official split — explicit deviation, flagged transparently** (per [Ash2025] Guidelines 5's own "transparency about deviations" ethos): this is an externally-fixed single split used for comparability with the CLOOME/CellCLIP papers, not a size-driven choice — at 10,680 compounds it's within the range the paper would otherwise recommend 5×5 CV for. To still extract a repeated-measures structure from this one fixed split without retraining differently on it, bootstrap-resample the test set (~25 resamples, matching the 25-sample target from Guidelines 1) and feed those into the same repeated-measures ANOVA + Tukey HSD machinery as the CV splits, so all three split types go through one consistent statistical pipeline (no DeLong anywhere, per your instruction).
- Fold-balance diagnostic: report per-fold class prevalence (already collected as `test_prevalence`) for visual inspection, per [Ash2025] Section 3.1.3's explicit recommendation for advanced/grouped splits.
- Lower performance bound: implicit in AUC itself (0.5 = no-skill baseline), satisfies Guidelines 3's "null model" lower-limit recommendation without extra computation.

### 3.4 Models
Linear logistic regression (all representations) + the one `MLP_CLOOME_arch` condition. No other MLP variants in the main comparison (the earlier FP+PhysChem MLP robustness check already answered "does nonlinearity/concatenation help" — no — and stays as a documented side result, not part of this main comparison).

### 3.5 Outputs → `results/<dataset>/`
- `cv_scores.csv` (25 rows × representation × target × split — the raw repeated-measures samples)
- `tukey_hsd.csv` (pairwise comparisons with corrected p-values, RM-ANOVA-based CI, Cohen's d)
- `official_split_bootstrap.csv` (BBBC only)

## Notebook 4: `04_plots_and_tables.ipynb`

Reads both datasets' `results/` folders. Implements [Ash2025] Guidelines 4's recommended visualizations:
- **Simultaneous confidence interval plot** (Figure 7 style): forest plot per target/split, best-performing representation in blue with dashed reference lines, statistically-equivalent representations in gray, significantly-worse in red.
- **MCSim plot** (Figure 8 style): heatmap of all pairwise mean differences, sorted by average performance, color = effect size (Cohen's d), star annotations = significance level.
- Summary tables (Table 1 style): CI of pairwise mean differences, one table per target/split/dataset.

---

## Decisions on the previously-open items

You asked me to just decide these rather than keep them open:

1. **JUMP final population: all active (~12,220, toxicity-tested) ∪ inactive padding matching the active count 1:1 (~12,000, untested/defaulted non-toxic)**, giving a roughly class-balanced population of ~24-25k total, per Notebook 2 Step 5. Inactive padding is stratified proportionally across `Metadata_Source`. **Flag if you want a different active:inactive ratio.**
2. **Cell-count column: `Nuclei_Number_Object_Number`** — confirmed present in the Interpretable product's schema (along with `Cytoplasm_Number_Object_Number`). Note: this column is already variance/MAD-normalized by JUMP, not raw counts; we still apply our own per-plate DMSO-centering on top for methodological consistency with BBBC.
3. **k-medoids k for JUMP: ~190.** Proportional scaling from BBBC's k=80 at ~10,680 compounds (ratio ≈ 1 cluster per 133 compounds) applied to ~24-25k compounds gives ≈184-190; will finalize the exact number once the final population size is known after Step 5.
4. **Yes, consult the paper's reference GitHub implementation** (`polaris-hub/polaris-method-comparison`) for the exact RM-ANOVA-into-Tukey-HSD mechanics before coding notebook 3's statistics — cheap (just reading their code), and removes any doubt about matching the paper's actual procedure rather than my own reconstruction of it.
