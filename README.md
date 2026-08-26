# The limits of chemical surrogates for phenotypic drug discovery

Code and analysis notebooks for the paper (`main.tex`). Evaluates whether phenotype-pretrained
molecular encoders (CLOOME, CellCLIP) generalize as surrogates for phenotypic activity prediction,
under an evaluation protocol that controls for pretraining-boundary leakage and for the
activity/toxicity confound, on two Cell Painting screens (BBBC036v1, JUMP-CP).

## Layout

```
chemical_surrogate_study/
├── notebooks/     01-04, run in order (see below)
├── results/       CV scores, official-split results, Tukey HSD tables written by notebook 3
├── figures/       plots written by notebook 4
├── aime_style.py  shared matplotlib style used by notebook 4
├── environment.yml
└── main.tex       the paper
```

## Environment

```
conda env create -f environment.yml
conda activate chemical-surrogate-limits
```

Notebooks were developed and run against `python=3.12`, `rdkit>=2025.03`, `pytorch>=2.7`,
`scikit-learn>=1.7`, `pingouin>=0.5`, `statsmodels>=0.14`, `copairs>=0.5.1`. GPU is not required;
everything runs on CPU (CellCLIP's BERT-based embedding step in particular is CPU-bound and is the
slowest single step in the pipeline — expect on the order of tens of minutes for JUMP-CP's ~24k
compounds if those embeddings are not already cached, see below).

## Before running

Notebooks resolve their own paths from their working directory (`PROJECT_ROOT =
Path.cwd().parent.parent if Path.cwd().name == "notebooks" else Path.cwd()`), so no path editing
is needed as long as you run them from inside `chemical_surrogate_study/notebooks/` — Jupyter's
default working directory when opening a notebook from that folder.

CLOOME and CellCLIP embeddings are computed by notebook 3 itself via inference from the public
checkpoints (`anasanchezf/cloome`, `suinleelab/CellCLIP` on Hugging Face) — no precomputed cache is
required. On first use for a given data set, notebook 3 downloads each checkpoint, runs every
compound's SMILES through it, and writes the result to a local cache
(`data/_{dataset}_cloome_cache.parquet`, `data/_{dataset}_cellclip_cache.parquet`); later runs
reuse that cache. This step needs internet access and takes a while the first time — CellCLIP's
BERT-based inference in particular is CPU-bound and is the slowest single step in the pipeline,
on the order of tens of minutes for JUMP-CP's ~24k compounds. If a precomputed cache happens to
already exist under `PROJECT_ROOT / "outputs" / "full_pipeline"` (`{dataset}_cloome_bioactivity_
cache.parquet`, `{dataset}_cellclip_cache.parquet`), it's used as an optional local speedup instead
of recomputing.

## Data prerequisites

- **BBBC036v1 raw well-level profiles**: notebook 1 expects
  `PROJECT_ROOT / "data" / "train_set_30kcpds_normalized_profiles.csv.gz"` (the well-level,
  CellProfiler-derived profile table distributed with CPMolGAN, ~2 GB). This is not fetched
  automatically — download it from the CPMolGAN data release and place it at that path before
  running notebook 1. If a cached, already-TVN-corrected version
  (`outputs/full_pipeline/tvn_corrected_wells_nocellcount.parquet`) is present, notebook 1 will use
  it directly and skip reprocessing the raw CSV (saves ~20-25 minutes); otherwise it recomputes TVN
  from the raw CSV.
- **Everything else is fetched automatically**: notebooks 1 and 2 download the official
  BBBC036v1/CLOOME/CellCLIP train/validation/test split from the `suinleelab/CellCLIP` Hugging Face
  repo, and notebook 2 downloads JUMP-RR's corrected activity $p$-values from Zenodo and JUMP-CP
  metadata/profiles from the public Cell Painting Gallery S3 bucket. Notebook 3 downloads the
  CLOOME and CellCLIP model checkpoints as described above. All of this requires internet access
  on first run; results are cached to `data/` afterward.

## Running the notebooks

Run in numeric order. Each notebook writes its outputs to disk and can be re-run independently once
its inputs exist (later notebooks do not need earlier ones to still be "live" in memory).

1. **`01_prepare_bbbc036v1.ipynb`** — builds the standardized BBBC036v1 table
   (`data/bbbc036v1_standardized.parquet`): TVN-corrected, compound-level morphology, and the
   `is_active` / `is_active_and_toxic` / `is_active_not_toxic` labels from permutation testing.
   Slowest step is TVN correction from the raw CSV if no cache is present (~20-25 min); the
   permutation tests themselves are cached under `data/_bbbc_*_null_cache/` so repeated runs are
   fast.
2. **`02_prepare_jumpcp.ipynb`** — builds the standardized JUMP-CP table
   (`data/jumpcp_standardized.parquet`) the same way, reusing JUMP-RR's published activity
   $p$-values and computing toxicity ourselves. Involves downloading JUMP-CP profile/metadata
   parquet files from S3; expect this to be the longest-running notebook on a slow connection.
3. **`03_train_and_compare_bbbc036v1.ipynb`** and **`03_train_and_compare_jumpcp.ipynb`** — for
   each data set: computes (or loads a local cache of) CLOOME/CellCLIP embeddings, builds the
   chemical ($k$-medoids on Tanimoto distance) and batch/plate generalization splits, runs 5×5
   repeated grouped cross-validation plus (for BBBC036v1) the official split evaluation across all
   representations and targets, and runs the repeated-measures Tukey HSD comparison. Writes
   `results/{dataset}/*.csv`. This is the slow, compute-heavy step — the BBBC036v1 cross-validation
   loop alone took about 45-50 minutes single-threaded on the original development machine (plus
   the CLOOME/CellCLIP embedding computation on first run); JUMP-CP is larger and takes longer.
4. **`04_plots_and_tables.ipynb`** — reads `results/{bbbc036v1,jumpcp}/*.csv` and writes every
   figure (`figures/*.pdf`, `*.png`) and table (`tables/*.tex`) used in the paper. Fast (seconds).

## Reproducing the paper from released artifacts only

If you just want to regenerate the figures/tables without rerunning the full analysis, you only
need `results/{bbbc036v1,jumpcp}/*.csv` (released alongside the code, see main.tex) and notebook 4.
