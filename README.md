<<<<<<< HEAD
# Schizophrenia Subject-Specific Morphometric Graph Starter

Starter implementation for:

**Explainable Subject-Specific Morphometric Graph Learning for Schizophrenia Classification from Structural MRI**

## Research-safe workflow

1. Download raw T1 MRI + metadata.
2. Freeze a cohort manifest.
3. Run one pinned FreeSurfer version for every dataset.
4. QC FreeSurfer outputs before graph construction.
5. Extract DK68 regional morphometry from vertex-level FreeSurfer outputs.
6. Build conventional MSN matrices.
7. Build MIND matrices with the official MIND implementation.
8. Create aligned subject arrays.
9. Run conventional ML baselines with nested CV.
10. Run subject-level GNN graph classification with outer/inner CV.
11. Explain only outer-test subjects using the model that did not train on them.
12. Lock the development pipeline before external validation.

## Important caveats

- `participants.tsv` fields differ by dataset. Inspect columns and diagnostic values before creating the cohort.
- The official MIND repository states that its code was tested on older FreeSurfer output. Validate file names and ROI names when using FreeSurfer 7.4.1.
- This starter ranks edges by absolute similarity to create equal-density graphs for architecture comparison. Positive-only and weighted-edge choices should be preregistered/evaluated as ablations.
- Any graph density, harmonization, feature selection, scaling, or hyperparameter choice for final results must use training data only.
- Do not tune models on the external validation dataset.

## Suggested first run

Use COBRE as the development cohort. While waiting for NITRC approval, download OpenNeuro ds000030 and use only 1–3 subjects to verify that FreeSurfer/MIND code runs; do not use external-set classification performance for model selection.

## Python environment

Create a virtual environment, then install PyTorch for your CPU/GPU using the official PyTorch selector. After PyTorch is installed:

```bash
pip install -r requirements.txt
```

## MIND

```bash
git clone https://github.com/isebenius/MIND.git external/MIND
```

## Pipeline

```bash
python scripts/01_make_cohort.py \
  --bids-dir data/raw/ds000030 \
  --participants data/raw/ds000030/participants.tsv \
  --dataset ds000030 \
  --diagnosis-column diagnosis \
  --case-values SCHZ Schizophrenia \
  --control-values CONTROL Control \
  --out data/processed/cohort_ds000030.csv

bash scripts/02_run_freesurfer.sh \
  data/raw/ds000030 \
  data/derivatives/freesurfer \
  /ABSOLUTE/PATH/license.txt \
  ds000030

python scripts/03_qc_freesurfer.py \
  --subjects-dir data/derivatives/freesurfer \
  --dataset ds000030 \
  --out data/processed/freesurfer_qc_ds000030.csv

python scripts/04_extract_roi_features.py \
  --subjects-dir data/derivatives/freesurfer \
  --dataset ds000030 \
  --out-dir data/processed/roi_features

python scripts/05_build_msn.py \
  --roi-dir data/processed/roi_features \
  --out-dir data/processed/msn

python scripts/06_build_mind.py \
  --subjects-dir data/derivatives/freesurfer \
  --dataset ds000030 \
  --mind-repo external/MIND \
  --out-dir data/processed/mind

python scripts/07_prepare_arrays.py \
  --cohort data/processed/cohort_ds000030.csv \
  --roi-dir data/processed/roi_features \
  --msn-dir data/processed/msn \
  --mind-dir data/processed/mind \
  --out data/processed/datasets/ds000030.npz

python scripts/08_ml_nested_cv.py \
  --dataset data/processed/datasets/ds000030.npz \
  --representation roi \
  --model svm \
  --out results/ml_roi_svm

python scripts/08_ml_nested_cv.py \
  --dataset data/processed/datasets/ds000030.npz \
  --representation mind_edges \
  --model svm \
  --out results/ml_mind_edges_svm

python scripts/10_train_gnn.py \
  --dataset data/processed/datasets/ds000030.npz \
  --representation mind \
  --model gatv2 \
  --out results/gnn_mind_gatv2
```

Read every generated CSV before moving to the next phase.
=======
# schizophrenia-morphometric-graph
>>>>>>> 2c24a11483f18fb76f3fcccd4686e1b68b22f45e
