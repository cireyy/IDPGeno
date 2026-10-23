# Fold-local preprocessing audit

Reviewed baseline: `44cf888e7251d0f17981bfde43ff35b659bb6f62`.

## Change and API contract

`build_dataset(config)` now always loads raw IDPs. After each stratified split,
`build_fold_datasets` computes population mean/std (`ddof=0`) on training rows
only, using float64 accumulation and float32 stored statistics. Standard deviations
below `1e-8` become 1. The same fitted values transform both folds. Separate subset
views share the large genomic arrays without overwriting the base IDP matrix or
other folds. The batched `__getitems__` path is overridden so DataLoader does not
bypass the transform. `normalize_idp: false` preserves raw values.

Direct construction with `normalize_idp=True` now **requires both** externally
fitted `idp_mean` and `idp_std`; the previous implicit full-table fit intentionally
raises an error. Use `normalize_idp=False` for raw data. Nonfinite IDPs, malformed
statistics, duplicate/missing sample IDs, and invalid/overlapping fold indices
fail explicitly. Missing-value imputation is not silently introduced.

Every checkpoint now includes `preprocessing` (enabled flag, ordered IDP names,
mean/std tensors and ddof) and `feature_info` (including ordered genes and functional
features). For inference, check/reorder the incoming IDP columns to exactly match
`preprocessing['idp_cols']`, check the genomic feature schema against `feature_info`,
and supply the saved statistics to `IDPGenoProcessedDataset`. Never refit on the
inference cohort. Old checkpoints do not contain this state and cannot be assumed
to support reproducible, fold-correct inference; retrain them.

The existing `gene_encoder.py` import was also corrected from `models` to
`src.models`; otherwise the documented module entry point could not be imported.

## Other preprocessing and evaluation checks

| Stage | Repository evidence | Assessment / required follow-up |
| --- | --- | --- |
| IDP scaling | Previously fit in dataset constructor before CV | Confirmed leakage; repaired here. |
| Class weighting | `build_loss_fn(subset_labels(dataset, train_idx))` | Computed from training labels only; no cross-fold fit found. |
| Model/optimizer state | Constructed inside each fold | Fresh state per fold; validation uses eval mode and no gradients. |
| LayerNorm, attention normalization, sequence pooling | Operate within a sample/representation | No cross-participant preprocessing fit found. |
| IDP feature list | All sample columns except ID and label | Schema detection, not statistical selection. Upstream selection and accidental outcome/proxy columns still require review. Freeze an explicit research feature list before evaluation. |
| Gene order and token dimensions | Derived from the entire input token table | Shared schema, not a numerical estimator. Only appropriate if the panel/order is fixed independently of held-out subjects. If held-out subjects determine gene/SNP inclusion, fit the panel on training data or use a prespecified external manifest. The current patch does not establish that provenance. |
| MAF and `within_gene_weight_a` | Read unchanged from processed CSV | Upstream derivation is absent. External reference MAF/fixed annotations are acceptable; cohort-estimated MAF, association-derived weights, and cohort filtering must be training-only and applied unchanged to validation. Do not blindly recompute from `geno_code` without confirming allele orientation and missing/padding semantics. |
| Imputation, GWAS/IDP/SNP selection, LD pruning, PCA, residualisation, harmonisation | No fitting implementations in this snapshot | Cannot certify these steps. If cohort-learned, nest their fitting inside training folds (and inner folds when tuning). Independently fixed external references can be reused. |
| Participant independence | StratifiedKFold splits rows; sample IDs now checked unique | Duplicate IDs fail. Related participants or repeated observations with different IDs still need family/group-aware splits and provenance checks. |
| Epoch selection | Best validation AUROC selected and reported on the same fold | Separate model-selection optimism remains; this patch does not make this an unbiased outer test estimate. Use inner validation for epoch/hyperparameter selection, then evaluate once on an untouched outer fold, or prespecify epochs. |
| Threshold | Metrics use a fixed 0.5 default | No threshold fitting found. The YAML threshold is currently not forwarded; outside the normalization fix. Any future optimized threshold must be selected on training/inner validation data. |

The repository contains processed examples, not the real research preprocessing
pipeline. This audit therefore fixes one confirmed issue; it does not certify
that all upstream or outcome-timing leakage has been excluded.

## Reproduce validation

From the repository root:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/test_fold_preprocessing.py
python -m src.training.train --config configs/configs.yaml
```

The tests cover shifted validation values, training-statistic invariance under
validation perturbation, independent fold state, constant columns, raw-data
preservation, disabled scaling, supplied statistics, invalid data/statistics/splits,
batched DataLoader behavior, five-fold training and checkpoint reload on a small
synthetic fixture, and scaling all five splits of the checked-in simulated CSVs.

## What simulated data can and cannot establish

The checked-in data have 1,000 samples (500 per class), 14 IDPs and 800,000 token
rows (100 genes, 8 tokens per gene). They can validate input compatibility,
fold isolation, numerical transformations, tensor shapes, training/validation
execution and checkpoint serialization. Their AUROC is a software smoke result,
not a reproduction of UKBB performance or an estimate of leakage-induced bias.

With authorized real data, rerun all affected models using saved identical splits,
seeds and hyperparameters for an initial paired old-vs-fixed comparison. Retain
out-of-fold predictions and report changes in AUROC, AUPRC, discrimination at a
prespecified threshold, and calibration with appropriate paired uncertainty.
Do not infer either the sign or magnitude of the change from this code review.
Then address epoch/hyperparameter selection using nested CV or an untouched test
set; tune all cohort-derived preprocessing only on the permitted training data.
Recompute main results, baselines using IDPs, ablations, confidence intervals,
subgroup results, model comparisons and downstream interpretation/FiLM attribution
that depend on retrained models. Audit upstream data provenance, relatedness and
baseline imaging/outcome timing first. The README's AUROC 0.834 and biological
claims are **not validated or revised by these simulated-data tests**.

## Executed verification (2026-10-23)

- `python -m pytest -q tests/test_fold_preprocessing.py`: **22 passed** (5.77 s).
- Full checked-in simulated dataset: all five folds completed with the default
  model and seed 42, `num_epochs=1`, `device=cpu`, `num_workers=0`, and one CPU
  thread. All five saved checkpoint means were independently checked against
  their training rows. Smoke AUROC mean 0.8183, fold SD 0.0249; this number has
  no interpretation as real-data performance or leakage effect size. The
  default ten-epoch experiment and real-data experiments were not run.
- Runtime: Python 3.12; torch 2.14.1+cu130 (CPU execution), NumPy 2.3.5,
  pandas 2.2.3, scikit-learn 1.8.0, pytest 9.1.1.
- `git diff --check`: passed.
