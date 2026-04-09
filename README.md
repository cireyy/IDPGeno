# IDPGeno: Brain-Structure-Modulated Genomic Modeling for Personalised Ischaemic Stroke Risk Prediction

This repository implements **IDPGeno**, a deep learning framework for ischaemic stroke risk prediction by jointly modelling genomic variants and brain imaging-derived phenotypes.

---

## Abstract

Ischaemic stroke (IS) risk arises from complex interactions between genetic susceptibility and environmental factors. While polygenic risk scores have improved population-level risk stratification, their predictive performance varies substantially at the individual level because the heterogeneous nature of environmental exposures and their biological consequences differ significantly across individuals. Modelling how environmental factors modulate genetic susceptibility remains challenging. In this work, we leverage brain structural integrity (i.e., an integrated effect of environmental impact on the brain) to capture gene–environment interactions for IS risk prediction. We propose IDPGeno, a deep learning framework that integrates genome-wide variants with baseline brain Imaging-Derived Phenotypes (IDPs). IDPGeno utilises a Feature-wise Linear Modulation mechanism to dynamically condition genetic representations on individual brain microstructural states, hypothesizing that compromised white matter integrity amplifies the penetrance of genetic risk. Benchmarking on the UK Biobank dataset ($N=472,569$), IDPGeno achieved an AUROC of 0.834, outperforming gene-only baselines (AUROC $\approx$ 0.65), traditional machine learning approaches (AUROC 0.75--0.79), and static multimodal fusion models (AUROC $\approx$ 0.81). Downstream analyses revealed that diffusion-derived phenotypes in the internal capsule and corona radiata significantly modulated the risk contribution of specific vascular genes (e.g., COL4A2, NOTCH3), confirming that IDPs serve as critical intermediate phenotypes. This study demonstrates that modelling the context-dependent interaction between inherited risk and brain structure provides a more accurate and biologically grounded approach for personalised IS risk prediction.

---

## Framework Overview

The IDPGeno framework consists of the following stages:

1. **Gene-level SNP representation**
   - SNPs are grouped into genes after preprocessing.
   - For each SNP token, the model combines:
     - genotype embedding
     - relative position embedding
     - functional/annotation feature embedding
   - SNP tokens within each gene are aggregated by **masked attention pooling** to obtain fixed-length gene embeddings.

2. **IDP encoding**
   - Baseline MRI-derived IDPs are encoded into a latent representation using an MLP.

3. **IDP-guided FiLM modulation**
   - The encoded IDP features generate gene-specific FiLM parameters:
     - scaling term `γ`
     - shifting term `β`
   - Each gene embedding is modulated as:
     `h_tilde = γ ⊙ h + β`

4. **Sequential genomic modelling**
   - The IDP-conditioned gene sequence is processed by a **MoE-Transformer** backbone.

5. **Risk prediction**
   - The pooled sequence representation is passed to a classification head for binary IS risk prediction.

---

## Data Source

The original study is based on data from the **UK Biobank (UKBB)**, a large-scale biomedical resource containing genetic, imaging, and health-related information from over 500,000 participants.

To access the real UK Biobank dataset, researchers must apply through the official portal:

[UK Biobank Access Application](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access)

Because the original data cannot be publicly released, this repository includes **simulated processed example data** with the same input structure expected by the model.

---

## Data Format

### 1. `src/data/processed_samples_simulated.csv`

A sample-level table containing:

- `participant_id`
- binary `label`
- 14 brain IDP features

Each row corresponds to one individual.

### 2. `src/data/processed_snp_tokens_simulated.csv`

A token-level table containing one row per:

- participant
- gene
- SNP token position within gene

Required columns include:

- `participant_id`
- `gene_id`
- `token_index_within_gene`
- `snp_id`
- `geno_code`
- `relative_pos`
- `maf`
- `within_gene_weight_a`
- `is_valid`

These fields are converted into the model inputs:

- `geno_ids`
- `pos_values`
- `func_values`
- `snp_mask`

---

