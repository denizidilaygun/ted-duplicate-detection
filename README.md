# Similarity-Enhanced Duplicate Detection for TED Procurement Data

A unified similarity-based framework for (1) detecting near-duplicate records in EU procurement data and (2) auditing near-duplicate memorisation in generative AI models trained on tabular data.

This repository accompanies a Master's thesis submitted to the Department of Data Science & Society at Tilburg University (May 2026).

---

## Overview

The framework addresses two challenges in a single similarity-based pipeline:

**RQ1 — Entity Resolution.** A hybrid classifier combining string-similarity features (Levenshtein, Jaccard, Jaro-Winkler), structured features (value ratio, CPV match, country match), and Sentence-BERT semantic embeddings detects near-duplicate procurement records in the EU Tenders Electronic Daily (TED) dataset.

**RQ2 — Generative AI Auditing.** The same similarity framework is repurposed as an audit metric for two tabular generative models (CTGAN and TVAE), revealing near-duplicate memorisation that exact-match auditing misses entirely.

---

## Repository Structure

```
.
├── rq1_entity_resolution/   # RQ1 pipeline: preprocessing -> blocking -> features -> models
├── rq2_genai_audit/         # RQ2 pipeline: GenAI training -> synthesis -> audit
├── disparate_impact/        # Per-country fairness analysis
├── eda/                     # Exploratory data analysis scripts
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Dataset

The framework is demonstrated on the **EU Tenders Electronic Daily (TED) Contract Award Notices** dataset for 2023, publicly available from the EU Open Data Portal:

https://data.europa.eu/euodp/en/data/dataset/ted-csv

The dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. It contains administrative procurement records with no personally identifiable information.

This repository **does not include the dataset**. Users must download it separately from the EU Open Data Portal.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/denizidilaygun/ted-duplicate-detection.git
cd ted-duplicate-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10 or higher.

---

## Reproducing the Results

### RQ1: Entity Resolution Pipeline

```bash
cd rq1_entity_resolution

# 1. Preprocess the raw TED data
python preprocessing.py

# 2. Generate candidate pairs via CPV blocking
python candidate_pairs.py

# 3. Compute string-similarity and structured features
python feature_engineering.py

# 4. Inject synthetic duplicates (v3 perturbation profile)
python synthetic_injection.py

# 5. (Optional) Compute SBERT semantic similarity
python sbert_embeddings.py

# 6. Train and evaluate classifiers
python real_model_training.py --input annotated_gold_standard.csv

# 7. Evaluate on real annotated pairs
python evaluate_on_real.py

# 8. Generate confusion matrices for in-distribution + OOD evaluation
python make_confusion_matrices.py
```

### RQ2: Generative AI Audit Pipeline

```bash
cd rq2_genai_audit

# 1. Apply cardinality reduction for tractable GenAI training
python prepare_ted_for_genai.py

# 2. Train CTGAN (approximately 18 minutes on consumer hardware)
python train_ctgan.py

# 3. Train TVAE (approximately 18 seconds)
python train_tvae.py

# 4. Generate synthetic records
python generate_synthetic.py

# 5. Run memorisation audit (exact-match + similarity-based near-duplicate)
python audit_memorization.py

# 6. Evaluate distributional fidelity (SDV + KS + TVD)
python evaluate_fidelity.py

# 7. Combine RQ2 metrics
python rq2_summary.py
```

### Disparate Impact Analysis

```bash
cd disparate_impact
python disparate_impact_analysis.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{aygun2026similarity,
  title  = {Similarity-Enhanced Duplicate Detection: A Framework for
            Auditing Generative AI Memorization in Tabular Data},
  author = {Aygun, Deniz Idil},
  school = {Tilburg University, Department of Data Science \& Society},
  year   = {2026}
}
```

---

## License

The code in this repository is released under the MIT License (see `LICENSE`).

The TED data referenced by this code is released by the Publications Office of the European Union under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## Acknowledgments

This work was conducted under the supervision of Dr. Gorkem Saygili (Tilburg University).
