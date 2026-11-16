# Interpretability and Performance Analysis of Fuzzy Rule-Based Models (FRBMs)

## Student Information
- **Student Name:** Bayram KÜLTER
- **Student ID:** 1306220123
- **Course:** Bulanık Mantık
- **GitHub Repository:** https://github.com/Bayraminadresi/fuzzy-logic-project

---

## Original Paper
- **Authors:** L. Pickering, K. Cohen, and B. De Baets
- **Title:** *A Narrative Review on the Interpretability of Fuzzy Rule-Based Models from a Modern Interpretable Machine Learning Perspective*
- **Journal:** International Journal of Fuzzy Systems (2025)
- **DOI:** `10.1007/s40815-025-02022-z`

---

## Problem Description
Fuzzy Rule-Based Models (FRBMs) are widely valued for combining linguistic transparency with non-linear reasoning. However, Pickering et al. (2025) demonstrate that interpretability in FRBMs is often claimed informally rather than evaluated systematically across stakeholder perspectives and model stages. In their illustrative case study (Section 7, Fig. 7), the authors evaluate a 12-rule FisPro classifier on the UCI Wine dataset using their **Three-Lens Framework** (*Scale*, *Audience*, *Stage*), highlighting critical limitations:
1. **Absence of Post-hoc Interpretability:** The baseline FisPro model lacks post-hoc rule importance ranking or compactness analysis (*Stage → Post-hoc* is rated *Poor/Red*).
2. **Partition Sensitivity:** Uniform grid-based Strong Fuzzy Partitions (SFPs) can misalign with skewed feature distributions, degrading classification accuracy when rules are heavily pruned to 4 rules per class.
3. **Nominal Class Defuzzification Bias:** Standard continuous centroid defuzzification over `[0, 2]` treats nominal class labels `{0, 1, 2}` as ordinal quantities, causing rules from Class 0 and Class 2 to average toward Class 1.

---

## Method
To address both structural reproduction and empirical improvement, the project follows a four-stage methodology:
1. **Structural Baseline Reproduction:** Reconstruct the FisPro-style Mamdani FRBM in Python with 4 input features, 3 linguistic terms per feature (`Low`, `Medium`, `High`), Ruspini Strong Fuzzy Partitions ($\sum_{j=1}^3 \mu_{i,j}(x) = 1$), Wang-Mendel rule induction, confidence-based pruning (`4 rules/class = 12 rules`), and centroid defuzzification.
2. **Three-Lens Interpretability Audit:** Quantitatively and qualitatively evaluate the model across *Scale* (local active rules per instance, global rule count), *Audience* (model designer, domain expert, end user), and *Stage* (pre-model, in-model, post-hoc).
3. **Controlled Extensions (K1, K2, K3):** Implement three targeted extensions derived from the review's open challenges and evaluate each under identical `5x5` Repeated Stratified Cross-Validation splits (`SEED = 42`).
4. **Statistical & Baseline Benchmarking:** Compare baseline and extended FRBMs against interpretable classical controls (`DecisionTreeClassifier(max_depth=3)` and `LogisticRegression(L2)`) using accuracy, macro-F1, rule count, execution time, and paired Wilcoxon signed-rank tests.

---

## Fuzzy System Architecture

### System Pipeline
1. **Input Variables (`FEATURES`):** `flavanoids`, `proline`, `color_intensity`, `alcohol` (4 chemical descriptors).
2. **Fuzzification (`src/fuzzy_engine.py`):** 3-term triangular Ruspini Strong Fuzzy Partitions (`Low`, `Medium`, `High`) with shoulder boundaries at the outer centers so membership degrees sum to $1.0$ across the entire domain.
3. **Rule Induction & Pruning (`src/rule_induction.py`):**
   - Candidate rules generated via Wang-Mendel single-pass mapping from `(n_samples, 4, 3)` membership tensors; conflicting antecedents resolved by highest firing product degree.
   - Rule weight computed as fuzzy confidence: $w_r = \frac{\sum_{y_i = c_r} a_r(x_i)}{\sum_{i=1}^N a_r(x_i)}$, where $a_r(x_i) = \min_{f=1 \dots 4} \mu_{f, \text{ant}_{r,f}}(x_{i,f})$.
   - Top 4 rules per class selected by $\text{support} \times \text{confidence}$ (`12 rules` total).
4. **Inference & Defuzzification (`src/fuzzy_engine.py`):**
   - Weighted Mamdani min t-norm firing: $f_r(x) = w_r \cdot a_r(x)$.
   - *Baseline Defuzzification:* Mamdani min-clipping and max-aggregation over a 201-point grid on `[0, 2]`, followed by centroid calculation and nearest-integer rounding.
   - *Proposed Vote Defuzzification (K3):* Class-wise maximum firing aggregation ($\hat{y} = \arg\max_{c \in \{0,1,2\}} \max_{r: c_r = c} f_r(x)$) with training-set majority-class fallback when $\max_r f_r(x) = 0$.

### Repository Structure
```text
fuzzy-logic-project/
├── README.md                              # Project documentation (this file)
├── LICENSE                                # Open-source repository license
├── requirements.txt                       # Python dependencies and minimum versions
├── .gitignore                             # Ignored temporary/cache files
├── src/
│   ├── __init__.py                        # Package initializer
│   ├── rule_induction.py                  # Wang-Mendel rule generation, weighting, and pruning
│   ├── fuzzy_engine.py                    # Ruspini SFP, Mamdani inference, Centroid/Vote defuzzification
│   ├── evaluation.py                      # UCI Wine loader and 5x5 Repeated Stratified CV engine
│   ├── posthoc.py                         # K1 post-hoc rule importance ranking and top-k pruning
│   └── lens_analysis.py                   # Pickering et al. (2025) Fig. 7 Three-Lens evaluation table
├── experiments/
│   ├── baseline/
│   │   └── run_baseline.py                # Baseline FRBM + Decision Tree + Logistic Regression benchmark
│   ├── proposed/
│   │   ├── run_k1_topk.py                 # Contribution K1: Post-hoc rule importance & top-k curve
│   │   ├── run_k2_partitions.py           # Contribution K2: Uniform vs. Quantile SFP comparison
│   │   └── run_k3_defuzz.py               # Contribution K3: Centroid vs. Max-Vote defuzzification
│   └── run_all.py                         # End-to-end reproduction and aggregation script
├── notebooks/
│   └── 01_data_exploration.ipynb          # Exploratory Mutual Information & correlation feature screening
├── results/
│   ├── tables/                            # Generated CSV metrics and statistical summaries
│   └── figures/                           # Generated 300-DPI PNG figures
├── presentation/
│   └── presentation.pdf                   # Project presentation slides
├── report/
│   └── report.pdf                         # 8-12 page IEEE technical report
└── references/
    ├── README.md                          # Verified IEEE references
    └── literature_table.md                # Comparative literature review table
