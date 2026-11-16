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
Fuzzy Rule-Based Models (FRBMs) are widely valued for combining linguistic transparency with non-linear reasoning. However, Pickering et al. (2025) demonstrate that interpretability in FRBMs is often claimed informally rather than evaluated systematically across stakeholder perspectives and model stages. In their illustrative case study (Section 7, Fig. 7), the authors evaluate a 12-rule FisPro classifier on the UCI Wine dataset using their **Three-Lens Framework** (*Scale*, *Audience*, *Stage*), highlighting that the *Post-hoc* aspect is missing (red arrow in Fig. 7). The paper explicitly recommends post-hoc analysis tailored to FRBMs, such as finding the most important rules by instances covered or by average firing strength (Section 6.4).

The paper is a narrative review and reports **no quantitative experiment** (no accuracy, no membership-function parameters, no feature list for the Wine example). It only states that the example is a classification model with 12 rules and 4 features built in FisPro. Therefore the reproduction here is *structural*, and every concrete choice below (features, partitions, rule induction) is **our own decision**, not the paper's.

Two further weaknesses were identified **by us** while building the baseline (they are our observations, not claims of the paper):
1. **Partition sensitivity:** uniform grid-based Strong Fuzzy Partitions can misalign with skewed feature distributions.
2. **Nominal class defuzzification bias:** centroid defuzzification over `[0, 2]` treats the nominal labels `{0, 1, 2}` as ordinal quantities, so simultaneous firing of Class 0 and Class 2 rules averages toward Class 1.

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
├── requirements.txt                       # Pinned Python dependencies
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
    └── README.md                          # Verified IEEE references
```

---

## Dataset
- **UCI Wine** (Aeberhard & Forina, 1992): 178 samples, 13 chemical features, 3 cultivar classes (59 / 71 / 48 samples).
- Loaded with `sklearn.datasets.load_wine`; no data file is stored in the repository, so it is downloaded/loaded automatically when the code runs (see `data/README.md`).
- **Features (our choice, the paper does not list them):** `flavanoids`, `proline`, `color_intensity`, `alcohol`. They were chosen in `notebooks/01_data_exploration.ipynb` using mutual information, correlation and a quick decision-tree check on the **whole** dataset (an exploratory analysis; see Limitations).

## Installation
Python 3.10+ is recommended (developed and tested with the Python version of a Google Colab / Linux environment).

```bash
git clone https://github.com/Bayraminadresi/fuzzy-logic-project.git
cd fuzzy-logic-project
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

## Requirements
Exact versions are pinned in `requirements.txt`: `numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.2`, `scipy==1.13.1`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `scikit-fuzzy==0.5.0`. The fuzzy engine itself is implemented in NumPy only; `scikit-fuzzy` is used only in the exploration notebook. `matplotlib>=3.9` is needed (`boxplot(tick_labels=...)`). No API keys, passwords or tokens are used or stored.

## How to Run
Everything is seeded (`SEED = 42`). From the repository root:

```bash
python experiments/run_all.py                       # baseline -> K1 -> K2 -> K3 -> final comparison
```

or step by step (each script can be started from any directory and always writes to `results/`):

```bash
python experiments/baseline/run_baseline.py         # baseline FRBM, controls, 12 rules, MFs, lens table
python experiments/proposed/run_k1_topk.py          # K1: post-hoc rule importance + top-k curve
python experiments/proposed/run_k2_partitions.py    # K2: uniform vs quantile SFP
python experiments/proposed/run_k3_defuzz.py        # K3: centroid vs max-vote
```

In Colab: clone the repository, run `pip install -r requirements.txt`, then `python experiments/run_all.py`. A full run takes a few minutes on CPU.

## Reproduction Experiments
The paper's only concrete example (Section 7, Fig. 7) is a FisPro classifier on Wine with 12 rules and 4 features, assessed through three lenses. It gives no accuracy or parameters, so the comparison with the paper is structural:

| Aspect (Fig. 7) | Paper (FisPro) | This reproduction (`results/tables/lens_analysis.csv`) |
|---|---|---|
| Task / data | Classification, Wine | Classification, Wine |
| Rules / features | 12 rules, 4 features | 12 rules, 4 features (3 terms per feature) |
| Scale / Local | green | green (mean 2.48 rules fire per sample, measured) |
| Scale / Global | orange | orange (12 rules, 4 features) |
| Audience (designer / expert / end user) | green / orange / red | green / orange / red (qualitative, **not measured**) |
| Stage (pre / in / post-hoc) | green / green / red | green / green / red for the baseline; orange after K1 |

Baseline (`experiments/baseline/run_baseline.py`), 5x5 repeated stratified CV (25 folds, same folds for every model):

| Model | Accuracy | Macro-F1 | Rules |
|---|---|---|---|
| FRBM baseline (uniform SFP + centroid) | 0.604 ± 0.070 | 0.501 ± 0.084 | 12 |
| Decision tree (max_depth=3), same 4 features | 0.930 ± 0.049 | 0.931 ± 0.048 | 6.9 leaves |
| Logistic regression (L2), same 4 features | 0.963 ± 0.039 | 0.964 ± 0.038 | – |

The baseline's low accuracy is explained by the centroid defuzzification weakness (see K3), not by the rule base: with max-vote the same uniform partitions reach 0.953. The 12 extracted linguistic rules are printed by the baseline script and the membership functions are in `results/figures/baseline_mfs.png`.

## Proposed Extension
Three controlled extensions, all derived from the paper's recommendations or from weaknesses found in the baseline, evaluated on identical folds:

- **K1 – Post-hoc rule importance (paper Section 6.4).** `src/posthoc.py` ranks rules by mean firing strength (secondary: number of samples where the rule fires strongest) computed on the training fold only, then keeps the top-k rules (k = 1..12). This model-specific post-hoc method does not change the model.
- **K2 – Quantile-based Strong Fuzzy Partitions (paper Section 4.2).** Centres at the 10/50/90 % quantiles of the training data (shoulders keep the partition a Ruspini SFP) instead of uniform centres.
- **K3 – Max-vote defuzzification (optional).** Class-wise maximum firing strength, `argmax` over classes, instead of centroid over the numeric class axis.

The "proposed" model is K2+K3 (quantile SFP + vote). K1 is applied to the proposed model (rule importance is only meaningful for a model that classifies well) and, for comparison, to the baseline; both curves use the same folds.

## Experimental Results
Accuracy / macro-F1 over 25 folds (mean ± std; all tables in `results/tables/`):

| Model | Accuracy | Macro-F1 | Rules |
|---|---|---|---|
| Baseline (uniform + centroid) | 0.604 ± 0.070 | 0.501 ± 0.084 | 12 |
| K2: quantile + centroid | 0.944 ± 0.037 | 0.944 ± 0.038 | 12 |
| K3: uniform + vote | 0.953 ± 0.046 | 0.953 ± 0.045 | 12 |
| K2+K3: quantile + vote (proposed) | 0.950 ± 0.038 | 0.951 ± 0.038 | 12 |
| K1 on proposed, k = 9 | 0.938 ± 0.035 | 0.938 | 9 |
| K1 on proposed, k = 6 | 0.924 ± 0.045 | 0.920 | 6 |

Baseline vs proposed (`results/tables/final_comparison.csv`): accuracy +34.7 pp, macro-F1 +44.9 pp, rule count 12 → 12 (→ 9 with K1, within 0.02 accuracy of the full model). Paired Wilcoxon signed-rank tests (uniform vs quantile with centroid; centroid vs vote with uniform) give p = 5.96e-08 (the second-method score is higher in all 25 folds); because CV folds are not independent this is descriptive only.

K1 top-k curve (proposed model): k=3 → 0.847, k=6 → 0.924, k=9 → 0.938, k=12 → 0.950 accuracy. On the baseline model the curve is flat at about 0.60 for k ≥ 2 (`results/tables/k1_topk_curve_baseline.csv`), which shows that its weakness is the defuzzifier and not the number of rules.

Figures: `baseline_mfs.png`, `baseline_boxplot.png`, `lens_analysis.png`, `k1_topk_curve.png`, `k2_partitions.png`, `final_comparison.png` in `results/figures/`.

**Honest reading of the results.** The large gain is mostly due to replacing centroid with vote. Once vote is used, quantile partitions are not better than uniform ones (0.950 vs 0.953). K2 helps only under centroid. The proposed model is therefore not the single best configuration (uniform + vote is slightly higher); it was kept because it combines both extensions. The execution-time difference is within measurement noise (the same baseline configuration took roughly 1.9–2.5 ms per fold across repeated runs).

## Limitations
- **No numerical reproduction of the paper:** it reports no accuracy or parameters, and FisPro (Java) was not run. Our rule induction (Wang-Mendel + confidence pruning) only approximates FisPro's.
- **Feature-selection leakage:** the 4 features were selected on the whole dataset (exploratory notebook), not inside CV folds. Accuracy may be slightly optimistic.
- **Small dataset:** Wine has 178 samples; results may not generalise.
- **No user study:** the Audience ratings in the three-lens table are qualitative judgements; interpretability was not measured with people. The Scale/Local rating uses our own thresholds (≤3 green, 3–6 orange).
- **Statistics:** CV folds overlap, so Wilcoxon p-values are descriptive. No hyper-parameter search was performed.
- **K1 scope:** importance is a heuristic (mean firing strength); the "k within 0.02 of the full model" criterion is our own choice. Only Mamdani classification on tabular data is covered.
- **Timing:** wall-clock times are small and noisy.

## Attribution and Tools
All code in `src/` and `experiments/` was written for this project; the paper has no official code and FisPro code was not used. Libraries used: NumPy, pandas, scikit-learn, SciPy, Matplotlib (see `requirements.txt`). Use of generative AI tools is declared in the report (section "Generative AI Usage Disclosure").

## References
1. L. Pickering, K. Cohen, and B. De Baets, "A narrative review on the interpretability of fuzzy rule-based models from a modern interpretable machine learning perspective," *Int. J. Fuzzy Syst.*, 2025, doi: 10.1007/s40815-025-02022-z.
2. S. Aeberhard and M. Forina, "Wine," UCI Machine Learning Repository, 1992, doi: 10.24432/C5PC7J.
3. S. Guillaume and B. Charnomordic, "Learning interpretable fuzzy inference systems with FisPro," *Inf. Sci.*, vol. 181, no. 20, pp. 4409–4427, 2011, doi: 10.1016/j.ins.2011.03.025.
4. E. H. Ruspini, "A new approach to clustering," *Inf. Control*, vol. 15, no. 1, pp. 22–32, 1969.
5. C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nat. Mach. Intell.*, vol. 1, no. 5, pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.

The full, verified reference list is in `references/README.md` and the report.
