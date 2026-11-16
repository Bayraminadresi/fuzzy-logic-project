import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Repo kok dizinini Python arama yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.fuzzy_engine import FuzzyClassifier
from src.evaluation import FEATURES, SEED, load_wine_xy, cv_evaluate, summarize


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    X, y = load_wine_xy()

    # Uniform SFP ile Quantile SFP ayni 25 fold uzerinde karsilastirilir
    df_uni = cv_evaluate(
        lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="uniform", defuzz="centroid"),
        X, y, "Uniform SFP (Centroid)", n_splits=5, n_repeats=5, seed=SEED
    )
    df_qtl = cv_evaluate(
        lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="quantile", defuzz="centroid"),
        X, y, "Quantile SFP (Centroid)", n_splits=5, n_repeats=5, seed=SEED
    )

    k2_df = pd.concat([df_uni, df_qtl], ignore_index=True)
    k2_df.to_csv("results/tables/k2_partitions_cv.csv", index=False)

    # Eslestirilmis Wilcoxon isaretli sira testi (betimleyici istatistik)
    stat, p_val = wilcoxon(df_uni["accuracy"].values, df_qtl["accuracy"].values)
    print("=== K2: UNIFORM SFP vs QUANTILE SFP (5x5 CV) ===")
    print(summarize(k2_df)[["model", "accuracy_fmt", "macro_f1_fmt", "n_rules", "seconds_fmt"]].to_string(index=False))
    print(f"Paired Wilcoxon signed-rank test on fold accuracy (descriptive, non-independent folds): stat={stat:.2f}, p={p_val:.6f}")

    # Uyelik fonksiyonlari karsilastirmasi ve kutu grafigi (k2_partitions.png)
    m_uni = FuzzyClassifier(features=FEATURES, partition_method="uniform").fit(X, y)
    m_qtl = FuzzyClassifier(features=FEATURES, partition_method="quantile").fit(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=300)
    for idx, fname in enumerate(["flavanoids", "proline"]):
        ax = axes[idx]
        x_grid = np.linspace(X[fname].min(), X[fname].max(), 300)
        mu_u = m_uni.partitions_[fname].fuzzify(x_grid)
        mu_q = m_qtl.partitions_[fname].fuzzify(x_grid)
        for t_idx, lbl in enumerate(["Low", "Medium", "High"]):
            ax.plot(x_grid, mu_u[:, t_idx], linestyle="--", alpha=0.65, linewidth=1.6, label=f"Uniform {lbl}" if idx == 0 else None)
            ax.plot(x_grid, mu_q[:, t_idx], linestyle="-", linewidth=2.0, label=f"Quantile {lbl}" if idx == 0 else None)
        ax.set_title(f"{fname}: Uniform vs Quantile SFP", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Membership Degree (μ)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="right")

    ax_box = axes[2]
    bp = ax_box.boxplot(
        [df_uni["accuracy"].values, df_qtl["accuracy"].values],
        labels=["Uniform SFP\n(Centroid)", "Quantile SFP\n(Centroid)"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#f5b7b1")
    bp["boxes"][1].set_facecolor("#a9dfbf")
    ax_box.set_title(f"5x5 CV Accuracy (Wilcoxon p={p_val:.1e})", fontsize=9.5, fontweight="bold")
    ax_box.set_ylabel("Accuracy")
    ax_box.set_ylim(0.40, 1.05)
    ax_box.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/k2_partitions.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
