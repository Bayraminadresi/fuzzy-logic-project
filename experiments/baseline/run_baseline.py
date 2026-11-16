import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Repo kok dizinini Python arama yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Goreli yollar (results/...) her zaman repo kokune yazilsin diye calisma dizinini repo kokune al
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.fuzzy_engine import FuzzyClassifier
from src.rule_induction import format_rule
from src.evaluation import FEATURES, SEED, load_wine_xy, cv_evaluate, summarize
from src.lens_analysis import build_lens_table, render_lens_figure


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    X, y = load_wine_xy()

    # Baseline bulanik model ve iki klasik kontrol modeli tanimlanir
    models = [
        ("FuzzyClassifier (Baseline)", lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="uniform", defuzz="centroid")),
        ("DecisionTree (max_depth=3)", lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED)),
        ("LogisticRegression (L2)", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))),
    ]

    # 5x5 Repeated Stratified CV calistir
    dfs = [cv_evaluate(builder, X, y, name, n_splits=5, n_repeats=5, seed=SEED) for name, builder in models]
    cv_df = pd.concat(dfs, ignore_index=True)
    summary_df = summarize(cv_df)

    cv_df.to_csv("results/tables/baseline_cv.csv", index=False)
    summary_df.to_csv("results/tables/baseline_summary.csv", index=False)

    print("=== BASELINE 5x5 CV SUMMARY ===")
    print(summary_df[["model", "accuracy_fmt", "macro_f1_fmt", "n_rules", "seconds_fmt"]].to_string(index=False))

    # Tum veri uzerinde kurulan modelle 12 kurali ve uyelik fonksiyonlarini ciz
    full_model = FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="uniform", defuzz="centroid").fit(X, y)
    print("\n=== EXTRACTED 12 BASELINE LINGUISTIC RULES ===")
    for idx, r in enumerate(full_model.rules_, 1):
        print(f"R{idx:02d}: {format_rule(r, FEATURES)}")

    # Uyelik fonksiyonlari grafigi (baseline_mfs.png)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=300)
    for idx, fname in enumerate(FEATURES):
        ax = axes[idx // 2, idx % 2]
        part = full_model.partitions_[fname]
        x_grid = np.linspace(X[fname].min(), X[fname].max(), 300)
        mus = part.fuzzify(x_grid)
        for t_idx, lbl in enumerate(part.labels):
            ax.plot(x_grid, mus[:, t_idx], linewidth=2.0, label=lbl)
        ax.set_title(f"{fname} (Uniform SFP)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Membership Degree (μ)")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)
        if idx == 1:
            ax.legend(loc="right")
    plt.tight_layout()
    plt.savefig("results/figures/baseline_mfs.png", dpi=300)
    plt.close()

    # Kutu grafigi (baseline_boxplot.png)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    order = ["FuzzyClassifier (Baseline)", "DecisionTree (max_depth=3)", "LogisticRegression (L2)"]
    data_box = [cv_df.loc[cv_df["model"] == m, "accuracy"].values for m in order]
    bp = ax.boxplot(
        data_box,
        tick_labels=["FuzzyClassifier\n(Baseline, 12 rules)", "Decision Tree\n(max_depth=3)", "Logistic Regression\n(L2)"],
        patch_artist=True,
    )
    colors = ["#f5b7b1", "#aed6f1", "#a9dfbf"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("5x5 Repeated Stratified CV Accuracy")
    ax.set_title("Baseline Model Accuracy Distribution Across 25 Folds", fontweight="bold")
    ax.set_ylim(0.40, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/baseline_boxplot.png", dpi=300)
    plt.close()

    # Uc Mercek baseline tablosunu olustur (lens_analysis.csv ve lens_analysis.png)
    lens_df = build_lens_table(full_model, X, posthoc_updated=False)
    lens_df.to_csv("results/tables/lens_analysis.csv", index=False)
    render_lens_figure(lens_df, "results/figures/lens_analysis.png")


if __name__ == "__main__":
    main()
