import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Repo kok dizinini Python arama yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Goreli yollar (results/...) her zaman repo kokune yazilsin diye calisma dizinini repo kokune al
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.fuzzy_engine import FuzzyClassifier
from src.evaluation import FEATURES, SEED, load_wine_xy
from src.posthoc import rule_importance, top_k_model
from src.lens_analysis import build_lens_table, render_lens_figure


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    X, y = load_wine_xy()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    records = {k: {"acc": [], "f1": [], "fb": []} for k in range(1, 13)}

    # Her fold icinde egitim setinden kural onemi hesaplanir ve k=1..12 test edilir
    for train_idx, test_idx in rskf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        model = FuzzyClassifier(
            features=FEATURES,
            rules_per_class=4,
            partition_method="quantile",
            defuzz="vote",
        ).fit(X_train, y_train)

        imp_df = rule_importance(model, X_train)
        for k in range(1, 13):
            mk = top_k_model(model, imp_df, k)
            y_pred = mk.predict(X_test)
            records[k]["acc"].append(accuracy_score(y_test, y_pred))
            records[k]["f1"].append(f1_score(y_test, y_pred, average="macro"))
            records[k]["fb"].append(mk.n_fallback_)

    rows = []
    for k in range(1, 13):
        rows.append(
            {
                "k": k,
                "accuracy_mean": round(float(np.mean(records[k]["acc"])), 4),
                "accuracy_std": round(float(np.std(records[k]["acc"])), 4),
                "macro_f1_mean": round(float(np.mean(records[k]["f1"])), 4),
                "n_fallback_mean": round(float(np.mean(records[k]["fb"])), 2),
            }
        )
    k1_df = pd.DataFrame(rows)
    k1_df.to_csv("results/tables/k1_topk_curve.csv", index=False)

    print("=== K1 POST-HOC RULE IMPORTANCE TOP-K CURVE (5x5 CV) ===")
    print(k1_df.to_string(index=False))

    # Tum veri uzerinde post-hoc kural onemi siralamasini kaydet
    full_proposed = FuzzyClassifier(
        features=FEATURES, rules_per_class=4, partition_method="quantile", defuzz="vote"
    ).fit(X, y)
    imp_full = rule_importance(full_proposed, X)
    imp_full.drop(columns=["rule_idx"]).to_csv("results/tables/k1_rule_importance.csv", index=False)
    print("\n=== POST-HOC RULE IMPORTANCE RANKING (FULL DATASET) ===")
    print(imp_full[["rule_id", "mean_firing", "winner_count", "text"]].to_string(index=False))

    # Top-k egrisi grafigi (k1_topk_curve.png)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=300)
    ax.plot(k1_df["k"], k1_df["accuracy_mean"], marker="o", linewidth=2.2, color="#1f77b4", label="Accuracy (5x5 CV)")
    ax.fill_between(
        k1_df["k"],
        k1_df["accuracy_mean"] - k1_df["accuracy_std"],
        k1_df["accuracy_mean"] + k1_df["accuracy_std"],
        alpha=0.18,
        color="#1f77b4",
    )
    ax.plot(k1_df["k"], k1_df["macro_f1_mean"], marker="s", linestyle="--", linewidth=1.8, color="#ff7f0e", label="Macro-F1 (5x5 CV)")
    full_acc = float(k1_df.loc[k1_df["k"] == 12, "accuracy_mean"].iloc[0])
    ax.axhline(full_acc, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Full 12-Rule Model ({full_acc:.3f})")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Number of Active Rules (Top-k by Post-hoc Importance)")
    ax.set_ylabel("Performance Score")
    ax.set_title("Post-hoc Rule Importance: Rule Count vs. Accuracy Curve (K1)", fontweight="bold")
    ax.set_ylim(0.30, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/figures/k1_topk_curve.png", dpi=300)
    plt.close()

    # Uc Mercek tablosunu K1 kaniti ile guncelle (T21)
    lens_df = build_lens_table(full_proposed, X, posthoc_updated=True, k1_curve_path="results/tables/k1_topk_curve.csv", baseline_acc=full_acc)
    lens_df.to_csv("results/tables/lens_analysis.csv", index=False)
    render_lens_figure(lens_df, "results/figures/lens_analysis.png")


if __name__ == "__main__":
    main()
