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

# K1 iki modelde calistirilir (ayni 25 fold):
#  - "proposed": K2+K3 modeli (quantile SFP + vote). Birincil analiz: kural onemi ancak
#    iyi siniflandiran bir modelde anlamlidir. Cikti: k1_topk_curve.csv (lens analizi ve final tablo bunu okur).
#  - "baseline": makalenin yapisal yeniden uretimi (uniform SFP + centroid). Karsilastirma amacli.
#    Baseline'in dogrulugu defuzzification yuzunden dusuk oldugu icin top-k egrisi duzdur (bkz. rapor).
CONFIGS = {
    "proposed": dict(partition_method="quantile", defuzz="vote"),
    "baseline": dict(partition_method="uniform", defuzz="centroid"),
}
K_RANGE = range(1, 13)


def make_model(cfg):
    return FuzzyClassifier(features=FEATURES, rules_per_class=4, **cfg)


def topk_curve(X, y):
    """Her konfigurasyon icin k=1..12 top-k egrisini 5x5 CV ile hesaplar (importance her fold'un egitiminde)."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    records = {name: {k: {"acc": [], "f1": [], "fb": []} for k in K_RANGE} for name in CONFIGS}

    for train_idx, test_idx in rskf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        for name, cfg in CONFIGS.items():
            model = make_model(cfg).fit(X_train, y_train)
            imp_df = rule_importance(model, X_train)
            for k in K_RANGE:
                mk = top_k_model(model, imp_df, k)
                y_pred = mk.predict(X_test)
                records[name][k]["acc"].append(accuracy_score(y_test, y_pred))
                records[name][k]["f1"].append(f1_score(y_test, y_pred, average="macro"))
                records[name][k]["fb"].append(mk.n_fallback_)

    curves = {}
    for name in CONFIGS:
        rows = []
        for k in K_RANGE:
            r = records[name][k]
            rows.append(
                {
                    "k": k,
                    "accuracy_mean": round(float(np.mean(r["acc"])), 4),
                    "accuracy_std": round(float(np.std(r["acc"], ddof=1)), 4),  # ddof=1: diger tablolarla (pandas .std()) tutarli
                    "macro_f1_mean": round(float(np.mean(r["f1"])), 4),
                    "n_fallback_mean": round(float(np.mean(r["fb"])), 2),
                }
            )
        curves[name] = pd.DataFrame(rows)
    return curves


def plot_curve(ax, df, title):
    ax.plot(df["k"], df["accuracy_mean"], marker="o", linewidth=2.2, color="#1f77b4", label="Accuracy (5x5 CV)")
    ax.fill_between(
        df["k"],
        df["accuracy_mean"] - df["accuracy_std"],
        df["accuracy_mean"] + df["accuracy_std"],
        alpha=0.18,
        color="#1f77b4",
    )
    ax.plot(df["k"], df["macro_f1_mean"], marker="s", linestyle="--", linewidth=1.8, color="#ff7f0e", label="Macro-F1 (5x5 CV)")
    full_acc = float(df.loc[df["k"] == 12, "accuracy_mean"].iloc[0])
    ax.axhline(full_acc, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Full 12-Rule Model ({full_acc:.3f})")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Number of Active Rules (Top-k by Post-hoc Importance)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0.10, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    X, y = load_wine_xy()
    curves = topk_curve(X, y)
    k1_df = curves["proposed"]
    k1_base_df = curves["baseline"]

    k1_df.to_csv("results/tables/k1_topk_curve.csv", index=False)
    k1_base_df.to_csv("results/tables/k1_topk_curve_baseline.csv", index=False)

    print("=== K1 TOP-K CURVE: PROPOSED MODEL (quantile SFP + vote), 5x5 CV ===")
    print(k1_df.to_string(index=False))
    print("\n=== K1 TOP-K CURVE: BASELINE MODEL (uniform SFP + centroid), 5x5 CV ===")
    print(k1_base_df.to_string(index=False))

    # Tum veri uzerinde post-hoc kural onemi siralamasini kaydet (her iki model)
    full_models = {name: make_model(cfg).fit(X, y) for name, cfg in CONFIGS.items()}
    imp_full = rule_importance(full_models["proposed"], X)
    imp_full.drop(columns=["rule_idx"]).to_csv("results/tables/k1_rule_importance.csv", index=False)
    imp_base = rule_importance(full_models["baseline"], X)
    imp_base.drop(columns=["rule_idx"]).to_csv("results/tables/k1_rule_importance_baseline.csv", index=False)
    print("\n=== POST-HOC RULE IMPORTANCE RANKING (PROPOSED MODEL, FULL DATASET) ===")
    print(imp_full[["rule_id", "mean_firing", "winner_count", "text"]].to_string(index=False))

    # Top-k egrisi grafigi (k1_topk_curve.png): sol = proposed, sag = baseline
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=300, sharey=True)
    plot_curve(axes[0], k1_df, "Proposed model (Quantile SFP + Vote)")
    plot_curve(axes[1], k1_base_df, "Baseline model (Uniform SFP + Centroid)")
    axes[0].set_ylabel("Performance Score")
    fig.suptitle("K1: Post-hoc Rule Importance, Rule Count vs. Performance", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/k1_topk_curve.png", dpi=300)
    plt.close()

    # Uc Mercek tablosunu K1 kaniti ile guncelle (birincil model = proposed)
    full_acc = float(k1_df.loc[k1_df["k"] == 12, "accuracy_mean"].iloc[0])
    lens_df = build_lens_table(full_models["proposed"], X, posthoc_updated=True, k1_curve_path="results/tables/k1_topk_curve.csv", baseline_acc=full_acc)
    lens_df.to_csv("results/tables/lens_analysis.csv", index=False)
    render_lens_figure(lens_df, "results/figures/lens_analysis.png")


if __name__ == "__main__":
    main()
