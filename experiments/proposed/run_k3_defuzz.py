import os
import sys
import pandas as pd
from scipy.stats import wilcoxon

# Repo kok dizinini Python arama yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.fuzzy_engine import FuzzyClassifier
from src.evaluation import FEATURES, SEED, load_wine_xy, cv_evaluate, summarize


def main():
    os.makedirs("results/tables", exist_ok=True)
    X, y = load_wine_xy()

    # Centroid ile Max-Vote durulastirma yontemlerini karsilastir (K3)
    df_cent = cv_evaluate(
        lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="uniform", defuzz="centroid"),
        X, y, "Uniform + Centroid", n_splits=5, n_repeats=5, seed=SEED
    )
    df_vote = cv_evaluate(
        lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="uniform", defuzz="vote"),
        X, y, "Uniform + Vote (max)", n_splits=5, n_repeats=5, seed=SEED
    )
    df_q_vote = cv_evaluate(
        lambda: FuzzyClassifier(features=FEATURES, rules_per_class=4, partition_method="quantile", defuzz="vote"),
        X, y, "Quantile + Vote (max)", n_splits=5, n_repeats=5, seed=SEED
    )

    k3_df = pd.concat([df_cent, df_vote, df_q_vote], ignore_index=True)
    k3_df.to_csv("results/tables/k3_defuzz_cv.csv", index=False)

    stat, p_val = wilcoxon(df_cent["accuracy"].values, df_vote["accuracy"].values)
    print("=== K3: CENTROID vs MAX-VOTE DEFUZZIFICATION (5x5 CV) ===")
    print(summarize(k3_df)[["model", "accuracy_fmt", "macro_f1_fmt", "n_rules", "seconds_fmt"]].to_string(index=False))
    print(f"Paired Wilcoxon (Uniform Centroid vs Vote): stat={stat:.2f}, p={p_val:.6f}")


if __name__ == "__main__":
    main()
