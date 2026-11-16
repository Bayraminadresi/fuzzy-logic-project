import os
import subprocess
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_step(script_path: str):
    """Belirtilen deney betigini alt surec olarak calistirir."""
    print(f"\n>>> Running {script_path} ...")
    subprocess.run([sys.executable, script_path], check=True)


def build_final_comparison():
    """Tum deney CSV dosyalarini birlestirip nihai karsilastirma tablosunu ve grafigini uretir."""
    base_cv = pd.read_csv("results/tables/baseline_cv.csv")
    base_frbm = base_cv[base_cv["model"] == "FuzzyClassifier (Baseline)"]

    k3_cv = pd.read_csv("results/tables/k3_defuzz_cv.csv")
    prop_frbm = k3_cv[k3_cv["model"] == "Quantile + Vote (max)"]

    # K1: tam modelin dogrulugunun 0.02 icindeki en kucuk k (lens_analysis.py ile ayni kural)
    k1_df = pd.read_csv("results/tables/k1_topk_curve.csv").sort_values("k").reset_index(drop=True)
    k_max = int(k1_df["k"].max())
    full_acc = float(k1_df.loc[k1_df["k"] == k_max, "accuracy_mean"].iloc[0])
    ok = k1_df[(k1_df["k"] < k_max) & (k1_df["accuracy_mean"] >= full_acc - 0.02)]
    k_star = int(ok["k"].iloc[0]) if len(ok) > 0 else k_max
    k_row = k1_df[k1_df["k"] == k_star].iloc[0]

    b_acc, p_acc = base_frbm["accuracy"].mean(), prop_frbm["accuracy"].mean()
    b_f1, p_f1 = base_frbm["macro_f1"].mean(), prop_frbm["macro_f1"].mean()
    b_nr, p_nr = base_frbm["n_rules"].mean(), prop_frbm["n_rules"].mean()
    b_sec, p_sec = base_frbm["seconds"].mean() * 1000, prop_frbm["seconds"].mean() * 1000

    rows = [
        {
            "Metric": "Accuracy (5x5 CV)",
            "Baseline": f"{b_acc:.3f} ± {base_frbm['accuracy'].std():.3f}",
            "Proposed": f"{p_acc:.3f} ± {prop_frbm['accuracy'].std():.3f} (k={k_star}: {k_row['accuracy_mean']:.3f})",
            "Change": f"{(p_acc - b_acc) * 100:+.1f} pp",
        },
        {
            "Metric": "Macro-F1 (5x5 CV)",
            "Baseline": f"{b_f1:.3f} ± {base_frbm['macro_f1'].std():.3f}",
            "Proposed": f"{p_f1:.3f} ± {prop_frbm['macro_f1'].std():.3f} (k={k_star}: {k_row['macro_f1_mean']:.3f})",
            "Change": f"{(p_f1 - b_f1) * 100:+.1f} pp",
        },
        {
            "Metric": "Rule Count (n_rules)",
            "Baseline": f"{b_nr:.1f}",
            "Proposed": f"{p_nr:.1f} (compact K1: {k_star})",
            "Change": f"{p_nr - b_nr:+.1f} ({k_star - b_nr:+.1f} with K1)",
        },
        {
            "Metric": "Execution Time (fit+predict)",
            "Baseline": f"{b_sec:.2f} ms",
            "Proposed": f"{p_sec:.2f} ms",
            "Change": f"{p_sec - b_sec:+.2f} ms",
        },
    ]
    final_df = pd.DataFrame(rows)
    final_df.to_csv("results/tables/final_comparison.csv", index=False)
    print("\n=== FINAL BASELINE VS PROPOSED COMPARISON ===")
    print(final_df.to_string(index=False))

    # Nihai karsilastirma cubuk grafigi (final_comparison.png)
    k2_cv = pd.read_csv("results/tables/k2_partitions_cv.csv")
    q_cent = k2_cv[k2_cv["model"] == "Quantile SFP (Centroid)"]
    u_vote = k3_cv[k3_cv["model"] == "Uniform + Vote (max)"]
    dt_cv = base_cv[base_cv["model"] == "DecisionTree (max_depth=3)"]
    lr_cv = base_cv[base_cv["model"] == "LogisticRegression (L2)"]

    labels = [
        "Baseline FRBM\n(Uniform+Centroid)",
        "Proposed K2\n(Quantile+Centroid)",
        "Proposed K3\n(Uniform+Vote)",
        "Proposed K2+K3\n(Quantile+Vote)",
        "Decision Tree\n(max_depth=3)",
        "Logistic Reg.\n(L2)",
    ]
    groups = [base_frbm, q_cent, u_vote, prop_frbm, dt_cv, lr_cv]
    means = [g["accuracy"].mean() for g in groups]
    stds = [g["accuracy"].std() for g in groups]
    colors = ["#e74c3c", "#5dade2", "#48c9b0", "#f39c12", "#85c1e9", "#82e0aa"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("5x5 Repeated Stratified CV Accuracy")
    ax.set_title("Final Comparison: Baseline FRBM vs. Proposed Extensions & Controls", fontweight="bold")
    ax.set_ylim(0.45, 1.07)
    ax.grid(axis="y", alpha=0.3)
    for bar, m_val, s_val in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m_val + s_val + 0.012, f"{m_val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.xticks(fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/figures/final_comparison.png", dpi=300)
    plt.close()


def main():
    # Betikler goreli yollar (results/...) kullandigi icin her zaman repo kokunden calistir
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    scripts = [
        "experiments/baseline/run_baseline.py",
        "experiments/proposed/run_k1_topk.py",
        "experiments/proposed/run_k2_partitions.py",
        "experiments/proposed/run_k3_defuzz.py",
    ]
    for sc in scripts:
        run_step(sc)
    build_final_comparison()
    print("\nTum tablo ve grafikler results/ altinda basariyla olusturuldu.")


if __name__ == "__main__":
    main()
