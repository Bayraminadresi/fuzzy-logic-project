import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_lens_table(
    model,
    X,
    posthoc_updated: bool = False,
    k1_curve_path: str = "results/tables/k1_topk_curve.csv",
    baseline_acc: float | None = None,
) -> pd.DataFrame:
    """Pickering et al. (2025) Fig. 7 Uc Mercek (Scale, Audience, Stage) degerlendirme tablosunu olusturur."""
    firings = model.firing_strengths(X)
    # Yerel olcek: ornek basina 0.05 uzerinde ateslenen ortalama kural sayisi
    active_per_sample = np.sum(firings > 0.05, axis=1)
    mean_active = float(np.mean(active_per_sample))

    if mean_active <= 3.0:
        local_rating = "green"
    elif mean_active <= 6.0:
        local_rating = "orange"
    else:
        local_rating = "red"

    n_rules = len(model.rules_)
    n_features = len(model.features)

    # Post-hoc asamasi: Baseline'da kirmizi, K1 katkisindan sonra turuncu olarak guncellenir
    if posthoc_updated and os.path.exists(k1_curve_path):
        k1_df = pd.read_csv(k1_curve_path)
        if baseline_acc is None:
            baseline_acc = float(k1_df.loc[k1_df["k"] == k1_df["k"].max(), "accuracy_mean"].iloc[0])
        qualifying = k1_df[(k1_df["k"] < k1_df["k"].max()) & (k1_df["accuracy_mean"] >= baseline_acc - 0.02)]
        if len(qualifying) > 0:
            best_row = qualifying.iloc[0]
            k_val = int(best_row["k"])
            k_acc = float(best_row["accuracy_mean"])
            posthoc_rating = "orange"
            posthoc_evidence = (
                f"K1 post-hoc rule importance & top-k pruning: k={k_val} rules achieves "
                f"acc={k_acc:.3f} (within 0.02 of full {n_rules}-rule model; no user study)"
            )
        else:
            posthoc_rating = "orange"
            posthoc_evidence = (
                "K1 post-hoc rule importance & top-k curve computed; full rule set needed "
                "for peak accuracy (no user study)"
            )
    else:
        posthoc_rating = "red"
        posthoc_evidence = "No post-hoc explanation or rule importance ranking in baseline (matches Fig. 7)"

    rows = [
        {
            "lens": "Scale",
            "aspect": "Local (instance-level)",
            "evidence": f"Mean rules with firing > 0.05 per sample = {mean_active:.2f} (threshold <=3 green, 3-6 orange)",
            "rating": local_rating,
        },
        {
            "lens": "Scale",
            "aspect": "Global (model-level)",
            "evidence": f"Compact rule base: n_rules = {n_rules}, n_features = {n_features}, 3 linguistic terms/feature",
            "rating": "orange",
        },
        {
            "lens": "Audience",
            "aspect": "Model Designer",
            "evidence": "Full access to SFP parameters, rule weights, and firing strengths (not measured; qualitative)",
            "rating": "green",
        },
        {
            "lens": "Audience",
            "aspect": "Domain Expert (Oenologist)",
            "evidence": "4 chemical features with Low/Medium/High linguistic rules (not measured; qualitative)",
            "rating": "orange",
        },
        {
            "lens": "Audience",
            "aspect": "End User (Layperson)",
            "evidence": "12 multi-antecedent chemical rules require domain literacy (not measured; qualitative)",
            "rating": "red",
        },
        {
            "lens": "Stage",
            "aspect": "Pre-model",
            "evidence": "MI + correlation feature screening in 01_data_exploration (4 non-redundant features)",
            "rating": "green",
        },
        {
            "lens": "Stage",
            "aspect": "In-model",
            "evidence": "Ruspini Strong Fuzzy Partition (sum=1) + pruning to 4 rules/class (12 total)",
            "rating": "green",
        },
        {
            "lens": "Stage",
            "aspect": "Post-hoc",
            "evidence": posthoc_evidence,
            "rating": posthoc_rating,
        },
    ]
    return pd.DataFrame(rows)


def render_lens_figure(df: pd.DataFrame, out_path: str = "results/figures/lens_analysis.png") -> None:
    """Uc Mercek tablosunu renkli PNG gorseli olarak kaydeder."""
    color_map = {"green": "#a9dfbf", "orange": "#f9e79f", "red": "#f5b7b1"}
    label_map = {"green": "Strong (Green)", "orange": "Partial / Weak (Orange)", "red": "Missing / Poor (Red)"}

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
    ax.axis("off")

    cell_text = []
    cell_colors = []
    for _, row in df.iterrows():
        r_key = str(row["rating"]).lower()
        wrapped_ev = textwrap.fill(str(row["evidence"]), width=62)
        cell_text.append([row["lens"], row["aspect"], wrapped_ev, label_map.get(r_key, r_key)])
        cell_colors.append(["#f8f9f9", "#f8f9f9", "#ffffff", color_map.get(r_key, "#ffffff")])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Lens", "Aspect", "Measured / Qualitative Evidence", "Assessment"],
        cellColours=cell_colors,
        colColours=["#d5dbdb", "#d5dbdb", "#d5dbdb", "#d5dbdb"],
        loc="center",
        cellLoc="left",
        colWidths=[0.10, 0.21, 0.51, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.75)

    for (r_idx, c_idx), cell in table.get_celld().items():
        if r_idx == 0:
            cell.get_text().set_weight("bold")
        if c_idx == 3 and r_idx > 0:
            cell.get_text().set_weight("bold")

    ax.set_title(
        "Three-Lens Interpretability Evaluation (Adapted from Pickering et al., 2025, Fig. 7)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
