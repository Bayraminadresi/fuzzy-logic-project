import numpy as np
import pandas as pd

try:
    from rule_induction import FEATURES, format_rule
except ImportError:
    from src.rule_induction import FEATURES, format_rule


def rule_importance(model, X) -> pd.DataFrame:
    """Ortalama atesleme gucu ve kazanan ornek sayisina gore post-hoc kural onemini hesaplar (K1)."""
    firings = model.firing_strengths(X)  # (n, n_rules)
    n_rules = firings.shape[1]
    mean_firing = np.mean(firings, axis=0) if n_rules > 0 else np.array([])

    # Her ornekte en yuksek ateslenen (kazanan) kurali say
    winner_counts = np.zeros(n_rules, dtype=int)
    for i in range(firings.shape[0]):
        if np.any(firings[i] > 0.0):
            winner_counts[int(np.argmax(firings[i]))] += 1

    rows = []
    for idx, r in enumerate(model.rules_):
        rows.append(
            {
                "rule_id": f"R{idx + 1:02d}",
                "rule_idx": idx,
                "text": format_rule(r, getattr(model, "features", FEATURES)),
                "mean_firing": float(mean_firing[idx]),
                "winner_count": int(winner_counts[idx]),
                "importance": float(mean_firing[idx]),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(by=["importance", "winner_count"], ascending=[False, False]).reset_index(drop=True)
    return df


def top_k_model(model, importance_df: pd.DataFrame, k: int):
    """En onemli ilk k kurali tutarak budanmis bir FuzzyClassifier kopyasi dondurur."""
    top_indices = importance_df["rule_idx"].iloc[:k].tolist()
    selected_rules = [model.rules_[idx] for idx in top_indices]
    return model.with_rules(selected_rules)
