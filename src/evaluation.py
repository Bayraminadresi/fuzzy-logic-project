import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    from rule_induction import FEATURES
except ImportError:
    from src.rule_induction import FEATURES

# Ortak proje sabitleri (FEATURES tek kaynaktan, rule_induction.py'den gelir)
SEED = 42


def load_wine_xy() -> tuple[pd.DataFrame, np.ndarray]:
    """UCI Wine veri setini secili 4 ozellik (X) ve sinif etiketleri (y) olarak yukler."""
    data = load_wine(as_frame=True)
    X = data.data[FEATURES].copy()
    y = data.target.to_numpy(dtype=int)
    return X, y


def cv_evaluate(
    make_model,
    X,
    y,
    name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """5x5 RepeatedStratifiedKFold capraz dogrulama ile modeli olcer ve fold sonuclarini dondurur."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X_df = X[FEATURES] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
    y_arr = np.asarray(y, dtype=int)

    rows = []
    for split_idx, (train_idx, test_idx) in enumerate(rskf.split(X_df, y_arr)):
        repeat = split_idx // n_splits
        fold = split_idx % n_splits
        X_train, y_train = X_df.iloc[train_idx], y_arr[train_idx]
        X_test, y_test = X_df.iloc[test_idx], y_arr[test_idx]

        # Her fold icin sifirdan model olusturulur (egitim/test sizintisi onlenir)
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - t0

        if hasattr(model, "rules_"):
            n_rules = float(len(model.rules_))
        elif hasattr(model, "get_n_leaves"):
            n_rules = float(model.get_n_leaves())
        else:
            n_rules = np.nan

        rows.append(
            {
                "model": name,
                "repeat": repeat,
                "fold": fold,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "n_rules": n_rules,
                "seconds": float(elapsed),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Fold sonuclarini model bazinda ortalama +- standart sapma olarak ozetler."""
    summary_rows = []
    for model_name, grp in df.groupby("model", sort=False):
        acc_m, acc_s = grp["accuracy"].mean(), grp["accuracy"].std()
        f1_m, f1_s = grp["macro_f1"].mean(), grp["macro_f1"].std()
        nr_m = grp["n_rules"].mean()
        sec_m, sec_s = grp["seconds"].mean(), grp["seconds"].std()
        summary_rows.append(
            {
                "model": model_name,
                "accuracy_mean": round(float(acc_m), 4),
                "accuracy_std": round(float(acc_s), 4),
                "accuracy_fmt": f"{acc_m:.3f} ± {acc_s:.3f}",
                "macro_f1_mean": round(float(f1_m), 4),
                "macro_f1_std": round(float(f1_s), 4),
                "macro_f1_fmt": f"{f1_m:.3f} ± {f1_s:.3f}",
                "n_rules": round(float(nr_m), 1) if not np.isnan(nr_m) else np.nan,
                "seconds_mean": round(float(sec_m), 5),
                "seconds_fmt": f"{sec_m * 1000:.2f} ± {sec_s * 1000:.2f} ms",
            }
        )
    return pd.DataFrame(summary_rows)
