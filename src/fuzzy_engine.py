import copy
import numpy as np
import pandas as pd

try:
    from rule_induction import (
        FEATURES,
        DEFAULT_LABELS,
        Rule,
        generate_candidate_rules,
        weight_and_prune,
        format_rule,
    )
except ImportError:
    from src.rule_induction import (
        FEATURES,
        DEFAULT_LABELS,
        Rule,
        generate_candidate_rules,
        weight_and_prune,
        format_rule,
    )


class StrongFuzzyPartition:
    """Ucgen uyelik fonksiyonlu, uclari omuzlu Ruspini Guclu Bulanik Bolumleme (her x icin toplam = 1)."""

    def __init__(self, centers, labels=DEFAULT_LABELS):
        self.centers = np.asarray(centers, dtype=float)
        if len(self.centers) != len(labels):
            raise ValueError("Merkez sayisi etiket sayisina esit olmalidir.")
        if not np.all(np.diff(self.centers) > 0):
            raise ValueError("Merkezler kesin artan sirada olmalidir.")
        self.labels = tuple(labels)

    @classmethod
    def uniform(cls, lo: float, hi: float, labels=DEFAULT_LABELS):
        """[lo, hi] araliginda esit aralikli (uniform) merkezlerle bolumleme olusturur."""
        return cls(np.linspace(lo, hi, len(labels)), labels=labels)

    @classmethod
    def from_quantiles(cls, x, qs=(0.1, 0.5, 0.9), labels=DEFAULT_LABELS):
        """Egitim verisinin kantil (quantile) noktalarindan merkezler olusturur (K2 katkisi)."""
        centers = np.quantile(np.asarray(x, dtype=float), qs)
        # Tekrarlanan degerlerde merkezlerin kesin artan kalmasini garanti et
        for i in range(1, len(centers)):
            if centers[i] <= centers[i - 1]:
                centers[i] = centers[i - 1] + 1e-6
        return cls(centers, labels=labels)

    def fuzzify(self, x) -> np.ndarray:
        """Girdi vektorunu (n,) boyutundan (n, 3) uyelik matrisine donusturur (uclarda omuz korunur)."""
        x_arr = np.asarray(x, dtype=float)
        eye = np.eye(len(self.centers))
        return np.stack(
            [np.interp(x_arr, self.centers, eye[i]) for i in range(len(self.centers))],
            axis=-1,
        )


def firing_strengths(mu: np.ndarray, rules: list[Rule]) -> np.ndarray:
    """Agirlikli Mamdani kural atesleme guclerini hesaplar: f_r = w_r * min_f mu[:, f, ant_f]."""
    n_samples, n_features, _ = mu.shape
    if len(rules) == 0:
        return np.zeros((n_samples, 0), dtype=float)
    firings = np.zeros((n_samples, len(rules)), dtype=float)
    for idx, r in enumerate(rules):
        a_r = np.min(
            np.stack([mu[:, f, r.antecedent[f]] for f in range(n_features)], axis=1),
            axis=1,
        )
        firings[:, idx] = r.weight * a_r
    return firings


def aggregate(firings: np.ndarray, rules: list[Rule], grid: np.ndarray) -> np.ndarray:
    """[0, 2] cikis evreninde Mamdani min-gerektirme ve max-birlestirme uygular."""
    n_samples = firings.shape[0]
    mu_out = np.zeros((n_samples, len(grid)), dtype=float)
    if len(rules) == 0:
        return mu_out
    out_mfs = StrongFuzzyPartition.uniform(0.0, 2.0).fuzzify(grid)
    for idx, r in enumerate(rules):
        clipped = np.minimum(firings[:, idx : idx + 1], out_mfs[:, r.consequent][None, :])
        mu_out = np.maximum(mu_out, clipped)
    return mu_out


def defuzzify_centroid(mu_out: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Agirlik merkezi (centroid) durulastirmasi; hic kural ateslenmezse (toplam=0) NaN dondurur."""
    sums = np.sum(mu_out, axis=1)
    centroids = np.full(mu_out.shape[0], np.nan, dtype=float)
    valid = sums > 0.0
    if np.any(valid):
        centroids[valid] = np.sum(mu_out[valid] * grid[None, :], axis=1) / sums[valid]
    return centroids


def decide(centroids: np.ndarray, majority_class: int) -> tuple[np.ndarray, int]:
    """Surekli centroid degerlerini {0, 1, 2} siniflarina yuvarlar; atesleme yoksa cogunluk sinifini atar."""
    preds = np.full(len(centroids), int(majority_class), dtype=int)
    valid = ~np.isnan(centroids)
    n_fallback = int(np.sum(~valid))
    preds[valid] = np.clip(np.round(centroids[valid]), 0, 2).astype(int)
    return preds, n_fallback


class FuzzyClassifier:
    """Yorumlanabilir Mamdani Bulanik Kural Tabanli Siniflandirici."""

    def __init__(
        self,
        features=FEATURES,
        rules_per_class: int = 4,
        partition_method: str = "uniform",
        defuzz: str = "centroid",
        n_points: int = 201,
    ):
        if partition_method not in ("uniform", "quantile"):
            raise NotImplementedError(f"Desteklenmeyen partition_method: {partition_method}")
        if defuzz not in ("centroid", "vote"):
            raise NotImplementedError(f"Desteklenmeyen defuzz: {defuzz}")

        self.features = list(features)
        self.rules_per_class = rules_per_class
        self.partition_method = partition_method
        self.defuzz = defuzz
        self.n_points = n_points

        self.partitions_: dict[str, StrongFuzzyPartition] = {}
        self.rules_: list[Rule] = []
        self.majority_: int = 0
        self.n_fallback_: int = 0
        self.grid_ = np.linspace(0.0, 2.0, n_points)

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.features].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _fuzzify_X(self, X) -> np.ndarray:
        X_arr = self._to_array(X)
        return np.stack(
            [self.partitions_[f].fuzzify(X_arr[:, idx]) for idx, f in enumerate(self.features)],
            axis=1,
        )

    def fit(self, X, y):
        """Egitim verisinden SFP bolumlemelerini kurar, Wang-Mendel kurallarini cikarir ve budar."""
        X_arr = self._to_array(X)
        y_arr = np.asarray(y, dtype=int)
        vals, counts = np.unique(y_arr, return_counts=True)
        self.majority_ = int(vals[np.argmax(counts)])

        # Her ozellik icin yalnizca egitim verisini kullanarak SFP olustur (veri sizintisi onlenir)
        self.partitions_ = {}
        for idx, fname in enumerate(self.features):
            col = X_arr[:, idx]
            if self.partition_method == "quantile":
                self.partitions_[fname] = StrongFuzzyPartition.from_quantiles(col)
            else:
                self.partitions_[fname] = StrongFuzzyPartition.uniform(float(col.min()), float(col.max()))

        mu_train = self._fuzzify_X(X_arr)
        candidates = generate_candidate_rules(mu_train, y_arr)
        self.rules_ = weight_and_prune(
            candidates, mu_train, y_arr, rules_per_class=self.rules_per_class
        )
        self.n_fallback_ = 0
        return self

    def firing_strengths(self, X) -> np.ndarray:
        """Verilen ornekler icin tum kurallarin atesleme guclerini dondurur."""
        mu = self._fuzzify_X(X)
        return firing_strengths(mu, self.rules_)

    def predict(self, X) -> np.ndarray:
        """Centroid veya sinif bazli oylama (vote) yontemiyle sinif tahmini yapar."""
        firings = self.firing_strengths(X)
        n_samples = firings.shape[0]

        if len(self.rules_) == 0:
            self.n_fallback_ = n_samples
            return np.full(n_samples, self.majority_, dtype=int)

        # K3 katkisi: Her sinif icin en yuksek kural atesleme gucunu karsilastiran oylama yontemi
        if self.defuzz == "vote":
            preds = np.full(n_samples, self.majority_, dtype=int)
            n_fb = 0
            for i in range(n_samples):
                if np.all(firings[i] == 0.0):
                    n_fb += 1
                    continue
                scores = []
                for c in (0, 1, 2):
                    c_firings = [
                        firings[i, r_idx]
                        for r_idx, r in enumerate(self.rules_)
                        if r.consequent == c
                    ]
                    scores.append(max(c_firings) if c_firings else 0.0)
                preds[i] = int(np.argmax(scores))
            self.n_fallback_ = n_fb
            return preds

        # Varsayilan Mamdani + Centroid durulastirma
        mu_out = aggregate(firings, self.rules_, self.grid_)
        centroids = defuzzify_centroid(mu_out, self.grid_)
        preds, n_fb = decide(centroids, self.majority_)
        self.n_fallback_ = n_fb
        return preds

    def with_rules(self, rules: list[Rule]):
        """Ayni uyelik bolumlemelerini koruyarak yalnizca secilen kurallarla model kopyasi olusturur (K1 icin)."""
        clone = copy.copy(self)
        clone.rules_ = list(rules)
        clone.n_fallback_ = 0
        return clone

    def explain(self, x) -> str:
        """Tek bir ornek icin uyelik derecelerini, ateslenen kurallari ve tahmini metin olarak aciklar."""
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        mu = self._fuzzify_X(x_arr)[0]
        firings = self.firing_strengths(x_arr)[0]
        pred = int(self.predict(x_arr)[0])

        lines = ["Input membership degrees:"]
        for idx, fname in enumerate(self.features):
            m_str = ", ".join(
                f"{lbl}={mu[idx, t]:.2f}" for t, lbl in enumerate(DEFAULT_LABELS)
            )
            lines.append(f"  - {fname} ({x_arr[0, idx]:.2f}): {m_str}")

        lines.append("Activated rules (firing > 0):")
        active = [(firings[i], r) for i, r in enumerate(self.rules_) if firings[i] > 0.0]
        active.sort(key=lambda pair: pair[0], reverse=True)
        if not active:
            lines.append(f"  - None (fallback to majority class {self.majority_})")
        else:
            for f_val, r in active:
                lines.append(f"  - [f={f_val:.3f}] {format_rule(r, self.features)}")
        lines.append(f"Predicted class: {pred}")
        return "\n".join(lines)
