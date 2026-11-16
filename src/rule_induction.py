from dataclasses import dataclass
import numpy as np

# Projede kullanilan sabit 4 ozellik ve 3 dilsel terim
FEATURES = ["flavanoids", "proline", "color_intensity", "alcohol"]
DEFAULT_LABELS = ("Low", "Medium", "High")


@dataclass
class Rule:
    """Tek bir bulanik IF-THEN kuralini temsil eden veri yapisi."""
    antecedent: tuple[int, int, int, int]
    consequent: int
    weight: float = 1.0
    support: float = 0.0


def generate_candidate_rules(mu: np.ndarray, y: np.ndarray) -> list[Rule]:
    """Wang-Mendel yontemiyle (n, 4, 3) boyutlu uyelik tensorunden aday kurallari uretir."""
    y = np.asarray(y, dtype=int)
    # Her ornek ve her ozellik icin en yuksek uyelik derecesine sahip terim indeksi (0, 1 veya 2)
    best_terms = np.argmax(mu, axis=2)  # (n, 4)
    # Ornek bazinda kural aktivasyon derecesi (4 ozelligin maksimum uyeliklerinin carpimi)
    degrees = np.prod(np.max(mu, axis=2), axis=1)  # (n,)

    # Ayni oncul (antecedent) birden fazla ornekte cakisirsa en yuksek dereceli kural kazanir
    best_by_ant: dict[tuple[int, int, int, int], tuple[float, Rule]] = {}
    for i in range(len(y)):
        ant = tuple(int(t) for t in best_terms[i])
        deg = float(degrees[i])
        if ant not in best_by_ant or deg > best_by_ant[ant][0]:
            best_by_ant[ant] = (
                deg,
                Rule(antecedent=ant, consequent=int(y[i]), weight=deg, support=0.0),
            )
    return [item[1] for item in best_by_ant.values()]


def weight_and_prune(
    rules: list[Rule],
    mu: np.ndarray,
    y: np.ndarray,
    rules_per_class: int = 4,
) -> list[Rule]:
    """Kurallarin bulanik destek (support) ve guven (confidence) agirligini hesaplar, sinif basina en iyi kurallari secer."""
    y = np.asarray(y, dtype=int)
    n_features = mu.shape[1]
    weighted: list[Rule] = []

    for r in rules:
        # Kuralin tum ornekler uzerindeki minimum t-norm aktivasyonu: a_r(i) = min_f mu[i, f, ant_f]
        a_r = np.min(
            np.stack([mu[:, f, r.antecedent[f]] for f in range(n_features)], axis=1),
            axis=1,
        )
        total_act = float(np.sum(a_r))
        support = float(np.sum(a_r[y == r.consequent]))
        confidence = support / total_act if total_act > 0.0 else 0.0
        weighted.append(
            Rule(
                antecedent=r.antecedent,
                consequent=r.consequent,
                weight=confidence,
                support=support,
            )
        )

    # Her siniftan (support * confidence) skoruna gore en yuksek rules_per_class kurali tut (3 x 4 = 12 kural)
    pruned: list[Rule] = []
    for cls in sorted(np.unique(y)):
        cls_rules = [r for r in weighted if r.consequent == int(cls) and r.weight > 0.0]
        cls_rules.sort(key=lambda r: (r.support * r.weight, r.weight), reverse=True)
        pruned.extend(cls_rules[:rules_per_class])
    return pruned


def format_rule(
    rule: Rule,
    features: list[str] = FEATURES,
    labels: tuple[str, ...] = DEFAULT_LABELS,
) -> str:
    """Kurali okunabilir IF-THEN dilsel metnine donusturur."""
    conds = [f"{features[f]} is {labels[t]}" for f, t in enumerate(rule.antecedent)]
    return f"IF {' AND '.join(conds)} THEN class {rule.consequent} (w={rule.weight:.2f})"