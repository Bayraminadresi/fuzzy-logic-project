"""Mamdani tipi bulanık kural tabanlı sınıflandırıcı için çekirdek motor (NumPy).

Bu sürüm: yalnızca StrongFuzzyPartition (fuzzification).
"""
import numpy as np


class StrongFuzzyPartition:
    """Üçgen Strong Fuzzy Partition (Ruspini). Uçlar omuz (shoulder) şeklindedir."""

    def __init__(self, centers, labels=("Low", "Medium", "High")):
        self.centers = np.asarray(centers, dtype=float)
        assert len(self.centers) == len(labels)
        assert np.all(np.diff(self.centers) > 0), "Merkezler artan olmalı"
        self.labels = tuple(labels)

    @classmethod
    def uniform(cls, lo, hi, labels=("Low", "Medium", "High")):
        return cls(np.linspace(lo, hi, len(labels)), labels)

    def fuzzify(self, x):
        """x: (n,) -> (n, k) üyelik matrisi."""
        x = np.asarray(x, dtype=float)
        eye = np.eye(len(self.centers))
        # i. terim: merkezi i'de 1, diğer merkezlerde 0 olan parçalı doğrusal fonksiyon
        return np.stack([np.interp(x, self.centers, eye[i])
                         for i in range(len(self.centers))], axis=-1)
