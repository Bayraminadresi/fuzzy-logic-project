# Bulanık Kural Tabanlı Modellerde Yorumlanabilirlik: Wine Veri Seti Üzerinde Yeniden Gerçekleştirme ve Post-hoc Kural Önemi Analizi

## Student Information
- Ad Soyad: _(doldurulacak)_
- Öğrenci No: _(doldurulacak)_
- Ders: Bulanık Mantık, Dönem Projesi

## Original Paper
L. Pickering, K. Cohen, B. De Baets, "A Narrative Review on the Interpretability of Fuzzy Rule-Based Models from a Modern Interpretable Machine Learning Perspective," *Int. J. Fuzzy Syst.*, 2025. DOI: 10.1007/s40815-025-02022-z

## Problem Description
Makale, bulanık kural tabanlı modellerin (FRBM) ana akım yorumlanabilir makine öğrenmesi literatüründe neden yer almadığını üç mercekle (ölçek, hedef kitle, model aşaması) inceler. Bölüm 7'de FisPro ile UCI Wine veri seti üzerinde kurulmuş küçük bir FRBM örneği (12 kural, 4 özellik) üç mercekle değerlendirilir.

Makale nicel sonuç, özellik listesi veya üyelik fonksiyonu parametresi vermez. Bu nedenle yeniden gerçekleştirme **yapısal düzeydedir**.

## Method (Plan)
- Baseline: Wine üzerinde 4 özellikli, uniform Strong Fuzzy Partition (3 üçgen terim), yaklaşık 12 kurallı Mamdani/linguistik FRBM (NumPy ile kendi motorumuz).
- Katkı K1: Post-hoc kural önemi (ortalama ateşleme gücü, kapsama) ve kural sayısı-doğruluk eğrisi.
- Katkı K2: Uniform SFP ile veriye göre yerleştirilmiş SFP karşılaştırması.
- Katkı K3 (opsiyonel): Defuzzification karşılaştırması.
- Değerlendirme: Tekrarlı stratified k-fold CV; kontrol modelleri karar ağacı ve lojistik regresyon.

## Fuzzy System Architecture
_(Aşama 4'te doldurulacak)_

## Dataset
UCI Wine (178 örnek, 13 özellik, 3 sınıf), `sklearn.datasets.load_wine` ile yüklenir.

## Installation / Requirements / How to Run
```
pip install -r requirements.txt
```
Notebook'lar Google Colab'da çalıştırılabilir. _(Detaylar sonra eklenecek.)_

## Reproduction Experiments
_(Aşama 4)_

## Proposed Extension
_(Aşama 5-6)_

## Experimental Results
_(Aşama 6)_

## Limitations
_(Aşama 7)_

## References
Bkz. [references/README.md](references/README.md)
