# Go 19×19 — Comparaison d'architectures sous contrainte de 100k paramètres

> Projet académique — Master 2 IASD, Université Paris-Dauphine PSL (2025–2026)
> Mohamed ZOUAD · Thomas SINAPI · Amine ROUIBI 

---

## Vue d'ensemble

Ce projet compare 14 architectures de réseaux de neurones profonds pour la prédiction de coups et l'évaluation de position dans le jeu de Go 19×19, sous une contrainte stricte de **100 000 paramètres maximum**. Toutes les architectures sont entraînées avec **exactement les mêmes hyperparamètres** — seule l'architecture varie — afin de garantir une comparaison équitable et de n'attribuer les différences de performance qu'aux seuls choix architecturaux.

La tâche est une prédiction à deux têtes : une **tête politique** produisant une distribution de probabilité sur les 361 intersections du plateau (softmax), et une **tête valeur** estimant la probabilité de victoire du joueur courant (sigmoïde scalaire).

Le meilleur modèle atteint **44,56 % de précision politique en validation**, soit un gain de +43,99 points de pourcentage par rapport au baseline MLP.

---

## Structure du projet

Le dépôt est organisé autour de **deux phases** (Phase 1 et Phase 2). Le code est principalement dans des **notebooks Jupyter**, et les **résultats** générés sont stockés dans `output/`.

```text
.
├── notebook/
│   ├── phase1.ipynb              # Expériences / entraînements - Phase 1
│   ├── phase2.ipynb              # Expériences / entraînements - Phase 2
│   └── rapport_architectures.md  # Notes/rapport sur les architectures testées
├── output/
│   ├── phase1/
│   │   ├── all_runs_summary.csv  # Récapitulatif des runs (Phase 1)
│   │   ├── model*/               # Dossiers générés par run (hyperparams, seed, timestamp...)
│   │   └── résultats phase 1.jpeg
│   └── phase2/
│       ├── all_runs_summary.csv  # Récapitulatif des runs (Phase 2)
│       ├── model*/               # Dossiers générés par run
│       └── résultats phase 2.jpeg
└── report.pdf                    # Rapport final du projet
```

### Notes
- Les dossiers `output/phase*/model...` correspondent à des exécutions (runs) identifiées par un nom incluant souvent le modèle, les hyperparamètres, la seed et un timestamp.
- Les fichiers `.DS_Store` sont des fichiers système macOS et peuvent être ignorés (recommandé : ajouter `.DS_Store` au `.gitignore`).

---

## Résultats

| Rang | Modèle | Phase | Données | Acc. politique |
|------|--------|-------|---------|----------------|
| 🥇 1 | M18 – EfficientFormer-Go | 3 | 50k | **44,56 %** |
| 🥈 2 | M11 – MBConv + SE | 3 | 50k | 44,39 % |
| 🥉 3 | M12 – Inverted + ECA | 3 | 50k | 43,11 % |
| 4 | M15 – ConvNeXt-Lite † | 3 | 50k | 41,19 % |
| 5 | M4 – ResNet Tiny | 2 | 50k | 35,29 % |
| 6 | M9 – ResNet + Dilated | 2 | 50k | 35,01 % |
| 7 | M3 – CNN + BatchNorm | 2 | 50k | 33,87 % |
| … | M1 – MLP Baseline | 1 | 10k | 0,57 % |

† M15 n'avait pas convergé à la fin du run (courbe encore croissante).

---

## Structure du projet

Le fichier `rapport_architectures.md` contient le code complet et documenté des 14 architectures (M1 à M18), ainsi que l'intégralité du pipeline d'entraînement partagé. Le rapport `GO_report.pdf` présente l'analyse comparative complète sur 15 pages.

---

## Protocole expérimental

Tous les modèles partagent la même infrastructure d'entraînement. L'optimiseur Adam est utilisé avec un taux d'apprentissage initial de 0,001, une régularisation L2 de 1e-4 et une graine aléatoire fixe (seed = 42, avec `TF_DETERMINISTIC_OPS=1` pour la reproductibilité). La taille de batch est de 128 par GPU. La précision mixte float16 est activée. L'arrêt précoce est déclenché après 20 époques sans amélioration de la perte de validation, et le taux d'apprentissage est réduit d'un facteur 0,5 après 10 époques de stagnation. Le nombre maximal d'époques est fixé à 300.

L'entraînement s'appuie sur Keras/TensorFlow 2 avec la stratégie `MirroredStrategy` pour l'utilisation multi-GPU sur Kaggle Notebooks (2×T4 ou P100). Le suivi des métriques est assuré par Weights & Biases (W&B) et TensorBoard. Les checkpoints et historiques CSV sont synchronisés automatiquement vers Google Drive à la fin de chaque run.

---

## Les 14 architectures

### Phase 1 — Exploration initiale (N = 10 000 positions)

Dix architectures couvrant un large spectre sont évaluées dans un premier temps.

**M1 – MLP Baseline** repose sur un réseau entièrement connecté : l'entrée est aplatie en un vecteur de 11 191 dimensions, puis passée par trois couches denses. Aucun biais inductif spatial n'est exploité. Il sert de borne inférieure de référence et atteint 0,57 % de précision.

**M2 – CNN Shallow** est un backbone de trois convolutions 3×3 (32 filtres, ReLU), sans normalisation de batch. Il constitue la référence CNN minimale et atteint 11,12 %.

**M3 – CNN + BatchNorm** est identique à M2 avec l'ajout d'une BatchNormalization après chaque convolution (Conv → BN → ReLU). L'ajout de normalisation multiplie par 2,7 la précision finale (29,87 %), illustrant l'impact décisif du conditionnement des gradients.

**M4 – ResNet Tiny** introduit des blocs résiduels avec connexions skip et une initialisation γ = 0 sur la dernière BatchNorm de chaque bloc (zero-init residuals). Il atteint 31,08 %.

**M5 – Depthwise Separable CNN** remplace chaque convolution standard par une convolution dépthwise suivie d'une pointwise, inspiré de MobileNetV1. Sous la contrainte de 100k paramètres, la capacité résiduelle s'avère insuffisante et le modèle sature dès l'époque 40 (0,39 %).

**M6 – CNN Asymétrique** applique un backbone CNN standard mais alloue plus de capacité (deux couches denses) à la tête politique qu'à la tête valeur (une seule couche dense). Cette asymétrie n'apporte pas de gain significatif (6,13 %).

**M7 – MobileNet-style** utilise des blocs d'inverted residuals avec un taux d'expansion ×6, inspirés de MobileNetV2. La structure bottleneck opère la convolution dans un espace de grande dimension puis projette dans un espace compact. Il atteint 12,19 %.

**M8 – ResNet + SE** enrichit les blocs résiduels de M4 avec des modules Squeeze-and-Excitation pour la recalibration canal par canal. Les paramètres consommés par les goulots SE réduisent trop la capacité convolutionnelle utile sous 100k : il n'atteint que 11,41 %, en dessous de M4.

**M9 – ResNet + Dilated Conv** est le meilleur modèle de la Phase 1 (32,71 %). Des convolutions dilatées avec taux [1, 2, 3] sont utilisées dans chaque bloc résiduel, élargissant le champ récepteur effectif jusqu'à 21×21 sans paramètres supplémentaires. Sur un plateau 19×19, chaque neurone peut ainsi voir la quasi-totalité de la grille dès les premières couches.

**M10 – CNN + Transformer** combine un backbone CNN avec un bloc d'attention multi-têtes (4 têtes) sur les 361 tokens spatiaux du plateau. Sous 100k paramètres, la portion allouée à l'attention est trop contrainte et le modèle sature dès l'époque 60 (0,39 %).

| Modèle | Architecture | Params | Acc. Phase 1 |
|--------|-------------|--------|--------------|
| M1 | MLP Baseline | 99 904 | 0,57 % |
| M2 | CNN Shallow | 97 388 | 11,12 % |
| M3 | CNN + BatchNorm | 99 752 | 29,87 % |
| M4 | ResNet Tiny | 99 628 | 31,08 % |
| M5 | Depthwise Separable CNN | 99 752 | 0,39 % |
| M6 | CNN Asymétrique | 99 752 | 6,13 % |
| M7 | MobileNet-style | 99 752 | 12,19 % |
| M8 | ResNet + SE | 99 628 | 11,41 % |
| M9 | ResNet + Dilated Conv | 99 628 | **32,71 %** |
| M10 | CNN + Transformer | 99 498 | 0,39 % |

![Résultats phase 1](output/phase1/résultats%20phase%201.jpeg)
---

### Phase 2 — Réentraînement des top 3 (N = 50 000 positions)

Les trois meilleures architectures (M3, M4, M9) sont réentraînées avec un volume de données multiplié par 5. Les performances progressent de 2,3 à 4,2 points selon le modèle. Fait notable, M4 dépasse M9 en Phase 2 (35,29 % vs 35,01 %), alors que M9 dominait en Phase 1 : avec plus de données, les blocs résiduels standards comblent l'avantage structurel des convolutions dilatées. Les trois architectures plafonnent néanmoins autour de 35 %.

| Modèle | Phase 1 | Phase 2 | Gain |
|--------|---------|---------|------|
| M4 – ResNet Tiny | 31,08 % | **35,29 %** | +4,21 pp |
| M9 – ResNet + Dilated | 32,71 % | 35,01 % | +2,30 pp |
| M3 – CNN + BatchNorm | 29,87 % | 33,87 % | +4,00 pp |

---

### Phase 3 — Architectures efficientes modernes (N = 50 000 positions)

Quatre architectures conçues pour maximiser le rapport performance/paramètre sont évaluées directement avec 50k positions.

**M11 – MBConv + SE** utilise des blocs MBConv (Mobile Inverted Bottleneck Convolution) avec expansion ×4, convolution séparable 3×3, activation Swish et module SE. La technique de stochastic depth régularise le modèle. Inspiré d'EfficientNet. Il atteint 44,39 %.

**M12 – Inverted Residual + ECA** est une variante de M11 remplaçant le module SE par l'attention par canal efficiente (ECA) : une convolution 1D de noyau adaptatif ne coûtant que 3 à 5 paramètres contre ≈ 1 600 pour un SE équivalent. Le budget économisé est réalloué aux couches convolutionnelles. Il atteint 43,11 %.

**M15 – ConvNeXt-Lite** adapte l'architecture ConvNeXt avec des noyaux dépthwise 7×7, une normalisation LayerNorm, une activation GELU et une expansion MLP ×4. Cette architecture modernise le CNN classique en incorporant les biais inductifs des Transformers dans un cadre entièrement convolutionnel. Il atteint 41,19 % sans avoir convergé.

**M18 – EfficientFormer-Go** est une architecture étagée hybride CNN-Transformer : les premières couches sont convolutionnelles et les dernières appliquent une attention multi-têtes sur des tokens poolés, réduisant le coût quadratique de l'attention. Il atteint le meilleur score global : **44,56 %**.

| Modèle | Architecture | Attention | Params | Acc. |
|--------|-------------|-----------|--------|------|
| M11 | MBConv + SE | SE | 98 242 | 44,39 % |
| M12 | Inverted Residual + ECA | ECA (≤5 params) | 93 773 | 43,11 % |
| M15 | ConvNeXt-Lite | — | 92 863 | 41,19 % † |
| M18 | EfficientFormer-Go | MHA poolé | 99 666 | **44,56 %** |

---

## Conclusions

Trois enseignements principaux se dégagent de cette étude.

Le **biais inductif spatial est indispensable** : les modèles sans biais spatial approprié échouent systématiquement, tandis que l'ajout d'une simple BatchNorm à un CNN multiplie par 2,7 la précision finale.

Le **volume de données amplifie les bonnes architectures** : multiplier les données par 5 apporte de 2 à 4 points de précision selon le modèle, mais ne suffit pas à dépasser le plafond des ResNet classiques (~35 %).

Le **choix architectural est le levier primaire** : passer des ResNet aux architectures efficientes modernes apporte +9 points supplémentaires à budget paramétrique et volume de données identiques. Les formes légères d'attention globale (SE, ECA, MHA poolé) sont compatibles avec la contrainte de 100k paramètres ; l'attention MHA standard sur la totalité des tokens ne l'est pas.

---

## Perspectives

Plusieurs axes d'amélioration restent à explorer : l'augmentation de données par les symétries du plateau de Go (rotations 90°, réflexions) permettrait de multiplier le corpus par 8 sans collecte supplémentaire. Remplacer ReduceLROnPlateau par un scheduler cosinus améliorerait la convergence des architectures modernes. Une recherche automatique d'architecture (NAS) sous contrainte de 100k paramètres pourrait identifier des configurations encore plus efficaces. Enfin, la distillation de connaissances — entraîner un petit modèle en imitant un grand modèle sans contrainte — constitue une piste prometteuse.

---

## Références

- Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 2016
- Howard et al., *MobileNets*, arXiv 2017
- Sandler et al., *MobileNetV2: Inverted residuals and linear bottlenecks*, CVPR 2018
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018
- Wang et al., *ECA-Net: Efficient channel attention for deep convolutional neural networks*, CVPR 2020
- Tan & Le, *EfficientNet: Rethinking model scaling for convolutional neural networks*, ICML 2019
- Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- Li et al., *EfficientFormer: Vision transformers at MobileNet speed*, NeurIPS 2022

---

*Cours Apprentissage Profond — M2 IASD — Université Paris-Dauphine PSL — 2025/2026*
