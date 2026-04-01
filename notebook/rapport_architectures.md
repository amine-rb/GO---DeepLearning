# Architectures de réseaux de neurones pour le jeu de Go 19×19

---

## Infrastructure commune

### Environnement et données

Les données sont téléchargées depuis le serveur du LAMSADE (Dauphine) via le module `golois`, qui fournit des parties annotées par un moteur de force professionnelle. Chaque position est représentée par un tenseur de forme 19×19×31 (31 plans binaires). L'entraînement et la validation sont gérés via Kaggle Notebooks avec `MirroredStrategy` multi-GPU.

Le suivi des expériences est assuré par Weights & Biases (W&B) avec authentification via les secrets Kaggle, et par TensorBoard pour la visualisation locale. Les métriques sont enregistrées à chaque époque (perte totale, précision politique, MAE valeur). Les checkpoints `.keras` et les historiques CSV sont synchronisés automatiquement vers Google Drive à la fin de chaque run.

### Constantes globales

- Dossier de travail : `/kaggle/working/go_project3`
- Nombre de plans d'entrée (`PLANES`) : 31
- Nombre de coups possibles (`MOVES`) : 361
- Volume d'entraînement Phase 1 (`N`) : 10 000 positions

### Reproductibilité

La graine aléatoire est fixée à 42 pour Python, NumPy et TensorFlow, avec `TF_DETERMINISTIC_OPS=1` et `PYTHONHASHSEED` fixé. La fonction `set_seeds()` est appelée avant chaque construction de modèle pour garantir la reproductibilité entre les runs.

### Chargement des données

Les tableaux d'entrée (`input_data`, `policy`, `value`, `end`, `groups`) sont alloués en mémoire au format float32 puis remplis epoch par epoch via `golois.getBatch()`. Le jeu de validation est chargé une seule fois via `golois.getValidation()` et reste fixe pour toute l'étude.

### Pipeline d'entraînement universel

Tous les modèles passent par la même fonction `train_model(build_fn, config)`. Elle prend en entrée une fonction de construction du modèle et un dictionnaire de configuration, puis exécute la boucle d'entraînement epoch par epoch. À chaque époque, le batch est rechargé depuis `golois`, le modèle est entraîné, les métriques d'entraînement sont logguées dans W&B et TensorBoard, le modèle est évalué sur le jeu de validation, et le meilleur checkpoint est sauvegardé si la perte de validation s'améliore. L'arrêt anticipé est géré manuellement avec un compteur de patience (20 époques). Toutes les 10 époques, les courbes intermédiaires sont sauvegardées et synchronisées vers Drive.

---

## Hyperparamètres communs à tous les modèles

Tous les modèles partagent les hyperparamètres suivants, qui ne varient jamais :

- Optimiseur : Adam, lr = 0,001, clipnorm = 1,0
- Régularisation L2 : 1e-4 sur tous les noyaux
- Taille de batch : 128 × nombre de GPUs
- Précision numérique : mixed float16
- Graine aléatoire : 42
- Epochs maximum : 300
- Patience early stopping : 20 époques
- ReduceLROnPlateau : facteur 0,5, patience 10, min_lr = 1e-6
- Perte politique : entropie croisée catégorielle
- Perte valeur : MSE
- Métrique principale de comparaison : précision catégorielle politique en validation

---

## M1 — MLP Baseline

**Description structurelle :** Réseau entièrement connecté. L'entrée (19×19×31) est aplatie en un vecteur de 11 191 dimensions, puis passée par trois couches Dense successives (8, 16, 26 unités, ReLU). Les deux têtes de sortie (policy softmax 361 classes, value sigmoïde scalaire) partagent la même représentation aplatie. Aucune information spatiale n'est exploitée. Contrainte : < 100 000 paramètres (≈ 99 904 params).

**Hyperparamètres spécifiques :** hidden1 = 8, hidden2 = 16, hidden3 = 26.

**Résultat :** 0,57 % — le modèle n'apprend pratiquement rien. L'absence de biais inductif spatial empêche le réseau d'extraire les relations locales essentielles au jeu de Go.

---

## M2 — CNN Shallow

**Description structurelle :** Backbone composé de trois convolutions 3×3 (32 filtres, padding same, ReLU). Chaque tête (policy et value) applique ensuite une convolution 1×1 de réduction, un aplatissement, puis une couche Dense(64) avant la sortie finale. Architecture symétrique entre les deux têtes. Aucune normalisation de batch.

**Hyperparamètres spécifiques :** filters = 32, dense_head = 64. Paramètres totaux : ≈ 97 388.

**Résultat :** 11,12 % — convergence lente sur 200 époques. Sert de référence CNN minimale sans normalisation.

---

## M3 — CNN + BatchNorm

**Description structurelle :** Identique à M2, avec ajout d'une BatchNormalization après chaque convolution (Conv sans biais → BN → ReLU). La couche Dense des têtes est élargie de 64 à 66 unités pour maximiser le budget paramétrique sous 100k. La normalisation de batch stabilise les gradients et accélère la convergence.

**Hyperparamètres spécifiques :** filters = 32, dense_head = 66. Paramètres totaux : ≈ 99 752.

**Résultat :** 29,87 % — gain de ×2,7 par rapport à M2. Illustre l'impact décisif de la BatchNorm sur un CNN identique.

---

## M4 — ResNet Tiny

**Description structurelle :** Couche stem Conv(24, 3×3) + BN + ReLU, suivie de 3 blocs résiduels. Chaque bloc implémente le schéma Conv(no_bias) + BN + ReLU + Conv(no_bias) + BN + connexion identité (skip) + ReLU. L'initialisation `gamma=zeros` sur le dernier BN de chaque bloc (zero-init residuals) stabilise l'entraînement en préservant l'identité au début. Les deux têtes utilisent Dense(56).

**Hyperparamètres spécifiques :** filters = 24, n_blocks = 3, dense_head = 56. Paramètres totaux : ≈ 99 628.

**Résultat Phase 1 :** 31,08 % (époque 180). **Résultat Phase 2 :** 35,29 % (époque 170), meilleur modèle classique.

---

## M5 — Depthwise Separable CNN

**Description structurelle :** Backbone de 5 blocs DSConv, chacun composé d'une convolution depthwise 3×3 (opération spatiale par canal) suivie d'une convolution pointwise 1×1 (projection et mélange de canaux) et d'une activation ReLU. La factorisation dépthwise/pointwise réduit le coût paramétrique du backbone par rapport à des convolutions standards. Les têtes utilisent Dense(85).

**Hyperparamètres spécifiques :** filters = 32, n_layers = 5, dense_head = 85. Paramètres totaux : ≈ 99 576.

**Résultat :** 0,39 % — saturation dès l'époque 40. Sous la contrainte de 100k paramètres, la capacité résiduelle après séparation en profondeur est insuffisante pour la complexité de la tâche.

---

## M6 — CNN Asymétrique

**Description structurelle :** Backbone CNN identique à M2 avec 24 filtres. Architecture asymétrique des têtes : la tête politique reçoit deux couches Dense(87) successives (hypothèse que la prédiction du coup est plus difficile que l'évaluation de position), tandis que la tête valeur ne dispose que d'une couche Dense(32).

**Hyperparamètres spécifiques :** filters = 24, dense_policy = 87 (×2), dense_value = 32. Paramètres totaux : ≈ 99 721.

**Résultat :** 6,13 % — l'asymétrie des têtes n'apporte pas le gain escompté. Le backbone trop léger (24 filtres vs 32 pour M2) pénalise les deux têtes.

---

## M7 — MobileNet-style (Inverted Residuals)

**Description structurelle :** Couche stem Conv(16) + ReLU, suivie de 4 blocs d'inverted residuals inspirés de MobileNetV2. Chaque bloc applique : expansion pointwise 1×1 (×6 canaux), convolution depthwise 3×3 spatiale, projection pointwise 1×1 vers les canaux cibles, puis connexion identité si les dimensions d'entrée et de sortie sont identiques. Pas d'activation sur la projection (style MobileNetV2 linear bottleneck). Les têtes utilisent Dense(72).

**Hyperparamètres spécifiques :** filters = 16, expand = 6, n_blocks = 4, dense_head = 72. Paramètres totaux : ≈ 99 644.

**Résultat :** 12,19 % (époque 50, early stop). Malgré l'efficacité paramétrique des inverted residuals, l'architecture ne tire pas encore parti de cette structure sous 100k paramètres avec N = 10k.

---

## M8 — ResNet + Squeeze-and-Excitation

**Description structurelle :** Architecture ResNet Tiny (M4) augmentée d'un module Squeeze-and-Excitation dans chaque bloc résiduel. Le SE applique un Global Average Pooling sur les feature maps, puis deux couches Dense (réduction C → C/4 → C) pour produire des coefficients d'échelle par canal (activation sigmoid), qui pondèrent les feature maps avant la connexion skip. Les têtes utilisent Dense(55).

**Hyperparamètres spécifiques :** filters = 24, n_blocks = 3, se_ratio = 4, dense_head = 55. Paramètres totaux : ≈ 99 496.

**Résultat :** 11,41 % — en dessous de M4 (31,08 %). Les paramètres consommés par les goulots SE réduisent trop la capacité convolutionnelle utile sous ce budget.

---

## M9 — ResNet + Dilated Convolutions

**Description structurelle :** Architecture ResNet Tiny (M4) avec des convolutions dilatées croissantes dans les blocs résiduels. Chaque bloc résiduel utilise deux convolutions dilatées avec le même taux de dilatation (pas de biais, BN) et une connexion identité. Les taux [1, 2, 3] augmentent progressivement le champ récepteur effectif (respectivement 3×3, 7×7 et 13×13 par bloc, combinés jusqu'à 21×21) sans ajouter de paramètres. Sur un plateau 19×19, chaque neurone peut voir la quasi-totalité de la grille dès les premières couches. Les têtes utilisent Dense(56).

**Hyperparamètres spécifiques :** filters = 24, dilations = [1, 2, 3], dense_head = 56. Paramètres totaux : ≈ 99 628.

**Résultat Phase 1 :** 32,71 % (époque 230) — meilleur modèle de Phase 1. **Résultat Phase 2 :** 35,01 % (époque 240).

---

## M10 — CNN + Transformer Encoder

**Description structurelle :** Backbone de deux convolutions Conv(16, 3×3, ReLU) pour l'extraction de features locales. Les feature maps (19×19×16) sont reshapées en une séquence de 361 tokens de dimension 16 (une intersection = un token). Un bloc Transformer Encoder en style pre-norm est appliqué : LayerNormalization → Multi-Head Attention (4 têtes, key_dim = 16, self-attention) → connexion résiduelle. La sortie est reshapée en 19×19×16 pour les têtes Conv/Dense(81) standards.

**Hyperparamètres spécifiques :** filters = 16, mha_heads = 4, key_dim = 16, dense_head = 81. Paramètres totaux : ≈ 99 498.

**Résultat :** 0,39 % — saturation dès l'époque 60. Le mécanisme d'attention MHA sur 361 tokens requiert un bien plus grand nombre de têtes ou de dimensions pour être efficace ; sous 100k paramètres, la portion allouée à l'attention est trop contrainte.