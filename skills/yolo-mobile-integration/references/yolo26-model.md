# Référence Complète du Modèle YOLO26

> Guide de référence complet pour l'intégration mobile de YOLO26 — architecture, tâches, tailles, entraînement et compression.

---

## Table des matières

1. [Aperçu et Innovations Architecturales](#1-aperçu-et-innovations-architecturales)
2. [Les 6 Variantes de Tâches](#2-les-6-variantes-de-tâches)
3. [Les 5 Tailles de Modèles](#3-les-5-tailles-de-modèles)
4. [Métriques de Performance COCO](#4-métriques-de-performance-coco)
5. [Configuration YAML pour Architectures Personnalisées](#5-configuration-yaml-pour-architectures-personnalisées)
6. [Recette d'Entraînement YOLO26](#6-recette-dentraînement-yolo26)
7. [Distillation de Connaissances pour Compression Mobile](#7-distillation-de-connaissances-pour-compression-mobile)
8. [Quand Utiliser Pré-entraîné vs Entraînement Personnalisé](#8-quand-utiliser-pré-entraîné-vs-entraînement-personnalisé)
9. [YOLOE-26 : Détection Vocabulaire Ouvert](#9-yoloe-26--détection-vocabulaire-ouvert)
10. [Exemples de Code Complets](#10-exemples-de-code-complets)

---

## 1. Aperçu et Innovations Architecturales

YOLO26 est une famille unifiée de modèles de vision en temps réel décrite dans [Ultralytics YOLO26](https://arxiv.org/abs/2606.03748). Elle introduit des changements fondamentaux par rapport aux générations précédentes.

### Les 4 Piliers de Conception

#### 1.1 Inférence Native de Bout en Bout (End-to-End sans NMS)

La tête de détection par défaut **one-to-one** génère des prédictions **sans suppression non maximale (NMS)**, simplifiant le déploiement et réduisant le post-traitement.

```python
from ultralytics import YOLO

# Inférence end-to-end par défaut (pas de NMS requis)
model = YOLO("yolo26n.pt")
results = model("image.jpg")  # Sortie: (N, 300, 6) max 300 détections
```

**Architecture à double tête :**
- **Tête One-to-One (par défaut)** : Sortie `(N, 300, 6)` — inférence rapide, pas de NMS
- **Tête One-to-Many** : Sortie `(N, nc+4, 8400)` — précision légèrement supérieure, NMS requis

```python
# Basculer vers la tête one-to-many (précision maximale)
results = model.predict("image.jpg", end2end=False)
model.export(format="onnx", end2end=False)
```

#### 1.2 Régression sans DFL

YOLO26 supprime la Distribution Focal Loss (DFL), réduisant la complexité de la tête de détection et simplifiant l'exportation. Le gain `dfl` est réutilisé pour pondérer une perte L1 sur les distances de boîte normalisées.

#### 1.3 Optimiseur MuSGD

Optimiseur hybride combinant **SGD** avec des mises à jour orthogonalisées de style **Muon** pour les matrices de poids (paramètres avec `ndim >= 2`).

```python
# L'optimiseur MuSGD est sélectionné automatiquement
# pour les entraînements > 10 000 itérations
model.train(data="coco8.yaml", epochs=100)
```

| Composant | Description |
|-----------|-------------|
| `muon_w` | Poids de mise à jour Muon dans MuSGD |
| `sgd_w` | Poids de mise à jour SGD dans MuSGD |
| `cls_w` | Poids de classification interne |

#### 1.4 Progressive Loss + STAL

- **Progressive Loss** déplace l'accent de l'entraînement vers la tête d'inférence
- **STAL** améliore la couverture des étiquettes positives pour les petits objets

---

## 2. Les 7 Variantes de Tâches

| Modèle | Noms de fichiers | Tâche | Inférence | Val. | Train | Export |
|--------|------------------|-------|-----------|------|-------|--------|
| **YOLO26** | `yolo26n.pt` `yolo26s.pt` `yolo26m.pt` `yolo26l.pt` `yolo26x.pt` | Détection | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-seg** | `yolo26n-seg.pt` `yolo26s-seg.pt` `yolo26m-seg.pt` `yolo26l-seg.pt` `yolo26x-seg.pt` | Segmentation d'instance | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-sem** | `yolo26n-sem.pt` `yolo26s-sem.pt` `yolo26m-sem.pt` `yolo26l-sem.pt` `yolo26x-sem.pt` | Segmentation sémantique | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-depth** | `yolo26n-depth.pt` `yolo26s-depth.pt` `yolo26m-depth.pt` `yolo26l-depth.pt` `yolo26x-depth.pt` | Estimation de profondeur | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-cls** | `yolo26n-cls.pt` `yolo26s-cls.pt` `yolo26m-cls.pt` `yolo26l-cls.pt` `yolo26x-cls.pt` | Classification | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-pose** | `yolo26n-pose.pt` `yolo26s-pose.pt` `yolo26m-pose.pt` `yolo26l-pose.pt` `yolo26x-pose.pt` | Estimation de pose | ✅ | ✅ | ✅ | ✅ |
| **YOLO26-obb** | `yolo26n-obb.pt` `yolo26s-obb.pt` `yolo26m-obb.pt` `yolo26l-obb.pt` `yolo26x-obb.pt` | Détection orientée (OBB) | ✅ | ✅ | ✅ | ✅ |

### Variantes architecturales (YAML uniquement)

- `yolo26-p2.yaml` — tête P2 pour les **petits objets**
- `yolo26-p6.yaml` — tête P6 pour les **entrées large**

Aucun poids `.pt` n'est publié pour ces variantes. Il faut les instancier et les entraîner :

```python
from ultralytics import YOLO

# Créer un modèle P6 pour entrées large
model = YOLO("yolo26n-p6.yaml")
model.train(data="your-dataset.yaml", epochs=100)
```

### Exemples d'utilisation par tâche

```python
from ultralytics import YOLO

# Détection
model = YOLO("yolo26m.pt")
results = model("image.jpg")

# Segmentation d'instance
model = YOLO("yolo26m-seg.pt")
results = model("image.jpg")

# Segmentation sémantique
model = YOLO("yolo26m-sem.pt")
results = model("image.jpg")

# Estimation de profondeur (monoculaire, en mètres)
model = YOLO("yolo26m-depth.pt")
results = model("image.jpg")

# Classification
model = YOLO("yolo26m-cls.pt")
results = model("image.jpg")

# Estimation de pose (17 keypoints COCO par défaut)
model = YOLO("yolo26m-pose.pt")
results = model("image.jpg")

# Détection orientée
model = YOLO("yolo26m-obb.pt")
results = model("image.jpg")
```

---

## 3. Les 5 Tailles de Modèles

| Taille | Paramètres | FLOPs | ID | Cas d'utilisation mobile |
|--------|-----------|-------|-----|--------------------------|
| **Nano (n)** | ~3M | ~8G | N | Edge, mobile, temps réel CPU |
| **Small (s)** | ~11M | ~30G | S | Équilibre vitesse/précision |
| **Medium (m)** | ~20M | ~55G | M | Précision avec calcul modéré |
| **Large (l)** | ~26M | ~80G | L | Haute précision, GPU disponible |
| **Extra-Large (x)** | ~57M | ~170G | X | Précision maximale, serveur |

### Recommandations par cible de déploiement

| Modèle | Idéal pour | Batch conseillé |
|--------|------------|-----------------|
| YOLO26n | Appareils Edge, mobile, temps réel CPU | Grands lots (64-128) GPU |
| YOLO26s | Équilibre vitesse/précision | Lots moyens (32-64) |
| YOLO26m | Précision améliorée | Petits lots (16-32) |
| YOLO26l | Haute précision GPU | Petits lots (8-16) ou multi-GPU |
| YOLO26x | Précision maximale serveur | Petits lots (4-8) ou multi-GPU |

---

## 4. Métriques de Performance COCO

### Détection sur COCO

| Modèle | mAP val 50-95 | Latence T4 TensorRT (ms) | Params (M) | FLOPs (G) |
|--------|---------------|--------------------------|------------|-----------|
| YOLO26n | 40,9 | 1,7 | ~3 | ~8 |
| YOLO26s | 48,6 | ~3 | ~11 | ~30 |
| YOLO26m | 53,1 | ~5 | ~20 | ~55 |
| YOLO26l | 55,0 | ~8 | ~26 | ~80 |
| YOLO26x | 57,5 | 11,8 | ~57 | ~170 |

### Améliorations par rapport à YOLO11

- **Jusqu'à +43% d'inférence CPU ONNX plus rapide** pour YOLO26n vs YOLO11n (CPU Intel Xeon @ 2.00 GHz)
- **+2,5 AP boîte et +3,7 AP masque** sur segmentation d'instance COCO
- **+7,2 AP** sur estimation de pose COCO
- **+3,4 mAP** sur détection orientée DOTA-v1.0

> Les valeurs de paramètres et FLOPs concernent le modèle fusionné après `model.fuse()`.

---

## 5. Configuration YAML pour Architectures Personnalisées

### Structure de base

```yaml
# Parameters
nc: 80                          # Nombre de classes
scales:                         # Facteurs [depth, width, max_channels]
    n: [0.50, 0.25, 1024]      # Nano
    s: [0.50, 0.50, 1024]      # Small
    m: [0.50, 1.00, 512]       # Medium
    l: [1.00, 1.00, 512]       # Large
    x: [1.00, 1.50, 512]       # Extra-Large
kpt_shape: [17, 3]             # Pose seulement (x, y, visibility)

backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]]      # 0: Convolution initiale
    - [-1, 1, Conv, [128, 3, 2]]     # 1: Sous-échantillonnage
    - [-1, 3, C2f, [128, True]]      # 2: Extraction de caractéristiques
    - [-1, 1, Conv, [256, 3, 2]]     # 3: P3/8
    - [-1, 6, C2f, [256, True]]      # 4
    - [-1, 1, Conv, [512, 3, 2]]     # 5: P4/16
    - [-1, 6, C2f, [512, True]]      # 6
    - [-1, 1, Conv, [1024, 3, 2]]    # 7: P5/32
    - [-1, 3, C2f, [1024, True]]     # 8
    - [-1, 1, SPPF, [1024, 5]]       # 9

head:
    - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 10
    - [[-1, 6], 1, Concat, [1]]                  # 11
    - [-1, 3, C2f, [512]]                        # 12
    - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 13
    - [[-1, 4], 1, Concat, [1]]                  # 14
    - [-1, 3, C2f, [256]]                        # 15
    - [-1, 1, Conv, [256, 3, 2]]                 # 16
    - [[-1, 12], 1, Concat, [1]]                 # 17
    - [-1, 3, C2f, [512]]                        # 18
    - [-1, 1, Conv, [512, 3, 2]]                 # 19
    - [[-1, 9], 1, Concat, [1]]                  # 20
    - [-1, 3, C2f, [1024]]                       # 21
    - [[15, 18, 21], 1, Detect, [nc]]            # 22
```

### Format de spécification des couches : `[from, repeats, module, args]`

| Composant | Objectif | Exemples |
|-----------|----------|----------|
| `from` | Connexions d'entrée | `-1` (précédent), `6` (couche 6), `[4, 6, 8]` (multiples) |
| `repeats` | Nombre de répétitions | `1`, `3` |
| `module` | Type de module | `Conv`, `C2f`, `SPPF`, `Detect` |
| `args` | Arguments du module | `[64, 3, 2]` (canaux, noyau, foulée) |

### Modules disponibles

| Module | Objectif | Arguments |
|--------|----------|-----------|
| `Conv` | Convolution + BatchNorm + Activation | `[out_ch, kernel, stride]` |
| `C2f` | Goulot d'étranglement CSP | `[out_ch, shortcut, expansion]` |
| `SPPF` | Spatial Pyramid Pooling rapide | `[out_ch, kernel_size]` |
| `Concat` | Concaténation par canal | `[dimension]` |
| `Detect` | Tête de détection YOLO | `[nc]` |
| `TorchVision` | Modèle TorchVision | `[out_ch, model_name, ...]` |
| `Index` | Sélection de caractéristiques | `[out_ch, index]` |

### Exemple : Modèle avec backbone TorchVision

```yaml
nc: 80

backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]]

head:
    - [0, 1, Index, [192, 4]]   # P3
    - [0, 1, Index, [384, 6]]   # P4
    - [0, 1, Index, [768, 8]]   # P5
    - [[1, 2, 3], 1, Detect, [nc]]
```

### Module personnalisé

```python
# 1. Définir dans ultralytics/nn/modules/block.py
class CustomBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layers(x)

# 2. Exposer dans __init__.py
from .block import CustomBlock

# 3. Ajouter aux imports dans tasks.py
from ultralytics.nn.modules import CustomBlock

# 4. Gérer dans parse_model()
if m is CustomBlock:
    c1, c2 = ch[f], args[0]
    args = [c1, c2, *args[1:]]

# 5. Utiliser dans le YAML
# custom_model.yaml
nc: 1
backbone:
    - [-1, 1, CustomBlock, [64]]
head:
    - [-1, 1, Classify, [nc]]
```

---

## 6. Recette d'Entraînement YOLO26

### Hyperparamètres optimisés par taille

#### Optimiseur et taux d'apprentissage

| Paramètre | N | S | M | L | X |
|-----------|---|---|---|---|---|
| `optimizer` | MuSGD | MuSGD | MuSGD | MuSGD | MuSGD |
| `lr0` | 0.0054 | 0.00038 | 0.00038 | 0.00038 | 0.00038 |
| `lrf` | 0.0495 | 0.882 | 0.882 | 0.882 | 0.882 |
| `momentum` | 0.947 | 0.948 | 0.948 | 0.948 | 0.948 |
| `weight_decay` | 0.00064 | 0.00027 | 0.00027 | 0.00027 | 0.00027 |
| `warmup_epochs` | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| `epochs` | 245 | 70 | 80 | 60 | 40 |
| `batch` | 128 | 128 | 128 | 128 | 128 |
| `imgsz` | 640 | 640 | 640 | 640 | 640 |

> Le modèle N utilise un lr plus élevé avec décroissance abrupte (`lrf=0.0495`), tandis que S/M/L/X utilisent un lr plus bas avec planification plus douce (`lrf=0.882`).

#### Poids de perte

| Paramètre | N | S | M | L | X |
|-----------|---|---|---|---|---|
| `box` | 5.63 | 9.83 | 9.83 | 9.83 | 9.83 |
| `cls` | 0.56 | 0.65 | 0.65 | 0.65 | 0.65 |
| `dfl` | 9.04 | 0.96 | 0.96 | 0.96 | 0.96 |

#### Augmentation des données

| Paramètre | N | S | M | L | X |
|-----------|---|---|---|---|---|
| `mosaic` | 0.909 | 0.992 | 0.992 | 0.992 | 0.992 |
| `mixup` | 0.012 | 0.05 | 0.427 | 0.427 | 0.427 |
| `copy_paste` | 0.075 | 0.404 | 0.304 | 0.404 | 0.404 |
| `scale` | 0.562 | 0.9 | 0.95 | 0.95 | 0.95 |
| `fliplr` | 0.606 | 0.304 | 0.304 | 0.304 | 0.304 |
| `degrees` | 1.11 | ~0 | ~0 | ~0 | ~0 |
| `shear` | 1.46 | ~0 | ~0 | ~0 | ~0 |
| `translate` | 0.071 | 0.275 | 0.275 | 0.275 | 0.275 |
| `hsv_h` | 0.014 | 0.013 | 0.013 | 0.013 | 0.013 |
| `hsv_s` | 0.645 | 0.353 | 0.353 | 0.353 | 0.353 |
| `hsv_v` | 0.566 | 0.194 | 0.194 | 0.194 | 0.194 |

### Inspecter les hyperparamètres d'un checkpoint

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
print(model.ckpt["train_args"])
# Affiche batch, box, cls, dfl, epochs, lr0, lrf, optimizer, etc.
```

### Fine-tuning de base

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

### Réglages selon la taille du jeu de données

**Petits jeux de données (< 1000 images) :**
```python
model.train(
    data="small-dataset.yaml",
    epochs=50,
    patience=20,
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
    lr0=0.001,
    freeze=10
)
```

**Grands jeux de données (> 50 000 images) :**
```python
model.train(
    data="large-dataset.yaml",
    epochs=80,
    mosaic=1.0,
    mixup=0.3,
    scale=0.9,
    optimizer="MuSGD"
)
```

---

## 7. Distillation de Connaissances pour Compression Mobile

La distillation de connaissances transfère les connaissances d'un **modèle enseignant** large vers un **modèle étudiant** plus petit, idéal pour le déploiement mobile.

### Paires recommandées

| Étudiant (mobile) | Enseignant |
|--------------------|------------|
| `yolo26n.pt` | `yolo26s.pt` |
| `yolo26s.pt` | `yolo26m.pt` |
| `yolo26m.pt` | `yolo26x.pt` |
| `yolo26l.pt` | `yolo26x.pt` |

> La distillation inter-familles (ex: YOLO11 → YOLO26) n'est **pas prise en charge**.

### Code de distillation

```python
from ultralytics import YOLO

# Entraîner un étudiant YOLO26n avec YOLO26s comme enseignant
model = YOLO("yolo26n.pt")
results = model.train(
    data="coco8.yaml",
    epochs=100,
    distill_model="yolo26s.pt"  # Active la distillation
)
```

### Paramètres de distillation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `distill_model` | `str` | `None` | Chemin vers le modèle enseignant |
| `dis` | `float` | `6.0` | Poids de la perte de distillation |

```python
# Ajuster le poids de distillation
model.train(
    data="coco8.yaml",
    epochs=100,
    distill_model="yolo26s.pt",
    dis=10.0  # Poids plus élevé = plus d'accent sur la distillation
)
```

### Résultats de distillation sur COCO

| Modèle | mAP référence | mAP distillé | Gain |
|--------|---------------|--------------|------|
| YOLO26n-distill | 40,9 | **41,5** | +0,6 |
| YOLO26s-distill | 48,6 | **49,2** | +0,6 |
| YOLO26m-distill | 53,1 | **53,9** | +0,8 |
| YOLO26l-distill | 55,0 | **56,0** | +1,0 |
| YOLO26x-distill | 57,5 | **57,9** | +0,4 |

### Impact sur les performances

- Entraînement **1.2 à 1.5x plus lent**
- **~1.1x plus de mémoire GPU** (enseignant en mode eval, sans gradients)
- Utiliser `amp=True` pour réduire l'impact

### Support des tâches

| Tâche | Supporté | Vérifié expérimentalement |
|-------|----------|---------------------------|
| detect | ✅ | ✅ |
| segment | ✅ | ⚠️ Pas encore validé |
| pose | ✅ | ⚠️ Pas encore validé |
| obb | ✅ | ⚠️ Pas encore validé |
| classify | ❌ | — |
| semantic | ❌ | — |

---

## 8. Quand Utiliser Pré-entraîné vs Entraînement Personnalisé

### Modèles pré-entraînés (recommandé pour commencer)

```python
from ultralytics import YOLO

# Charger un modèle pré-entraîné sur COCO
model = YOLO("yolo26n.pt")
results = model("image.jpg")
```

**Utiliser un modèle pré-entraîné quand :**
- Les classes de ton jeu de données sont similaires à COCO
- Tu as peu de données d'entraînement
- Tu veux un point de départ rapide
- Le domaine est proche de l'imagerie naturelle

### Entraînement à partir de zéro

```python
# Charger une architecture YAML (pas de poids)
model = YOLO("yolo26n.yaml")
model.train(data="custom-dataset.yaml", epochs=200)
```

**Entraîner à partir de zéro quand :**
- Le domaine est fondamentalement différent de COCO (médical, satellite, radar)
- Tu as un grand jeu de données annoté
- Les caractéristiques pré-entraînées ne sont pas transférables

### Tableau comparatif

| Critère | Fine-Tuning | Entraînement à partir de zéro |
|---------|-------------|-------------------------------|
| **Poids de départ** | Pré-entraîné COCO (80 classes) | Initialisation aléatoire |
| **Convergence** | Rapide | Lente |
| **Données requises** | Peuvent être limitées | Beaucoup nécessaires |
| **Commande** | `YOLO("yolo26n.pt")` | `YOLO("yolo26n.yaml")` |

### Gel des couches pour le fine-tuning

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Geler les 10 premières couches du backbone
model.train(data="custom.yaml", epochs=50, freeze=10)
```

| Scénario | Gel recommandé |
|----------|----------------|
| Grand jeu de données, domaine similaire | `freeze=None` |
| Petit jeu de données, domaine similaire | `freeze=10` |
| Très petit jeu de données | `freeze=23` |
| Domaine éloigné de COCO | `freeze=None` |

### Fine-tuning en deux étapes (domaines éloignés)

```python
from ultralytics import YOLO

# Étape 1 : Geler le backbone, entraîner neck + tête
model = YOLO("yolo26n.pt")
model.train(data="medical.yaml", epochs=20, freeze=10, name="stage1", exist_ok=True)

# Étape 2 : Tout dégeler, fine-tuning avec lr faible
model = YOLO("runs/detect/stage1/weights/best.pt")
model.train(data="medical.yaml", epochs=30, lr0=0.001, name="stage2", exist_ok=True)
```

---

## 9. YOLOE-26 : Détection Vocabulaire Ouvert

YOLOE-26 étend YOLO26 avec des capacités de **vocabulaire ouvert** — détection et segmentation de catégories d'objets en ensemble ouvert via des **prompts textuels** ou **visuels**.

### Performance sur LVIS

| Modèle | Prompt textuel | Prompt visuel | Sans prompt |
|--------|---------------|---------------|-------------|
| YOLOE-26x | 40,6 AP | 38,5 AP | 31,1 AP |

### Exemple avec prompts textuels

```python
from ultralytics import YOLO

model = YOLO("yoloe-26l-seg.pt")
model.set_classes(["person", "bus", "car"])
results = model("image.jpg")
results[0].show()
```

---

## 10. Exemples de Code Complets

### Pipeline complet mobile : entraînement → validation → export

```python
from ultralytics import YOLO

# 1. Entraînement avec distillation
model = YOLO("yolo26n.pt")
model.train(
    data="my-mobile-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    distill_model="yolo26s.pt",
    device=0
)

# 2. Validation
metrics = model.val(data="my-mobile-dataset.yaml")
print(f"mAP50-95: {metrics.box.map}")

# 3. Inférence
results = model("test_image.jpg")

# 4. Export pour mobile
model.export(format="onnx", quantize=8, data="my-mobile-dataset.yaml")
```

### Export vers tous les formats mobiles

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# ONNX (universel)
model.export(format="onnx")

# TensorRT (GPU NVIDIA)
model.export(format="engine", half=True)

# CoreML (iOS)
model.export(format="coreml")

# LiteRT/TFLite (Android)
model.export(format="litert")

# NCNN (mobile léger)
model.export(format="ncnn")

# OpenVINO (Intel)
model.export(format="openvino")

# Edge TPU (Google Coral)
model.export(format="edgetpu")

# ExecuTorch (Meta)
model.export(format="executorch")
```

### Résumé des imports

```python
from ultralytics import YOLO
# C'est tout ! Le package ultralytics gère tout :
# - Chargement des modèles
# - Entraînement / Fine-tuning
# - Validation
# - Inférence
# - Export
# - Distillation
```

---

## Références

- [Document YOLO26](https://arxiv.org/abs/2606.03748)
- [Documentation Ultralytics](https://docs.ultralytics.com/fr)
- [Dépôt GitHub](https://github.com/ultralytics/ultralytics)
- [Licence AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [Licence Enterprise](https://www.ultralytics.com/license)
