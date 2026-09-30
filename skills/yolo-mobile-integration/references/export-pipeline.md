# Référence Complète du Pipeline d'Export YOLO26

> Guide complet du pipeline d'exportation pour le déploiement mobile — de la préparation des données au benchmark.

---

## Table des matières

1. [Pipeline Complet](#1-pipeline-complet)
2. [Tous les Formats d'Exportation](#2-tous-les-formats-dexportation)
3. [Options de Quantification](#3-options-de-quantification)
4. [Code d'Export pour Chaque Format Mobile](#4-code-dexport-pour-chaque-format-mobile)
5. [Commandes de Benchmarking](#5-commandes-de-benchmarking)
6. [Préparation des Données](#6-préparation-des-données)
7. [Commandes d'Entraînement](#7-commandes-dentraînement)
8. [Workflow de Validation](#8-workflow-de-validation)
9. [Dépannage des Problèmes d'Export](#9-dépannage-des-problèmes-dexport)

---

## 1. Pipeline Complet

```
Préparation des Données → Entraînement → Validation → Export → Benchmark
```

### Vue d'ensemble du pipeline

```python
from ultralytics import YOLO

# ═══════════════════════════════════════════
# ÉTAPE 1 : PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════
# Créer un fichier data.yaml (voir section 6)

# ═══════════════════════════════════════════
# ÉTAPE 2 : ENTRAÎNEMENT
# ═══════════════════════════════════════════
model = YOLO("yolo26n.pt")
model.train(
    data="my-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device=0
)

# ═══════════════════════════════════════════
# ÉTAPE 3 : VALIDATION
# ═══════════════════════════════════════════
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="my-dataset.yaml")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# ═══════════════════════════════════════════
# ÉTAPE 4 : EXPORT
# ═══════════════════════════════════════════
model.export(format="onnx", quantize=8, data="my-dataset.yaml")

# ═══════════════════════════════════════════
# ÉTAPE 5 : BENCHMARK
# ═══════════════════════════════════════════
!yolo benchmark model=yolo26n.onnx imgsz=640
```

### Pipeline en ligne de commande

```bash
# Étape 1 : Entraînement
yolo detect train data=my-dataset.yaml model=yolo26n.pt epochs=100 imgsz=640

# Étape 2 : Validation
yolo detect val model=runs/detect/train/weights/best.pt data=my-dataset.yaml

# Étape 3 : Inférence
yolo detect predict model=runs/detect/train/weights/best.pt source=image.jpg

# Étape 4 : Export
yolo export model=runs/detect/train/weights/best.pt format=onnx quantize=8

# Étape 5 : Benchmark
yolo benchmark model=runs/detect/train/weights/best.pt imgsz=640
```

---

## 2. Tous les Formats d'Exportation

### Tableau complet des 22 formats

| Format | Argument `format` | Modèle généré | Mobile? | Quantification |
|--------|-------------------|---------------|---------|----------------|
| **PyTorch** | `-` | `yolo26n.pt` | ❌ | ❌ |
| **TorchScript** | `torchscript` | `yolo26n.torchscript` | ⚠️ | FP16 GPU |
| **ONNX** | `onnx` | `yolo26n.onnx` | ✅ | FP16, INT8 |
| **OpenVINO** | `openvino` | `yolo26n_openvino_model/` | ✅ | FP16, INT8 |
| **TensorRT** | `engine` | `yolo26n.engine` | ⚠️ GPU | FP16, INT8 |
| **CoreML** | `coreml` | `yolo26n.mlpackage` | ✅ iOS | FP16, INT8, W8A16 |
| **Core AI** | `coreai` | `yolo26n.aimodel` | ✅ iOS 27+ | INT8 |
| **TF SavedModel** | `saved_model` | `yolo26n_saved_model/` | ✅ | INT8 |
| **TF GraphDef** | `pb` | `yolo26n.pb` | ❌ | ❌ |
| **Edge TPU** | `edgetpu` | `yolo26n_edgetpu.tflite` | ✅ | INT8 auto |
| **PaddlePaddle** | `paddle` | `yolo26n_paddle_model/` | ❌ | ❌ |
| **MNN** | `mnn` | `yolo26n.mnn` | ✅ | FP16, INT8 |
| **NCNN** | `ncnn` | `yolo26n_ncnn_model/` | ✅ | FP16 |
| **IMX500** | `imx` | `yolo26n_imx_model/` | ✅ | INT8 auto |
| **RKNN** | `rknn` | `yolo26n_rknn_model/` | ✅ | FP16, INT8 |
| **ExecuTorch** | `executorch` | `yolo26n_executorch_model/` | ✅ | ❌ |
| **Axelera** | `axelera` | `yolo26n_axelera_model/` | ✅ | INT8 auto |
| **DEEPX** | `deepx` | `yolo26n_deepx_model/` | ✅ | INT8 auto |
| **Qualcomm QNN** | `qnn` | `yolo26n_qnn.onnx` | ✅ | W8A16 auto |
| **Hailo** | `hailo` | `yolo26n_hailo_model/` | ✅ | INT8 |
| **Huawei Ascend** | `ascend` | `yolo26n_ascend_model/` | ✅ | INT8 |
| **LiteRT** | `litert` | `yolo26n.tflite` | ✅ | INT8, W8A16, W8A32 |

### Recommandations par plateforme mobile

| Plateforme | Format recommandé | Quantification |
|------------|-------------------|----------------|
| **Android (général)** | LiteRT / ONNX | INT8 ou W8A32 |
| **Android (GPU)** | ONNX / NCNN | FP16 |
| **iOS** | CoreML | INT8 ou W8A16 |
| **Raspberry Pi** | NCNN | FP16 |
| **Google Coral** | Edge TPU | INT8 |
| **Snapdragon** | Qualcomm QNN | W8A16 |
| **Rockchip** | RKNN | INT8 |
| **Sony IMX500** | IMX500 | INT8 |
| **Intel Movidius** | OpenVINO | INT8 |

---

## 3. Options de Quantification

### Tableau des précisions

| Valeur `quantize` | Alias | Signification |
|-------------------|-------|---------------|
| `8` | `"8"`, `"int8"`, `"w8a8"` | Poids et activations INT8 |
| `16` | `"16"`, `"fp16"`, `"w16a16"` | Poids et activations FP16 |
| `32` | `"32"`, `"fp32"`, `"w32a32"` | FP32 (par défaut) |
| `"w8a16"` | — | Poids INT8, activations FP16 |
| `"w8a32"` | — | Poids INT8, activations FP32 (LiteRT dynamique) |

### Matrice de support par format

| Format | FP32 | FP16 | INT8 | W8A16 | Notes |
|--------|------|------|------|-------|-------|
| TorchScript | ✅ | ✅ GPU | ❌ | ❌ | FP16 nécessite `device=0` |
| ONNX | ✅ | ✅ | ✅ | ❌ | INT8 = quantification statique |
| OpenVINO | ✅ | ✅ | ✅ | ❌ | INT8 = NNCF post-entraînement |
| TensorRT | ✅ | ✅ | ✅ | ❌ | INT8 nécessite calibration |
| CoreML | ✅ | ✅ | ✅ | ✅ | W8A16 = poids INT8 + activations FP16 |
| TF SavedModel | ✅ | ❌ | ✅ | ❌ | INT8 = calibration TensorFlow |
| Edge TPU | ❌ | ❌ | ✅ auto | ❌ | INT8 obligatoire |
| MNN | ✅ | ✅ | ✅ | ❌ | INT8 = quantification des poids |
| NCNN | ✅ | ✅ | ❌ | ❌ | Mobile/embarqué |
| IMX500 | ❌ | ❌ | ✅ auto | ✅ | INT8 obligatoire |
| RKNN | ❌ | ✅ (selon puce) | ✅ | ❌ | RV1103/RV1106 = INT8 uniquement |
| LiteRT | ✅ | ❌ | ✅ | ✅ | W8A32 = INT8 dynamique sans calibration |
| Axelera | ❌ | ❌ | ✅ auto | ❌ | INT8 obligatoire |
| DEEPX | ❌ | ❌ | ✅ auto | ❌ | INT8 obligatoire |
| Qualcomm QNN | ❌ | ❌ | ❌ | ✅ auto | W8A16 = poids INT8 + activations 16 bits |

### Quand utiliser chaque précision

| Précision | Quand l'utiliser | Impact taille | Impact vitesse |
|-----------|------------------|---------------|----------------|
| **FP32** | Développement, validation initiale | 1x | 1x |
| **FP16** | GPU compatible (mobile GPU, serveur) | 0.5x | ~1.5-2x |
| **INT8** | Edge/Embedded, Android, iOS | 0.25x | ~2-4x |
| **W8A16** | iOS CoreML, Qualcomm | 0.3x | ~2-3x |
| **W8A32** | LiteRT Android (sans calibration) | 0.3x | ~1.5-2x |

### Code de quantification

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# FP16 — moitié de la taille, accélération GPU
model.export(format="onnx", quantize=16)

# INT8 — quart de la taille, nécessite calibration
model.export(format="onnx", quantize=8, data="coco8.yaml")

# INT8 avec fraction du jeu de données
model.export(format="onnx", quantize=8, data="coco8.yaml", fraction=0.5)

# W8A16 — CoreML, Qualcomm
model.export(format="coreml", quantize="w8a16")

# W8A32 — LiteRT dynamique (pas de calibration)
model.export(format="litert", quantize="w8a32")
```

---

## 4. Code d'Export pour Chaque Format Mobile

### ONNX (Universel)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export basique
model.export(format="onnx")

# Export optimisé mobile
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,      # Simplifier le graphe
    dynamic=False,       # Taille fixe (plus rapide)
    opset=17,            # Version opset
    quantize=8,          # INT8
    data="coco8.yaml"    # Calibration
)

# Export avec taille dynamique
model.export(format="onnx", dynamic=True)
```

```bash
# CLI
yolo export model=yolo26n.pt format=onnx simplify=True opset=17
```

### CoreML (iOS)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# FP32
model.export(format="coreml")

# INT8 pour iOS
model.export(
    format="coreml",
    quantize=8,
    data="coco8.yaml"
)

# W8A16 — poids INT8 + activations FP16
model.export(format="coreml", quantize="w8a16")
```

### LiteRT / TFLite (Android)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# INT8 statique (avec calibration)
model.export(
    format="litert",
    quantize=8,
    data="coco8.yaml"
)

# W8A16 (poids INT8 + activations INT16)
model.export(
    format="litert",
    quantize="w8a16",
    data="coco8.yaml"
)

# W8A32 — INT8 dynamique (PAS de calibration requise)
model.export(format="litert", quantize="w8a32")
```

### TensorRT (GPU NVIDIA / Jetson)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# FP16 pour GPU
model.export(
    format="engine",
    half=True,
    workspace=4.0  # Espace de travail en GB
)

# INT8 pour Jetson
model.export(
    format="engine",
    quantize=8,
    data="coco8.yaml",
    workspace=4.0
)

# Taille dynamique
model.export(format="engine", dynamic=True, half=True)
```

```bash
# CLI
yolo export model=yolo26n.pt format=engine half=True workspace=4.0
```

### NCNN (Mobile léger)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export pour mobile/embarqué
model.export(format="ncnn")

# Avec quantification
model.export(format="ncnn", quantize=16)  # FP16
```

### OpenVINO (Intel)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# FP32
model.export(format="openvino")

# INT8
model.export(
    format="openvino",
    quantize=8,
    data="coco8.yaml"
)

# FP16
model.export(format="openvino", quantize=16)
```

### Edge TPU (Google Coral)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# INT8 automatique (obligatoire pour Edge TPU)
model.export(
    format="edgetpu",
    data="coco8.yaml"
)
```

### MNN (Mobile Alibaba)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export MNN
model.export(format="mnn")

# INT8
model.export(format="mnn", quantize=8)
```

### RKNN (Rockchip)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# INT8
model.export(
    format="rknn",
    quantize=8,
    data="coco8.yaml"
)

# FP16 (selon la puce)
model.export(format="rknn", quantize=16)
```

### Qualcomm QNN

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# W8A16 automatique
model.export(
    format="qnn",
    data="coco8.yaml"
)
```

### ExecuTorch (Meta)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="executorch")
```

### TorchScript

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# FP32
model.export(format="torchscript")

# FP16 (GPU uniquement)
model.export(format="torchscript", quantize=16, device=0)

# Optimisé mobile
model.export(format="torchscript", optimize=True)
```

---

## 5. Commandes de Benchmarking

### Benchmark Python

```python
from ultralytics import YOLO

# Benchmark d'un modèle PyTorch
model = YOLO("yolo26n.pt")
results = model.benchmark(
    imgsz=640,
    half=False,
    device=0,
    data="coco.yaml"
)

# Benchmark d'un modèle ONNX
model = YOLO("yolo26n.onnx")
results = model.benchmark(imgsz=640)
```

### Benchmark en ligne de commande

```bash
# Benchmark modèle PyTorch
yolo benchmark model=yolo26n.pt imgsz=640 data=coco.yaml

# Benchmark modèle ONNX
yolo benchmark model=yolo26n.onnx imgsz=640

# Benchmark modèle TensorRT
yolo benchmark model=yolo26n.engine imgsz=640

# Benchmark avec FP16
yolo benchmark model=yolo26n.pt imgsz=640 half=True
```

### Métriques de benchmark

Le benchmark mesure :
- **Latence** (ms/image) — temps moyen d'inférence
- **Débit** (images/seconde) — nombre d'images traitées par seconde
- **Taille du modèle** (MB) — taille du fichier exporté
- **Précision** (mAP) — précision sur le jeu de validation

### Benchmark comparatif mobile

```python
from ultralytics import YOLO

# Comparer les formats pour le mobile
formats = ["onnx", "ncnn", "litert", "coreml"]
results = {}

for fmt in formats:
    try:
        model = YOLO("yolo26n.pt")
        model.export(format=fmt)
        # Mesurer la latence...
        results[fmt] = {"latence": "...", "taille": "..."}
    except Exception as e:
        print(f"Format {fmt} non disponible: {e}")
```

---

## 6. Préparation des Données

### Structure du jeu de données COCO

```
my-dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── img100.jpg
│   │   └── ...
│   └── labels/
│       ├── img100.txt
│       └── ...
└── data.yaml
```

### Format des annotations YOLO

Chaque fichier `.txt` contient une ligne par objet :

```
<class_id> <x_center> <y_center> <width> <height>
```

Exemple (2 objets) :
```
0 0.5 0.4 0.3 0.6
1 0.2 0.7 0.15 0.25
```

Les coordonnées sont normalisées entre 0 et 1.

### Fichier data.yaml

```yaml
# my-dataset.yaml
path: /path/to/my-dataset  # Chemin racine du jeu de données
train: train/images        # Chemin relatif vers les images d'entraînement
val: val/images            # Chemin relatif vers les images de validation

# Names (noms des classes)
names:
  0: person
  1: car
  2: bicycle

nc: 3  # Nombre de classes
```

### Formats de détection orientée (OBB)

```
<class_id> <x_center> <y_center> <width> <height> <angle_degrés>
```

### Formats de segmentation

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Les points du polygone sont normalisés entre 0 et 1.

### Formats de pose

```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> ... <kp17_x> <kp17_y> <kp17_v>
```

Où `v` est la visibilité : 0 (absent), 1 (présent mais caché), 2 (visible).

### Annotation avec Roboflow

```python
# Installation
# pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("my-project")
version = project.version(1)
dataset = version.download("yolov8")

# Le dataset est prêt dans le format YOLO
print(dataset.location)
```

### Outils d'annotation recommandés

| Outil | Usage | Format |
|-------|-------|--------|
| [Roboflow](https://roboflow.com/) | Annotation + annotation assistée par IA | YOLO, COCO, Pascal VOC |
| [Label Studio](https://labelstud.io/) | Annotation open-source | Multi-format |
| [CVAT](https://cvat.ai/) | Annotation vidéo/images | YOLO, COCO |
| [Labelme](https://github.com/labelmeai/labelme) | Annotation polygonale | JSON → converter |

### Conversion entre formats

```python
# Convertir COCO → YOLO
# pip install pycocotools

import json
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Indexer les images
    images = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        # Convertir bbox COCO (x, y, w, h) → YOLO (x_center, y_center, w, h)
        x, y, bw, bh = ann["bbox"]
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        label = f"{ann['category_id']} {x_center} {y_center} {norm_w} {norm_h}\n"

        txt_path = output_dir / f"{Path(img['file_name']).stem}.txt"
        with open(txt_path, "a") as f:
            f.write(label)
```

---

## 7. Commandes d'Entraînement

### Entraînement Python

```python
from ultralytics import YOLO

# Fine-tuning depuis un modèle pré-entraîné
model = YOLO("yolo26n.pt")
results = model.train(
    data="my-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device=0,
    patience=20,          # Arrêt précoce
    save=True,
    project="runs/train",
    name="yolo26n-custom"
)

# Entraînement depuis zéro
model = YOLO("yolo26n.yaml")
results = model.train(
    data="my-dataset.yaml",
    epochs=200,
    imgsz=640,
    batch=32
)
```

### Paramètres d'entraînement courants

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `epochs` | 100 | Nombre d'époques |
| `imgsz` | 640 | Taille d'entrée |
| `batch` | 16 | Taille du lot |
| `lr0` | 0.01 | Taux d'apprentissage initial |
| `lrf` | 0.01 | Taux d'apprentissage final |
| `momentum` | 0.937 | Momentum SGD |
| `weight_decay` | 0.0005 | Régularisation L2 |
| `warmup_epochs` | 3.0 | Époques de warmup |
| `patience` | 100 | Arrêt précoce |
| `mosaic` | 1.0 | Augmentation mosaic |
| `mixup` | 0.0 | Augmentation mixup |
| `copy_paste` | 0.0 | Augmentation copy-paste |
| `device` | `None` | GPU (0, 0,1) ou CPU |
| `optimizer` | `auto` | SGD, Adam, AdamW, MuSGD |
| `freeze` | `None` | Couches à geler |
| `distill_model` | `None` | Modèle enseignant (distillation) |
| `dis` | 6.0 | Poids de distillation |

### CLI d'entraînement

```bash
# Fine-tuning
yolo detect train data=my-dataset.yaml model=yolo26n.pt epochs=100 imgsz=640

# Entraînement depuis zéro
yolo detect train data=my-dataset.yaml model=yolo26n.yaml epochs=200

# Avec distillation
yolo detect train data=my-dataset.yaml model=yolo26n.pt epochs=100 distill_model=yolo26s.pt

# Multi-GPU
yolo detect train data=my-dataset.yaml model=yolo26n.pt epochs=100 device=0,1
```

### Entraînement par tâche

```bash
# Détection
yolo detect train data=coco.yaml model=yolo26n.pt epochs=100

# Segmentation
yolo segment train data=coco-seg.yaml model=yolo26n-seg.pt epochs=100

# Classification
yolo classify train data=imagenet.yaml model=yolo26n-cls.pt epochs=100

# Pose
yolo pose train data=coco-pose.yaml model=yolo26n-pose.pt epochs=100

# OBB
yolo obb train data=dota.yaml model=yolo26n-obb.pt epochs=100
```

---

## 8. Workflow de Validation

### Validation Python

```python
from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO("runs/detect/train/weights/best.pt")

# Validation complète
metrics = model.val(
    data="my-dataset.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.6,
    device=0
)

# Résultats
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Précision: {metrics.box.mp}")
print(f"Rappel: {metrics.box.mr}")
```

### Validation en ligne de commande

```bash
# Validation standard
yolo detect val model=best.pt data=my-dataset.yaml

# Validation avec seuil de confiance personnalisé
yolo detect val model=best.pt data=my-dataset.yaml conf=0.1

# Validation sur un répertoire d'images
yolo detect val model=best.pt data=my-dataset.yaml source=path/to/images/

# Exporter les résultats
yolo detect val model=best.pt data=my-dataset.yaml save_json=True
```

### Validation après export

```python
from ultralytics import YOLO

# Valider un modèle ONNX exporté
model = YOLO("yolo26n.onnx")
metrics = model.val(data="my-dataset.yaml")

# Valider un modèle TensorRT
model = YOLO("yolo26n.engine")
metrics = model.val(data="my-dataset.yaml")
```

### Métriques de validation

| Métrique | Description |
|----------|-------------|
| `mAP50` | mAP à IoU=0.50 |
| `mAP50-95` | mAP moyen de IoU=0.50 à 0.95 |
| `Précision` | True Positives / (True Positives + False Positives) |
| `Rappel` | True Positives / (True Positives + False Negatives) |
| `F1-Score` | Moyenne harmonique précision/rappel |

---

## 9. Dépannage des Problèmes d'Export

### Problèmes courants et solutions

| Problème | Cause | Solution |
|----------|-------|----------|
| **Erreur de calibration INT8** | Jeu de données manquant | Ajouter `data="coco8.yaml"` ou un jeu de données représentatif |
| **Modèle trop gros** | Pas de quantification | Utiliser `quantize=8` ou `quantize=16` |
| **Latence élevée après export** | Format non optimisé | Essayer TensorRT (GPU) ou ONNX avec simplification |
| **output0 en FP32 avec INT8** | Comportement attendu avec `end2end=True` | Normal — les indices de classe nécessitent FP32 |
| **Erreur de compatibilité opset** | Version ONNX trop ancienne | Utiliser `opset=17` ou plus |
| **Taille d'entrée dynamique échoue** | Format ne supporte pas | Vérifier la matrice de support (ONNX, TensorRT, OpenVINO) |
| **Modèle ne détecte rien après export** | Seuil de confiance trop élevé | Réduire `conf=0.1` lors de l'inférence |
| **Erreur de mémoire GPU** | Batch trop grand | Réduire `batch` ou utiliser `device=cpu` |

### Export avec end2end=True (défaut)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# End-to-end par défaut (pas de NMS, sortie (N, 300, 6))
model.export(format="onnx")

# Forcer le mode non end-to-end (NMS requis, sortie traditionnelle)
model.export(format="onnx", end2end=False)
```

### Vérifier la sortie du modèle exporté

```python
import onnxruntime as ort
import numpy as np

# Charger le modèle ONNX
session = ort.InferenceSession("yolo26n.onnx")

# Vérifier les shapes de sortie
for output in session.get_outputs():
    print(f"Output: {output.name}, Shape: {output.shape}")
# Détection end-to-end: (batch, 300, 6)
# Détection classique: (batch, 84, 8400)
```

### Optimiser la taille du modèle

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Stratégie 1 : Quantification
model.export(format="onnx", quantize=8)  # ~75% de réduction

# Stratégie 2 : Taille réduite
model.export(format="onnx", imgsz=320)  # Entrée plus petite

# Stratégie 3 : Simplification du graphe
model.export(format="onnx", simplify=True)

# Stratégie 4 : Combinaison
model.export(
    format="onnx",
    imgsz=480,
    quantize=8,
    simplify=True,
    data="coco8.yaml"
)
```

### Débogage d'export

```python
from ultralytics import YOLO

# 1. Vérifier que le modèle se charge correctement
model = YOLO("yolo26n.pt")
model.info()  # Affiche les FLOPs et la structure

# 2. Tester l'inférence avant export
results = model("test_image.jpg")
print(results[0].boxes)

# 3. Exporter et vérifier
model.export(format="onnx")

# 4. Tester le modèle exporté
model_exported = YOLO("yolo26n.onnx")
results = model_exported("test_image.jpg")
print(results[0].boxes)

# 5. Comparer les résultats
# Les résultats devraient être très similaires
```

---

## Résumé Rapide des Formats Mobiles

| Cible | Format | Commande rapide |
|-------|--------|-----------------|
| **Android général** | ONNX INT8 | `model.export(format="onnx", quantize=8, data="data.yaml")` |
| **Android LiteRT** | LiteRT W8A32 | `model.export(format="litert", quantize="w8a32")` |
| **iOS** | CoreML INT8 | `model.export(format="coreml", quantize=8, data="data.yaml")` |
| **NVIDIA Jetson** | TensorRT FP16 | `model.export(format="engine", half=True)` |
| **Google Coral** | Edge TPU INT8 | `model.export(format="edgetpu", data="data.yaml")` |
| **Snapdragon** | QNN W8A16 | `model.export(format="qnn", data="data.yaml")` |
| **Rockchip** | RKNN INT8 | `model.export(format="rknn", quantize=8, data="data.yaml")` |
| **Intel** | OpenVINO INT8 | `model.export(format="openvino", quantize=8, data="data.yaml")` |
| **Ultra-léger** | NCNN FP16 | `model.export(format="ncnn", quantize=16)` |

---

## Références

- [Documentation Export Ultralytics](https://docs.ultralytics.com/fr/modes/export)
- [Guide Fine-Tuning](https://docs.ultralytics.com/fr/guides/finetuning-guide)
- [Guide Distillation](https://docs.ultralytics.com/fr/guides/knowledge-distillation)
- [Configuration YAML](https://docs.ultralytics.com/fr/guides/model-yaml-config)
- [Recette d'entraînement YOLO26](https://docs.ultralytics.com/fr/guides/yolo26-training-recipe)
