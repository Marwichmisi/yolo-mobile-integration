# Référence Complète : Classification d'Images avec YOLO26-cls

> Guide complet pour l'intégration de la classification d'images dans une application mobile utilisant YOLO26-cls — classification mono-label, probabilités de classe, et déploiement multiplateforme.

---

## Table des matières

1. [Description de la Tâche et Cas d'Utilisation](#1-description-de-la-tâche-et-cas-dutilisation)
2. [Variantes de Modèles et Nommage](#2-variantes-de-modèles-et-nommage)
3. [Format des Résultats](#3-format-des-résultats)
4. [Commandes d'Export pour Mobiles](#4-commandes-dexport-pour-mobiles)
5. [Exemples de Code d'Inférence](#5-exemples-de-code-dinférence)
6. [Intégration Mobile](#6-intégration-mobile)
7. [Considérations Spécifiques au Mobile](#7-considérations-spécifiques-au-mobile)
8. [Conseils de Performance](#8-conseils-de-performance)
9. [Dépannage](#9-dépannage)

---

## 1. Description de la Tâche et Cas d'Utilisation

La classification d'images attribue une étiquette à une image entière parmi un ensemble de classes prédéfinies. YOLO26-cls est optimisé pour la classification mono-label (une seule classe par image).

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Identification d'espèces** | Reconnaître plantes, animaux, oiseaux | `yolo26s-cls.pt` |
| **Tri de produits** | Catégoriser produits sur photo | `yolo26n-cls.pt` |
| **Contrôle qualité** | Détecter produits défectueux | `yolo26m-cls.pt` |
| **Reconnaissance alimentaire** | Identifier plats, ingrédients | `yolo26s-cls.pt` |
| **Diagnostic médical** | Classifier images médicales | `yolo26m-cls.pt` |
| **Recherche visuelle** | Trouver produits similaires | `yolo26s-cls.pt` |
| **Modération de contenu** | Filtrer contenu inapproprié | `yolo26n-cls.pt` |
| **Inventaire** | Compter et catégoriser objets | `yolo26m-cls.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | Top-1 ImageNet | Idéal pour |
|--------|---------|--------|-------|----------------|------------|
| YOLO26n-cls | `yolo26n-cls.pt` | ~3M | ~8G | 70,1% | Mobile temps réel |
| YOLO26s-cls | `yolo26s-cls.pt` | ~11M | ~30G | 77,2% | Équilibre vitesse/précision |
| YOLO26m-cls | `yolo26m-cls.pt` | ~20M | ~55G | 81,5% | Précision améliorée |
| YOLO26l-cls | `yolo26l-cls.pt` | ~26M | ~80G | 83,0% | Haute précision |
| YOLO26x-cls | `yolo26x-cls.pt` | ~57M | ~170G | 84,2% | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle de classification pré-entraîné
model = YOLO("yolo26n-cls.pt")

# Vérifier les classes
print(model.names)  # {0: 'tench', 1: 'goldfish', ...}
```

---

## 3. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("image.jpg")

result = results[0]

# Probabilités de classification
print(result.probs)           # Probs object
print(result.probs.top1)      # ID de la classe la plus probable — int
print(result.probs.top1conf)  # Confiance de la classe top-1 — float
print(result.probs.top5)      # IDs des 5 classes les plus probables — list
print(result.probs.top5conf)  # Confiances des 5 classes — list

# Noms des classes
print(result.names)           # {0: 'tench', 1: 'goldfish', ...}
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("image.jpg")

for result in results:
    # Classe la plus probable
    top1_id = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    top1_name = result.names[top1_id]
    
    print(f"Classe: {top1_name} ({top1_conf:.2%})")
    
    # Top 5 classes
    print("\nTop 5:")
    for i in range(5):
        cls_id = result.probs.top5[i]
        conf = result.probs.top5conf[i].item()
        name = result.names[cls_id]
        print(f"  {i+1}. {name}: {conf:.2%}")
```

### Format de sortie pour intégration mobile

```python
import json

def classification_to_json(result):
    return json.dumps({
        "top1": {
            "class_id": result.probs.top1,
            "class_name": result.names[result.probs.top1],
            "confidence": result.probs.top1conf.item()
        },
        "top5": [
            {
                "class_id": result.probs.top5[i],
                "class_name": result.names[result.probs.top5[i]],
                "confidence": result.probs.top5conf[i].item()
            }
            for i in range(5)
        ]
    })
```

---

## 4. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8
model.export(format="onnx", quantize=8, data="imagenet.yaml")
```

### Export iOS (CoreML)

```python
# CoreML — utilise Apple Neural Engine
model.export(format="coreml", quantize=8)

# CoreML W8A16
model.export(format="coreml", quantize="w8a16")
```

### Export Android (LiteRT)

```python
# LiteRT INT8 dynamique
model.export(format="litert", quantize="w8a32")

# LiteRT INT8 statique
model.export(format="litert", quantize=8, data="imagenet.yaml")
```

### Export cross-platform (NCNN)

```python
# NCNN — ultra-léger pour ARM
model.export(format="ncnn")
```

### Tableau récapitulatif

| Cible | Format | Commande | Quantification |
|-------|--------|----------|----------------|
| iOS | CoreML | `model.export(format="coreml")` | `quantize=8` |
| Android | LiteRT | `model.export(format="litert")` | `quantize="w8a32"` |
| Cross-platform | NCNN | `model.export(format="ncnn")` | — |
| Universel | ONNX | `model.export(format="onnx")` | `quantize=8` |

---

## 5. Exemples de Code d'Inférence

### Inférence simple

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("image.jpg")

for result in results:
    top1_name = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    print(f"Classification: {top1_name} ({top1_conf:.2%})")
```

### Top 5 classes

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("image.jpg")

for result in results:
    print("Top 5 classes:")
    for i in range(5):
        cls_id = result.probs.top5[i]
        conf = result.probs.top5conf[i].item()
        name = result.names[cls_id]
        print(f"  {i+1}. {name}: {conf:.2%}")
```

### Inférence par lots

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

# Traiter plusieurs images
results = model(["img1.jpg", "img2.jpg", "img3.jpg"])

for i, result in enumerate(results):
    top1_name = result.names[result.probs.top1]
    print(f"Image {i}: {top1_name}")
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

results = model("video.mp4", stream=True)

for i, result in enumerate(results):
    top1_name = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    print(f"Frame {i}: {top1_name} ({top1_conf:.2%})")
```

---

## 6. Intégration Mobile

### Architecture générale

```
┌─────────────────────────────────────────────────────┐
│                Application Mobile                    │
├─────────────────────────────────────────────────────┤
│  UI Layer (Flutter / React Native / Native)         │
│    ↓                                                 │
│  Platform Channel / Native Module                   │
│    ↓                                                 │
│  Inference Engine (CoreML / LiteRT / NCNN)          │
│    ↓                                                 │
│  YOLO26-cls Model                                   │
│    ↓                                                 │
│  Results (class_id, class_name, confidence)         │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
import 'package:flutter/services.dart';

class YOLOClassificationService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_cls');

  Future<Map<String, dynamic>> classifyImage(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('classifyImage', {
      'image': imageBytes,
    });
    return Map<String, dynamic>.from(result);
  }
}
```

### React Native — Native Module

```javascript
import { NativeModules } from 'react-native';

const { YOLOClassification } = NativeModules;

export async function classifyImage(imagePath) {
  const result = await YOLOClassification.classifyImage(imagePath);
  return result; // {top1: {class_id, class_name, confidence}, top5: [...]}
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLOClassifier {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try! YOLO26n_cls(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func classify(pixelBuffer: CVPixelBuffer, completion: @escaping (ClassificationResult) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNClassificationObservation] else { return }
            let top1 = results.first!
            let result = ClassificationResult(
                classId: top1.identifier,
                confidence: top1.confidence
            )
            completion(result)
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
}
```

### Android — Kotlin (LiteRT)

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class YOLOClassifier(private val interpreter: Interpreter) {
    
    fun classify(imageData: ByteBuffer): ClassificationResult {
        val output = Array(1) { FloatArray(1000) }  // 1000 classes ImageNet
        interpreter.run(imageData, output)
        
        val probabilities = output[0]
        val top1Idx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        
        return ClassificationResult(
            classId = top1Idx,
            confidence = probabilities[top1Idx]
        )
    }
}
```

---

## 7. Considérations Spécifiques au Mobile

### Gestion de la mémoire

| Conséquence | Solution |
|-------------|----------|
| Modèle trop grand | Utiliser `yolo26n-cls.pt` + quantification INT8 |
| Images trop grandes | Redimensionner avant inférence (224x224) |
| Fuites mémoire | Libérer les résultats après traitement |
| Trop de classes | Limiter aux classes pertinentes |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=224` |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n-cls.pt` |
| GPU/NPU delegate | +5x vitesse | CoreML ANE / LiteRT GPU |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Seuils de confiance

```python
# Sur mobile, ajuster le seuil selon le cas d'usage
results = model("image.jpg", conf=0.5)  # Seuil standard
results = model("image.jpg", conf=0.7)  # Plus strict
results = model("image.jpg", conf=0.3)  # Plus permissif
```

---

## 8. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n-cls | ~30 | ~8 | 33-125 |
| YOLO26s-cls | ~80 | ~20 | 12-50 |
| YOLO26m-cls | ~150 | ~35 | 7-28 |
| YOLO26l-cls | ~250 | ~55 | 4-18 |
| YOLO26x-cls | ~500 | ~100 | 2-10 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n-cls.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 224 si possible
- [ ] Activer le delegate GPU/NPU
- [ ] Utiliser `simplify=True` pour ONNX
- [ ] Traiter les frames en arrière-plan
- [ ] Limiter le nombre de classes

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=224,        # Taille standard ImageNet
    verbose=False     # Pas de logs
)
```

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Classification incorrecte | Image de mauvaise qualité | Améliorer l'éclairage, résolution |
| Confiance faible | Image hors distribution | Utiliser un modèle entraîné sur votre domaine |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-cls.pt` + `imgsz=224` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Erreur mémoire | Batch trop grand | Traiter image par image |
| Classes incorrectes | Mauvais modèle | Vérifier `model.names` |
| Résultats différents après export | Mode end2end | Vérifier les paramètres d'export |

---

## Références

- [Documentation YOLO26-cls Ultralytics](https://docs.ultralytics.com/fr/tasks/classify/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
