# Référence Complète : Détection d'Objets avec YOLO26

> Guide complet pour l'intégration de la détection d'objets dans une application mobile utilisant YOLO26 — 80 classes COCO, inférence NMS-free, et déploiement multiplateforme.

---

## Table des matières

1. [Description de la Tâche et Cas d'Utilisation](#1-description-de-la-tâche-et-cas-dutilisation)
2. [Variantes de Modèles et Nommage](#2-variantes-de-modèles-et-nommage)
3. [Les 80 Classes COCO](#3-les-80-classes-coco)
4. [Inférence NMS-Free par Défaut](#4-inférence-nms-free-par-défaut)
5. [Format des Résultats](#5-format-des-résultats)
6. [Commandes d'Export pour Mobiles](#6-commandes-dexport-pour-mobiles)
7. [Exemples de Code d'Inférence](#7-exemples-de-code-dinférence)
8. [Intégration Mobile](#8-intégration-mobile)
9. [Considérations Spécifiques au Mobile](#9-considérations-spécifiques-au-mobile)
10. [Conseils de Performance](#10-conseils-de-performance)
11. [Dépannage](#11-dépannage)

---

## 1. Description de la Tâche et Cas d'Utilisation

La détection d'objets identifie et localise des objets dans une image en produisant des boîtes englobantes (bounding boxes) avec une classe et un score de confiance.

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Comptage d'objets** | Compter les personnes, véhicules, animaux | `yolo26n.pt` |
| **Sécurité / Alarme** | Détecter intrusions, objets suspects | `yolo26s.pt` |
| **Parking** | Détecter places occupées/libres | `yolo26n.pt` |
| **Réalité augmentée** | Superposer informations sur objets | `yolo26s.pt` |
| **Tri automatique** | Catégoriser objets sur un tapis roulant | `yolo26m.pt` |
| **Assistance aveugles** | Décrire environnement visuel | `yolo26s.pt` |
| **Qualité industrielle** | Détecter défauts sur chaîne de production | `yolo26m.pt` |
| **Agriculture** | Compter fruits, détecter maladies | `yolo26n.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | mAP COCO | Idéal pour |
|--------|---------|--------|-------|----------|------------|
| YOLO26n | `yolo26n.pt` | ~3M | ~8G | 40,9 | Mobile temps réel, CPU |
| YOLO26s | `yolo26s.pt` | ~11M | ~30G | 48,6 | Équilibre vitesse/précision |
| YOLO26m | `yolo26m.pt` | ~20M | ~55G | 53,1 | Précision améliorée |
| YOLO26l | `yolo26l.pt` | ~26M | ~80G | 55,0 | Haute précision GPU |
| YOLO26x | `yolo26x.pt` | ~57M | ~170G | 57,5 | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle pré-entraîné COCO
model = YOLO("yolo26n.pt")

# Vérifier les classes disponibles
print(model.names)  # {0: 'person', 1: 'bicycle', ...}
```

---

## 3. Les 80 Classes COCO

YOLO26 est pré-entraîné sur le jeu de données COCO avec 80 classes :

| ID | Classe | ID | Classe | ID | Classe | ID | Classe |
|----|--------|----|--------|----|--------|----|--------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

---

## 4. Inférence NMS-Free par Défaut

YOLO26 utilise une tête **one-to-one** qui élimine le besoin de suppression non maximale (NMS). C'est un avantage majeur pour le mobile : moins de post-traitement, plus de vitesse.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Inférence end-to-end (PAS de NMS requis)
results = model("image.jpg")
# Sortie: (N, 300, 6) — max 300 détections, format [x1, y1, x2, y2, conf, class]
```

### Comparaison avec le mode traditionnel

```python
# Mode traditionnel (NMS requis) — pour compatibilité ou précision maximale
results = model.predict("image.jpg", end2end=False)
# Sortie: (N, 84, 8400) — nécessite NMS
```

| Mode | Sortie | NMS | Vitesse | Cas d'usage |
|------|--------|-----|---------|-------------|
| **One-to-One (défaut)** | `(N, 300, 6)` | Non | Rapide | Mobile, temps réel |
| **One-to-Many** | `(N, 84, 8400)` | Oui | Plus lent | Précision maximale |

---

## 5. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")

# results est une liste de objets Results (un par image)
result = results[0]

# Boîtes englobantes
print(result.boxes)           # Bboxes object
print(result.boxes.xyxy)      # Coordonnées (x1, y1, x2, y2) en pixels — shape (N, 4)
print(result.boxes.conf)      # Scores de confiance — shape (N,)
print(result.boxes.cls)       # IDs de classe — shape (N,)

# Noms des classes
print(result.names)           # {0: 'person', 1: 'bicycle', ...}

# Nombre de détections
print(len(result.boxes))      # Nombre d'objets détectés
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")

for result in results:
    boxes = result.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = boxes.conf[i].item()
        cls_id = int(boxes.cls[i].item())
        cls_name = result.names[cls_id]
        print(f"{cls_name}: {conf:.2f} — ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
```

### Format de sortie pour intégration mobile

```python
# Convertir en format JSON pour envoi à Flutter/React Native
import json

def results_to_json(result):
    boxes = result.boxes
    detections = []
    for i in range(len(boxes)):
        detections.append({
            "bbox": boxes.xyxy[i].tolist(),      # [x1, y1, x2, y2]
            "confidence": boxes.conf[i].item(),   # 0.0 - 1.0
            "class_id": int(boxes.cls[i].item()), # 0 - 79
            "class_name": result.names[int(boxes.cls[i].item())]
        })
    return json.dumps(detections)
```

---

## 6. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8 (quantifié)
model.export(format="onnx", quantize=8, data="coco8.yaml")
```

### Export iOS (CoreML)

```python
# CoreML — utilise Apple Neural Engine
model.export(format="coreml", quantize=8)

# CoreML W8A16 (poids INT8 + activations FP16)
model.export(format="coreml", quantize="w8a16")
```

### Export Android (LiteRT)

```python
# LiteRT INT8 dynamique (pas de calibration)
model.export(format="litert", quantize="w8a32")

# LiteRT INT8 statique (avec calibration)
model.export(format="litert", quantize=8, data="coco8.yaml")
```

### Export cross-platform (NCNN)

```python
# NCNN — ultra-léger pour ARM
model.export(format="ncnn")
```

### Export ExecuTorch (Meta)

```python
# ExecuTorch — pour React Native / PyTorch Mobile
model.export(format="executorch")
```

### Tableau récapitulatif

| Cible | Format | Commande | Quantification |
|-------|--------|----------|----------------|
| iOS | CoreML | `model.export(format="coreml")` | `quantize=8` |
| Android | LiteRT | `model.export(format="litert")` | `quantize="w8a32"` |
| Cross-platform | NCNN | `model.export(format="ncnn")` | — |
| Universel | ONNX | `model.export(format="onnx")` | `quantize=8` |
| React Native | ExecuTorch | `model.export(format="executorch")` | — |

---

## 7. Exemples de Code d'Inférence

### Inférence simple

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg", conf=0.5)  # Seuil de confiance 50%

for result in results:
    result.show()  # Afficher avec annotations
    result.save(filename="result.jpg")  # Sauvegarder
```

### Inférence sur flux vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Traitement vidéo en temps réel
results = model("video.mp4", stream=True, conf=0.5, iou=0.6)

for result in results:
    boxes = result.boxes
    print(f"Frame: {len(boxes)} objets détectés")
```

### Inférence par lots (batch)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Traiter plusieurs images en une fois
results = model(["img1.jpg", "img2.jpg", "img3.jpg"], conf=0.5)

for i, result in enumerate(results):
    print(f"Image {i}: {len(result.boxes)} objets")
```

### Inférence avec taille personnalisée

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Réduire la taille pour plus de vitesse (mobile)
results = model("image.jpg", imgsz=320, conf=0.5)

# Taille personnalisée
results = model("image.jpg", imgsz=(480, 640), conf=0.5)
```

---

## 8. Intégration Mobile

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
│  YOLO26 Model (yolo26n.mlpackage / .tflite / .bin) │
│    ↓                                                 │
│  Results (boxes, confidence, class_id)              │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
// Dart side
import 'package:flutter/services.dart';

class YOLODetectionService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_detect');

  Future<List<Map<String, dynamic>>> detectObjects(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('detectObjects', {
      'image': imageBytes,
      'confidence': 0.5,
    });
    return List<Map<String, dynamic>>.from(result);
  }
}
```

### React Native — Native Module

```javascript
// JS side
import { NativeModules } from 'react-native';

const { YOLODetection } = NativeModules;

export async function detectObjects(imagePath, confidence = 0.5) {
  const results = await YOLODetection.detectObjects(imagePath, confidence);
  return results; // [{bbox: [x1,y1,x2,y2], confidence: 0.95, class_id: 0, class_name: "person"}]
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLODetector {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU + GPU + Neural Engine
        let coreMLModel = try! YOLO26n(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func detect(pixelBuffer: CVPixelBuffer, completion: @escaping ([Detection]) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            let detections = results.map { obs in
                Detection(
                    bbox: obs.boundingBox,
                    confidence: obs.confidence,
                    class_id: obs.labels.first?.identifier ?? 0
                )
            }
            completion(detections)
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

class YOLODetector(private val interpreter: Interpreter) {
    
    fun detect(imageData: ByteBuffer): List<Detection> {
        val output = Array(1) { Array(300) { FloatArray(6) } }
        interpreter.run(imageData, output)
        
        return parseDetections(output[0])
    }
    
    private fun parseDetections(output: Array<FloatArray>): List<Detection> {
        val detections = mutableListOf<Detection>()
        for (i in output.indices) {
            val x1 = output[i][0]
            val y1 = output[i][1]
            val x2 = output[i][2]
            val y2 = output[i][3]
            val conf = output[i][4]
            val cls = output[i][5].toInt()
            
            if (conf > 0.5f) {
                detections.add(Detection(x1, y1, x2, y2, conf, cls))
            }
        }
        return detections
    }
}
```

---

## 9. Considérations Spécifiques au Mobile

### Gestion de la mémoire

| Conséquence | Solution |
|-------------|----------|
| Modèle trop grand | Utiliser `yolo26n.pt` + quantification INT8 |
| Fuites mémoire | Libérer les résultats après traitement |
| Images trop grandes | Redimensionner avant inférence (640→320) |
| Batch processing | Traiter image par image sur mobile |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=320` au lieu de 640 |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n.pt` |
| GPU/NPU delegate | +5x vitesse | CoreML ANE / LiteRT GPU |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Seuils de confiance

```python
# Sur mobile, ajuster le seuil selon le cas d'usage
results = model("image.jpg", conf=0.25)  # Plus de détections (rappel élevé)
results = model("image.jpg", conf=0.50)  # Équilibré
results = model("image.jpg", conf=0.75)  # Moins de faux positifs (précision élevée)
```

---

## 10. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n | ~50 | ~10 | 20-100 |
| YOLO26s | ~120 | ~25 | 8-40 |
| YOLO26m | ~250 | ~50 | 4-20 |
| YOLO26l | ~400 | ~80 | 2-12 |
| YOLO26x | ~800 | ~150 | 1-6 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 320 si possible
- [ ] Activer le delegate GPU/NPU
- [ ] Utiliser `simplify=True` pour ONNX
- [ ] Désactiver NMS (déjà le cas par défaut)
- [ ] Traiter les frames en arrière-plan
- [ ] Limiter le nombre de détections (max_det)

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=320,        # Taille réduite
    conf=0.5,         # Seuil de confiance
    max_det=50,       # Limiter les détections
    verbose=False     # Pas de logs
)
```

---

## 11. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Aucune détection | Seuil de confiance trop élevé | Réduire `conf=0.25` |
| Trop de faux positifs | Seuil trop bas | Augmenter `conf=0.75` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + `imgsz=320` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Erreur mémoire | Batch trop grand | Traiter image par image |
| Résultats différents après export | Mode end2end | Vérifier que `end2end=True` (défaut) |
| Classes incorrectes | Mauvais modèle | Vérifier `model.names` |

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
