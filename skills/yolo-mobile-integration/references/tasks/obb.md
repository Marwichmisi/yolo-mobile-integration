# Référence Complète : Détection Orientée (OBB) avec YOLO26-obb

> Guide complet pour l'intégration de la détection de boîtes orientées dans une application mobile utilisant YOLO26-obb — boîtes rotatives pour l'imagerie aérienne, documentaire et industrielle.

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

La détection de boîtes orientées (Oriented Bounding Boxes — OBB) détecte des objets avec des boîtes rotatives, contrairement aux boîtes alignées sur les axes (AABB). C'est essentiel pour les objets qui ne sont pas alignés avec l'axe horizontal/vertical.

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Imagerie aérienne** | Détecter véhicules, bâtiments sur photos satellite | `yolo26n-obb.pt` |
| **Documents numérisés** | Détecter tableaux, formulaires, signatures | `yolo26s-obb.pt` |
| **Inspection industrielle** | Détecter défauts sur pièces rotatives | `yolo26m-obb.pt` |
| **Navigation maritime** | Détecter navires, bouées | `yolo26s-obb.pt` |
| **Agriculture de précision** | Détecter rangées de culture | `yolo26m-obb.pt` |
| **Logistique** | Détecter colis sur tapis roulant | `yolo26n-obb.pt` |
| **Cartographie** | Détecter routes, ponts, infrastructures | `yolo26m-obb.pt` |
| **Assurance** | Évaluer dégâts sur photos de sinistres | `yolo26s-obb.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | mAP DOTA | Idéal pour |
|--------|---------|--------|-------|----------|------------|
| YOLO26n-obb | `yolo26n-obb.pt` | ~3M | ~8G | 42,1 | Mobile temps réel |
| YOLO26s-obb | `yolo26s-obb.pt` | ~11M | ~30G | 49,8 | Équilibre vitesse/précision |
| YOLO26m-obb | `yolo26m-obb.pt` | ~20M | ~55G | 54,2 | Précision améliorée |
| YOLO26l-obb | `yolo26l-obb.pt` | ~26M | ~80G | 56,5 | Haute précision |
| YOLO26x-obb | `yolo26x-obb.pt` | ~57M | ~170G | 58,8 | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle OBB pré-entraîné
model = YOLO("yolo26n-obb.pt")

# Vérifier les classes
print(model.names)  # {0: 'plane', 1: 'ship', ...}
```

---

## 3. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model("image.jpg")

result = results[0]

# Boîtes orientées
print(result.obb)             # OBB object
print(result.obb.xyxyxyxy)    # Coordonnées des 4 coins — shape (N, 4, 2)
print(result.obb.xyxyr)       # (x_center, y_center, width, height, angle) — shape (N, 5)
print(result.obb.conf)        # Scores de confiance — shape (N,)
print(result.obb.cls)         # IDs de classe — shape (N,)

# Noms des classes
print(result.names)           # {0: 'plane', 1: 'ship', ...}
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model("image.jpg")

for result in results:
    obb = result.obb
    
    for i in range(len(obb)):
        # Coordonnées des 4 coins (x, y)
        corners = obb.xyxyxyxy[i].cpu().numpy()  # (4, 2)
        
        # Format centre + angle
        x, y, w, h, angle = obb.xyxyr[i].cpu().numpy()
        
        conf = obb.conf[i].item()
        cls_id = int(obb.cls[i].item())
        cls_name = result.names[cls_id]
        
        print(f"{cls_name}: {conf:.2f}")
        print(f"  Centre: ({x:.1f}, {y:.1f}), Taille: {w:.1f}x{h:.1f}, Angle: {angle:.1f}°")
        print(f"  Corners: {corners}")
```

### Format de sortie pour intégration mobile

```python
import json

def obb_to_json(result):
    obb = result.obb
    detections = []
    
    for i in range(len(obb)):
        corners = obb.xyxyxyxy[i].cpu().numpy()
        x, y, w, h, angle = obb.xyxyr[i].cpu().numpy()
        
        detections.append({
            "corners": corners.tolist(),           # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            "center": [float(x), float(y)],
            "size": [float(w), float(h)],
            "angle": float(angle),
            "confidence": obb.conf[i].item(),
            "class_id": int(obb.cls[i].item()),
            "class_name": result.names[int(obb.cls[i].item())]
        })
    
    return json.dumps(detections)
```

---

## 4. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8
model.export(format="onnx", quantize=8, data="dota8.yaml")
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
model.export(format="litert", quantize=8, data="dota8.yaml")
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

model = YOLO("yolo26n-obb.pt")
results = model("image.jpg", conf=0.5)

for result in results:
    result.show()  # Afficher avec boîtes orientées
    result.save(filename="result_obb.jpg")
```

### Extraire les boîtes orientées

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model("image.jpg")

result = results[0]
obb = result.obb

if obb is not None:
    for i in range(len(obb)):
        # Coordonnées des 4 corners
        corners = obb.xyxyxyxy[i].cpu().numpy()  # (4, 2)
        
        # Format centre + angle
        x, y, w, h, angle = obb.xyxyr[i].cpu().numpy()
        
        cls_id = int(obb.cls[i].item())
        print(f"Classe {result.names[cls_id]}: angle={angle:.1f}°")
```

### Dessiner les boîtes orientées

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model("image.jpg")

result = results[0]
image = result.orig_img.copy()

if result.obb is not None:
    for i in range(len(result.obb)):
        corners = result.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
        
        # Dessiner le polygone
        cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Dessiner le centre
        x, y, w, h, angle = result.obb.xyxyr[i].cpu().numpy()
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

cv2.imwrite("result_obb.jpg", image)
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")

results = model("video.mp4", stream=True, conf=0.5)

for i, result in enumerate(results):
    if result.obb is not None:
        print(f"Frame {i}: {len(result.obb)} objets orientés détectés")
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
│  YOLO26-obb Model                                   │
│    ↓                                                 │
│  Results (rotated boxes, confidence, class_id)      │
│    ↓                                                 │
│  Rotated Box Rendering / Overlay                    │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
import 'package:flutter/services.dart';

class YOLOOBBService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_obb');

  Future<List<Map<String, dynamic>>> detectOBB(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('detectOBB', {
      'image': imageBytes,
      'confidence': 0.5,
    });
    return List<Map<String, dynamic>>.from(result);
  }
}
```

### React Native — Native Module

```javascript
import { NativeModules } from 'react-native';

const { YOLOOBB } = NativeModules;

export async function detectOBB(imagePath, confidence = 0.5) {
  const result = await YOLOOBB.detectOBB(imagePath, confidence);
  return result; // [{corners, center, size, angle, confidence, class_id, class_name}]
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLOOBBDetector {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try! YOLO26n_obb(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func detect(pixelBuffer: CVPixelBuffer, completion: @escaping ([OBBResult]) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            let detections = results.map { obs in
                OBBResult(
                    boundingBox: obs.boundingBox,
                    confidence: obs.confidence,
                    classId: obs.labels.first?.identifier ?? 0
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

class YOLOOBBDetector(private val interpreter: Interpreter) {
    
    fun detect(imageData: ByteBuffer): List<OBBResult> {
        val output = Array(1) { Array(300) { FloatArray(10) } }  // 4 bbox + angle + conf + class
        interpreter.run(imageData, output)
        return parseResults(output[0])
    }
    
    private fun parseResults(output: Array<FloatArray>): List<OBBResult> {
        val results = mutableListOf<OBBResult>()
        for (i in output.indices) {
            val conf = output[i][8]
            if (conf > 0.5f) {
                results.add(OBBResult(
                    x = output[i][0],
                    y = output[i][1],
                    w = output[i][2],
                    h = output[i][3],
                    angle = output[i][4],
                    confidence = conf,
                    classId = output[i][9].toInt()
                ))
            }
        }
        return results
    }
}
```

---

## 7. Considérations Spécifiques au Mobile

### Gestion de la mémoire

| Conséquence | Solution |
|-------------|----------|
| Trop d'objets | Limiter `max_det=50` |
| Images haute résolution | Redimensionner avant inférence |
| Fuites mémoire | Libérer les résultats après rendu |
| Boîtes trop précises | Simplifier les polygones |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=320` |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n-obb.pt` |
| GPU/NPU delegate | +5x vitesse | CoreML ANE / LiteRT GPU |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Rendu des boîtes orientées sur mobile

```dart
// Flutter — CustomPainter pour boîtes orientées
class OBBPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..color = Colors.green;
    
    for (final det in detections) {
      final corners = det['corners'] as List;
      final path = Path();
      
      path.moveTo(corners[0][0], corners[0][1]);
      for (int i = 1; i < corners.length; i++) {
        path.lineTo(corners[i][0], corners[i][1]);
      }
      path.close();
      
      canvas.drawPath(path, paint);
      
      // Dessiner l'angle
      final center = det['center'] as List;
      final angle = det['angle'] as double;
      // ... dessiner l'indicateur d'angle
    }
  }
}
```

---

## 8. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n-obb | ~52 | ~10 | 19-100 |
| YOLO26s-obb | ~125 | ~26 | 8-40 |
| YOLO26m-obb | ~255 | ~50 | 4-20 |
| YOLO26l-obb | ~410 | ~80 | 2-12 |
| YOLO26x-obb | ~820 | ~150 | 1-7 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n-obb.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 320 si possible
- [ ] Activer le delegate GPU/NPU
- [ ] Limiter `max_det=50`
- [ ] Utiliser `simplify=True` pour ONNX

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=320,
    conf=0.5,
    max_det=50,       # Limiter les détections
    verbose=False
)
```

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Boîtes mal orientées | Angle incorrect | Vérifier le format d'annotation |
| Aucune détection | Seuil trop élevé | Réduire `conf=0.25` |
| Trop de faux positifs | Seuil trop bas | Augmenter `conf=0.75` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-obb.pt` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Erreur mémoire | Batch trop grand | Traiter image par image |
| Angle incorrect | Format de données | Vérifier le format d'annotation OBB |
| Résultats différents après export | Mode end2end | Vérifier les paramètres d'export |

---

## Références

- [Documentation YOLO26-obb Ultralytics](https://docs.ultralytics.com/fr/tasks/obb/)
- [Dataset DOTA](https://captain-whu.github.io/DOTA/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
