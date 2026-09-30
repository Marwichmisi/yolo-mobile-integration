# Référence Complète : Segmentation d'Instances avec YOLO26-seg

> Guide complet pour l'intégration de la segmentation d'instances dans une application mobile utilisant YOLO26-seg — masques pixel par pixel, boîtes englobantes, et déploiement multiplateforme.

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

La segmentation d'instances identie chaque objet individuel dans une image et génère un masque pixel par pixel pour chaque instance, en plus des boîtes englobantes.

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Floutage d'arrière-plan** | Isoler le sujet, flouter le reste | `yolo26n-seg.pt` |
| **Réalité augmentée** | Superposer effets sur objets spécifiques | `yolo26s-seg.pt` |
| **Mesure d'objets** | Calculer dimensions réelles à partir du masque | `yolo26m-seg.pt` |
| **Retouche photo** | Sélectionner et modifier objets | `yolo26s-seg.pt` |
| **Comptage de personnes** | Compter individus dans une foule | `yolo26n-seg.pt` |
| **Navigation assistée** | Délimiter zones sûres/obstacles | `yolo26m-seg.pt` |
| **E-commerce** | Essai virtuel, recherche par image | `yolo26s-seg.pt` |
| **Agriculture de précision** | Détecter zones de culture/maladies | `yolo26m-seg.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | mAP boîte | mAP masque | Idéal pour |
|--------|---------|--------|-------|-----------|------------|------------|
| YOLO26n-seg | `yolo26n-seg.pt` | ~3M | ~8G | 40,9 | 33,1 | Mobile temps réel |
| YOLO26s-seg | `yolo26s-seg.pt` | ~11M | ~30G | 48,6 | 38,2 | Équilibre vitesse/précision |
| YOLO26m-seg | `yolo26m-seg.pt` | ~20M | ~55G | 53,1 | 41,5 | Précision améliorée |
| YOLO26l-seg | `yolo26l-seg.pt` | ~26M | ~80G | 55,0 | 43,0 | Haute précision |
| YOLO26x-seg | `yolo26x-seg.pt` | ~57M | ~170G | 57,5 | 45,2 | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle de segmentation pré-entraîné
model = YOLO("yolo26n-seg.pt")

# Vérifier les classes
print(model.names)  # {0: 'person', 1: 'bicycle', ...}
```

---

## 3. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
results = model("image.jpg")

result = results[0]

# Boîtes englobantes
print(result.boxes)           # Bboxes object
print(result.boxes.xyxy)      # Coordonnées (x1, y1, x2, y2) — shape (N, 4)
print(result.boxes.conf)      # Scores de confiance — shape (N,)
print(result.boxes.cls)       # IDs de classe — shape (N,)

# Masques de segmentation
print(result.masks)           # Masks object
print(result.masks.data)      # Masques tensor — shape (N, H, W) en coordonnées image
print(result.masks.xy)        # Polygones de contour — liste de (M, 2) arrays
print(result.masks.xyn)       # Polygones normalisés — liste de (M, 2) arrays
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
results = model("image.jpg")

for result in results:
    boxes = result.boxes
    masks = result.masks
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = boxes.conf[i].item()
        cls_id = int(boxes.cls[i].item())
        cls_name = result.names[cls_id]
        
        # Masque binaire (H, W)
        mask = masks.data[i].cpu().numpy()
        
        # Polygone de contour
        polygon = masks.xy[i]  # (M, 2) array
        
        print(f"{cls_name}: {conf:.2f} — masque {mask.shape}")
```

### Format de sortie pour intégration mobile

```python
import json
import numpy as np

def segmentation_to_json(result):
    boxes = result.boxes
    masks = result.masks
    segments = []
    
    for i in range(len(boxes)):
        # Convertir le masque en polygon pour réduire la taille
        polygon = masks.xy[i].tolist() if masks else []
        
        segments.append({
            "bbox": boxes.xyxy[i].tolist(),
            "confidence": boxes.conf[i].item(),
            "class_id": int(boxes.cls[i].item()),
            "class_name": result.names[int(boxes.cls[i].item())],
            "polygon": polygon  # Contour du masque
        })
    
    return json.dumps(segments)
```

---

## 4. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8
model.export(format="onnx", quantize=8, data="coco8.yaml")
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
model.export(format="litert", quantize=8, data="coco8.yaml")
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

model = YOLO("yolo26n-seg.pt")
results = model("image.jpg", conf=0.5)

for result in results:
    result.show()  # Afficher avec masques
    result.save(filename="result_seg.jpg")
```

### Extraire les masques

```python
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n-seg.pt")
results = model("image.jpg")

result = results[0]
masks = result.masks

if masks is not None:
    # Masques binaires (N, H, W)
    mask_tensor = masks.data.cpu().numpy()
    
    # Pour chaque instance
    for i in range(len(result.boxes)):
        mask = mask_tensor[i]  # (H, W) booléen
        cls_id = int(result.boxes.cls[i].item())
        print(f"Instance {i}: classe {result.names[cls_id]}, masque {mask.shape}")
```

### Appliquer un masque sur l'image

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo26n-seg.pt")
results = model("image.jpg")

result = results[0]
image = result.orig_img  # Image originale (H, W, 3)

if result.masks is not None:
    # Combiner tous les masques
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in result.masks.data:
        combined_mask |= mask.cpu().numpy().astype(np.uint8)
    
    # Appliquer le masque
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    cv2.imwrite("masked.jpg", masked_image)
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

results = model("video.mp4", stream=True, conf=0.5)

for i, result in enumerate(results):
    if result.masks is not None:
        print(f"Frame {i}: {len(result.boxes)} instances segmentées")
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
│  YOLO26-seg Model                                   │
│    ↓                                                 │
│  Results (boxes, masks, confidence, class_id)       │
│    ↓                                                 │
│  Mask Rendering / Overlay                           │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
import 'package:flutter/services.dart';

class YOLOSegmentationService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_seg');

  Future<List<Map<String, dynamic>>> segmentObjects(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('segmentObjects', {
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

const { YOLOSegmentation } = NativeModules;

export async function segmentObjects(imagePath, confidence = 0.5) {
  const results = await YOLOSegmentation.segmentObjects(imagePath, confidence);
  return results; // [{bbox, confidence, class_id, class_name, polygon}]
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLOSegmentation {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try! YOLO26n_seg(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func segment(pixelBuffer: CVPixelBuffer, completion: @escaping ([SegmentationResult]) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNPixelBufferObservation] else { return }
            // Traiter les masques
            completion(self.parseSegmentations(results))
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

class YOLOSegmentation(private val interpreter: Interpreter) {
    
    fun segment(imageData: ByteBuffer): List<SegmentationResult> {
        val output = Array(1) { Array(300) { FloatArray(6) } }
        interpreter.run(imageData, output)
        return parseResults(output[0])
    }
    
    private fun parseResults(output: Array<FloatArray>): List<SegmentationResult> {
        val results = mutableListOf<SegmentationResult>()
        for (i in output.indices) {
            val conf = output[i][4]
            if (conf > 0.5f) {
                results.add(SegmentationResult(
                    x1 = output[i][0],
                    y1 = output[i][1],
                    x2 = output[i][2],
                    y2 = output[i][3],
                    confidence = conf,
                    classId = output[i][5].toInt()
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
| Masques trop volumineux | Exporter les polygones au lieu des masques complets |
| Trop d'instances | Limiter `max_det=20` |
| Images haute résolution | Redimensionner avant inférence |
| Fuites mémoire | Libérer les masques après rendu |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=320` |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n-seg.pt` |
| Polygones au lieu de masques | -90% données | Exporter `masks.xy` |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Rendu des masques sur mobile

```dart
// Flutter — CustomPainter pour masques
class SegmentationPainter extends CustomPainter {
  final List<Map<String, dynamic>> segmentations;
  
  @override
  void paint(Canvas canvas, Size size) {
    for (final seg in segmentations) {
      final paint = Paint()
        ..color = Colors.green.withOpacity(0.3)
        ..style = PaintingStyle.fill;
      
      final path = Path();
      final polygon = seg['polygon'] as List;
      if (polygon.isNotEmpty) {
        path.moveTo(polygon[0][0], polygon[0][1]);
        for (int i = 1; i < polygon.length; i++) {
          path.lineTo(polygon[i][0], polygon[i][1]);
        }
        path.close();
        canvas.drawPath(path, paint);
      }
    }
  }
}
```

---

## 8. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n-seg | ~60 | ~12 | 16-80 |
| YOLO26s-seg | ~140 | ~30 | 7-33 |
| YOLO26m-seg | ~280 | ~55 | 4-18 |
| YOLO26l-seg | ~450 | ~85 | 2-12 |
| YOLO26x-seg | ~900 | ~160 | 1-7 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n-seg.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 320 si possible
- [ ] Exporter les polygones au lieu des masques complets
- [ ] Activer le delegate GPU/NPU
- [ ] Limiter `max_det=20`
- [ ] Utiliser `simplify=True` pour ONNX

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=320,
    conf=0.5,
    max_det=20,       # Limiter les instances
    verbose=False
)

# Exporter les polygones (plus léger que les masques)
for result in results:
    if result.masks:
        polygons = result.masks.xy  # Liste de (M, 2) arrays
```

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Masques de mauvaise qualité | Résolution trop basse | Augmenter `imgsz` à 640 |
| Trop d'instances | `max_det` trop élevé | Réduire `max_det=20` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-seg.pt` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Masques non affichés | `masks` est None | Vérifier que le modèle est bien un modèle `-seg` |
| Polygones incorrects | Erreur de conversion | Vérifier les coordonnées normalisées |
| Mémoire insuffisante | Trop d'instances | Réduire `max_det` et `imgsz` |

---

## Références

- [Documentation YOLO26-seg Ultralytics](https://docs.ultralytics.com/fr/tasks/segment/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
