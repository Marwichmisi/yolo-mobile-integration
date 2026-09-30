# Référence Complète : Segmentation Sémantique avec YOLO26-sem

> Guide complet pour l'intégration de la segmentation sémantique dans une application mobile utilisant YOLO26-sem — prédictions de classe par pixel, étiquetage dense, et déploiement multiplateforme.

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

La segmentation sémantique attribue une classe à chaque pixel d'une image, sans distinguer les instances d'une même classe. Contrairement à la segmentation d'instances, tous les objets d'une même classe sont traités comme une seule région.

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Conduite autonome** | Détecter route, trottoir, piétons | `yolo26s-sem.pt` |
| **Réalité augmentée** | Comprendre l'environnement 3D | `yolo26n-sem.pt` |
| **Aménagement urbain** | Classer zones vertes, bâtiments | `yolo26m-sem.pt` |
| **Agriculture** | Identifier cultures, sols, mauvaises herbes | `yolo26m-sem.pt` |
| **Météo** | Analyser couverture nuageuse | `yolo26s-sem.pt` |
| **Médecine** | Segmenter organes, tumeurs | `yolo26l-sem.pt` |
| **Environnement** | Surveiller déforestation, pollution | `yolo26m-sem.pt` |
| **Retail** | Analyser rayons, comportement clients | `yolo26s-sem.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | mIoU | Idéal pour |
|--------|---------|--------|-------|------|------------|
| YOLO26n-sem | `yolo26n-sem.pt` | ~3M | ~8G | 38,5 | Mobile temps réel |
| YOLO26s-sem | `yolo26s-sem.pt` | ~11M | ~30G | 45,2 | Équilibre vitesse/précision |
| YOLO26m-sem | `yolo26m-sem.pt` | ~20M | ~55G | 49,8 | Précision améliorée |
| YOLO26l-sem | `yolo26l-sem.pt` | ~26M | ~80G | 52,1 | Haute précision |
| YOLO26x-sem | `yolo26x-sem.pt` | ~57M | ~170G | 54,5 | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle de segmentation sémantique pré-entraîné
model = YOLO("yolo26n-sem.pt")

# Vérifier les classes
print(model.names)  # {0: 'background', 1: 'road', ...}
```

---

## 3. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg")

result = results[0]

# Masques sémantiques
print(result.masks)           # Masks object
print(result.masks.data)      # Masques tensor — shape (N, H, W) ou (C, H, W)
print(result.masks.xy)        # Polygones de contour — liste de (M, 2) arrays

# Pour la segmentation sémantique, les masques représentent les classes
# Chaque pixel a une classe associée
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg")

result = results[0]

if result.masks is not None:
    # Masques de segmentation
    mask_data = result.masks.data.cpu().numpy()
    
    # Pour la segmentation sémantique, on peut avoir :
    # - Un masque par classe (C, H, W)
    # - Ou un masque avec les IDs de classe (H, W)
    
    print(f"Shape des masques: {mask_data.shape}")
    
    # Si c'est un masque de classes (H, W)
    if mask_data.ndim == 2:
        # Obtenir les classes uniques présentes
        unique_classes = np.unique(mask_data)
        print(f"Classes présentes: {unique_classes}")
        
        # Pour chaque classe
        for cls_id in unique_classes:
            cls_name = result.names.get(int(cls_id), f"class_{cls_id}")
            pixel_count = (mask_data == cls_id).sum()
            percentage = pixel_count / mask_data.size * 100
            print(f"  {cls_name}: {pixel_count} pixels ({percentage:.1f}%)")
```

### Format de sortie pour intégration mobile

```python
import json
import numpy as np

def semantic_to_json(result):
    if result.masks is None:
        return json.dumps([])
    
    mask_data = result.masks.data.cpu().numpy()
    
    # Si c'est un masque de classes (H, W)
    if mask_data.ndim == 2:
        unique_classes = np.unique(mask_data)
        segments = []
        
        for cls_id in unique_classes:
            cls_id_int = int(cls_id)
            mask = (mask_data == cls_id).astype(np.uint8)
            
            segments.append({
                "class_id": cls_id_int,
                "class_name": result.names.get(cls_id_int, f"class_{cls_id_int}"),
                "pixel_count": int(mask.sum()),
                "percentage": float(mask.sum() / mask_data.size * 100)
            })
        
        return json.dumps(segments)
    
    return json.dumps([])
```

---

## 4. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8
model.export(format="onnx", quantize=8, data="coco8-sem.yaml")
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
model.export(format="litert", quantize=8, data="coco8-sem.yaml")
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

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg", conf=0.5)

for result in results:
    result.show()  # Afficher avec masques sémantiques
    result.save(filename="result_sem.jpg")
```

### Extraire les classes par pixel

```python
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg")

result = results[0]

if result.masks is not None:
    mask_data = result.masks.data.cpu().numpy()
    
    # Obtenir les classes uniques
    unique_classes = np.unique(mask_data)
    
    print(f"Classes détectées: {len(unique_classes)}")
    for cls_id in unique_classes:
        cls_name = result.names.get(int(cls_id), f"class_{cls_id}")
        print(f"  {cls_name}")
```

### Calculer la surface par classe

```python
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg")

result = results[0]

if result.masks is not None:
    mask_data = result.masks.data.cpu().numpy()
    total_pixels = mask_data.size
    
    # Pour chaque classe
    for cls_id in np.unique(mask_data):
        cls_id_int = int(cls_id)
        cls_name = result.names.get(cls_id_int, f"class_{cls_id_int}")
        pixel_count = (mask_data == cls_id).sum()
        percentage = pixel_count / total_pixels * 100
        
        print(f"{cls_name}: {percentage:.1f}% de l'image")
```

### Appliquer une couleur par classe

```python
import cv2
import numpy as np
from ultralytics import YOLO

# Palette de couleurs pour les classes
COLOR_PALETTE = {
    0: (0, 0, 0),       # background - noir
    1: (128, 64, 128),  # road - violet
    2: (244, 35, 232),  # sidewalk - rose
    3: (70, 70, 70),    # building - gris
    # ... ajouter plus de couleurs
}

model = YOLO("yolo26n-sem.pt")
results = model("image.jpg")

result = results[0]
image = result.orig_img.copy()

if result.masks is not None:
    mask_data = result.masks.data.cpu().numpy()
    
    # Créer une image colorée
    colored_mask = np.zeros_like(image)
    
    for cls_id, color in COLOR_PALETTE.items():
        colored_mask[mask_data == cls_id] = color
    
    # Fusionner avec l'image originale
    alpha = 0.5
    output = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    cv2.imwrite("result_sem_colored.jpg", output)
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")

results = model("video.mp4", stream=True, conf=0.5)

for i, result in enumerate(results):
    if result.masks is not None:
        print(f"Frame {i}: segmentation sémantique terminée")
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
│  YOLO26-sem Model                                   │
│    ↓                                                 │
│  Results (pixel-wise class predictions)             │
│    ↓                                                 │
│  Semantic Overlay / Heatmap Rendering               │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
import 'package:flutter/services.dart';

class YOLOSemanticService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_sem');

  Future<List<Map<String, dynamic>>> segmentSemantic(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('segmentSemantic', {
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

const { YOLOSemantic } = NativeModules;

export async function segmentSemantic(imagePath, confidence = 0.5) {
  const result = await YOLOSemantic.segmentSemantic(imagePath, confidence);
  return result; // [{class_id, class_name, pixel_count, percentage}]
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLOSemanticSegmenter {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try! YOLO26n_sem(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func segment(pixelBuffer: CVPixelBuffer, completion: @escaping ([SemanticResult]) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNPixelBufferObservation] else { return }
            let segments = results.map { obs in
                SemanticResult(
                    classId: obs.identifier,
                    confidence: obs.confidence
                )
            }
            completion(segments)
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

class YOLOSemanticSegmenter(private val interpreter: Interpreter) {
    
    fun segment(imageData: ByteBuffer): List<SemanticResult> {
        val output = Array(1) { Array(224) { FloatArray(224) } }  // H x W class map
        interpreter.run(imageData, output)
        return parseResults(output[0])
    }
    
    private fun parseResults(output: Array<FloatArray>): List<SemanticResult> {
        val results = mutableListOf<SemanticResult>()
        // Parser les résultats selon le format de sortie
        return results
    }
}
```

---

## 7. Considérations Spécifiques au Mobile

### Gestion de la mémoire

| Conséquence | Solution |
|-------------|----------|
| Masques trop volumineux | Réduire la résolution de sortie |
| Trop de classes | Limiter aux classes pertinentes |
| Images haute résolution | Redimensionner avant inférence |
| Fuites mémoire | Libérer les masques après rendu |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=320` |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n-sem.pt` |
| GPU/NPU delegate | +5x vitesse | CoreML ANE / LiteRT GPU |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Rendu des masques sémantiques sur mobile

```dart
// Flutter — CustomPainter pour segmentation sémantique
class SemanticPainter extends CustomPainter {
  final List<Map<String, dynamic>> segments;
  final Map<int, Color> colorPalette;
  
  @override
  void paint(Canvas canvas, Size size) {
    for (final seg in segments) {
      final classId = seg['class_id'] as int;
      final color = colorPalette[classId] ?? Colors.transparent;
      
      final paint = Paint()
        ..color = color.withOpacity(0.5)
        ..style = PaintingStyle.fill;
      
      // Dessiner la région (simplifié - en pratique, utiliser un Path)
      // ...
    }
  }
}
```

---

## 8. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n-sem | ~58 | ~12 | 17-85 |
| YOLO26s-sem | ~135 | ~28 | 7-38 |
| YOLO26m-sem | ~270 | ~54 | 4-20 |
| YOLO26l-sem | ~430 | ~84 | 2-12 |
| YOLO26x-sem | ~860 | ~158 | 1-7 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n-sem.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 320 si possible
- [ ] Activer le delegate GPU/NPU
- [ ] Limiter le nombre de classes
- [ ] Utiliser `simplify=True` pour ONNX

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=320,
    conf=0.5,
    verbose=False
)
```

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Segmentation de mauvaise qualité | Résolution trop basse | Augmenter `imgsz` à 640 |
| Trop de classes | Jeu de données trop large | Filtrer les classes pertinentes |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-sem.pt` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Masques non affichés | `masks` est None | Vérifier que le modèle est bien un modèle `-sem` |
| Erreur mémoire | Trop de classes | Réduire le nombre de classes |
| Résultats différents après export | Mode end2end | Vérifier les paramètres d'export |

---

## Références

- [Documentation YOLO26-sem Ultralytics](https://docs.ultralytics.com/fr/tasks/segment/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
