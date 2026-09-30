# Référence Complète : Estimation de Pose avec YOLO26-pose

> Guide complet pour l'intégration de l'estimation de pose dans une application mobile utilisant YOLO26-pose — 17 keypoints COCO, squelette, et déploiement multiplateforme.

---

## Table des matières

1. [Description de la Tâche et Cas d'Utilisation](#1-description-de-la-tâche-et-cas-dutilisation)
2. [Variantes de Modèles et Nommage](#2-variantes-de-modèles-et-nommage)
3. [Les 17 Keypoints COCO](#3-les-17-keypoints-coco)
4. [Définition du Squelette](#4-définition-du-squelette)
5. [Format des Résultats](#5-format-des-résultats)
6. [Commandes d'Export pour Mobiles](#6-commandes-dexport-pour-mobiles)
7. [Exemples de Code d'Inférence](#7-exemples-de-code-dinférence)
8. [Intégration Mobile](#8-intégration-mobile)
9. [Considérations Spécifiques au Mobile](#9-considérations-spécifiques-au-mobile)
10. [Conseils de Performance](#10-conseils-de-performance)
11. [Dépannage](#11-dépannage)

---

## 1. Description de la Tâche et Cas d'Utilisation

L'estimation de pose détecte les points clés du corps humain (keypoints) dans une image et les relie pour former un squelette. YOLO26-pose détecte 17 keypoints COCO par personne.

### Cas d'utilisation mobiles

| Cas d'utilisation | Description | Modèle recommandé |
|-------------------|-------------|-------------------|
| **Coach fitness** | Analyser forme, compter répétitions | `yolo26n-pose.pt` |
| **Yoga / Pilates** | Vérifier posture, donner corrections | `yolo26s-pose.pt` |
| **Rééducation** | Suivre progression, mesurer amplitude | `yolo26m-pose.pt` |
| **Jeux interactifs** | Contrôler personnage par le corps | `yolo26n-pose.pt` |
| **Sécurité** | Détecter chutes, comportements anormaux | `yolo26s-pose.pt` |
| **Mode virtuel** | Essayage virtuel de vêtements | `yolo26m-pose.pt` |
| **Sport de haut niveau** | Analyse technique, biomécanique | `yolo26l-pose.pt` |
| **Accessibilité** | Contrôle gestuel pour handicapés | `yolo26n-pose.pt` |

---

## 2. Variantes de Modèles et Nommage

### Modèles disponibles

| Modèle | Fichier | Params | FLOPs | mAP COCO | Idéal pour |
|--------|---------|--------|-------|----------|------------|
| YOLO26n-pose | `yolo26n-pose.pt` | ~3M | ~8G | 68,5 | Mobile temps réel |
| YOLO26s-pose | `yolo26s-pose.pt` | ~11M | ~30G | 74,2 | Équilibre vitesse/précision |
| YOLO26m-pose | `yolo26m-pose.pt` | ~20M | ~55G | 77,8 | Précision améliorée |
| YOLO26l-pose | `yolo26l-pose.pt` | ~26M | ~80G | 79,5 | Haute précision |
| YOLO26x-pose | `yolo26x-pose.pt` | ~57M | ~170G | 81,2 | Précision maximale |

### Chargement du modèle

```python
from ultralytics import YOLO

# Charger un modèle de pose pré-entraîné
model = YOLO("yolo26n-pose.pt")

# Vérifier la configuration des keypoints
print(model.kpt_shape)  # [17, 3] — 17 keypoints, (x, y, visibility)
```

---

## 3. Les 17 Keypoints COCO

### Indices et noms

| Indice | Nom | Description |
|--------|-----|-------------|
| 0 | `NOSE` | Pointe du nez |
| 1 | `LEFT_EYE` | Œil gauche |
| 2 | `RIGHT_EYE` | Œil droit |
| 3 | `LEFT_EAR` | Oreille gauche |
| 4 | `RIGHT_EAR` | Oreille droite |
| 5 | `LEFT_SHOULDER` | Épaule gauche |
| 6 | `RIGHT_SHOULDER` | Épaule droite |
| 7 | `LEFT_ELBOW` | Coude gauche |
| 8 | `RIGHT_ELBOW` | Coude droit |
| 9 | `LEFT_WRIST` | Poignet gauche |
| 10 | `RIGHT_WRIST` | Poignet droit |
| 11 | `LEFT_HIP` | Hanche gauche |
| 12 | `RIGHT_HIP` | Hanche droite |
| 13 | `LEFT_KNEE` | Genou gauche |
| 14 | `RIGHT_KNEE` | Genou droit |
| 15 | `LEFT_ANKLE` | Cheville gauche |
| 16 | `RIGHT_ANKLE` | Cheville droite |

### Constantes d'indices

```python
class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
```

---

## 4. Définition du Squelette

### Connexions COCO

```python
# Paires de keypoints pour dessiner le squelette
COCO_SKELETON = [
    [0, 1],   # nez → œil gauche
    [0, 2],   # nez → œil droit
    [1, 3],   # œil gauche → oreille gauche
    [2, 4],   # œil droit → oreille droite
    [5, 6],   # épaule gauche → épaule droite
    [5, 7],   # épaule gauche → coude gauche
    [7, 9],   # coude gauche → poignet gauche
    [6, 8],   # épaule droite → coude droit
    [8, 10],  # coude droit → poignet droit
    [5, 11],  # épaule gauche → hanche gauche
    [6, 12],  # épaule droite → hanche droite
    [11, 12], # hanche gauche → hanche droite
    [11, 13], # hanche gauche → genou gauche
    [13, 15], # genou gauche → cheville gauche
    [12, 14], # hanche droite → genou droit
    [14, 16], # genou droit → cheville droit
]
```

### Groupes articulaires

| Groupe | Indices | Couleur recommandée |
|--------|---------|---------------------|
| Tête | 0-4 | Bleu |
| Épaules | 5-6 | Orange |
| Bras | 7-10 | Rouge |
| Hanches | 11-12 | Violet |
| Jambes | 13-16 | Vert |

---

## 5. Format des Résultats

### Structure de résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("image.jpg")

result = results[0]

# Keypoints
print(result.keypoints)           # Keypoints object
print(result.keypoints.xy)        # Coordonnées (x, y) — shape (N, 17, 2)
print(result.keypoints.conf)      # Confiances — shape (N, 17)
print(result.keypoints.xyn)      # Coordonnées normalisées — shape (N, 17, 2)

# Boîtes englobantes
print(result.boxes)               # Bboxes object
print(result.boxes.xyxy)          # Coordonnées (x1, y1, x2, y2) — shape (N, 4)
print(result.boxes.conf)          # Scores de confiance — shape (N,)
print(result.boxes.cls)           # IDs de classe — shape (N,)
```

### Exemple d'accès aux résultats

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("image.jpg")

for result in results:
    keypoints = result.keypoints
    boxes = result.boxes
    
    for i in range(len(boxes)):
        # Keypoints de la personne i
        kp_xy = keypoints.xy[i].cpu().numpy()    # (17, 2)
        kp_conf = keypoints.conf[i].cpu().numpy() # (17,)
        
        # Boîte englobante
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf = boxes.conf[i].item()
        
        print(f"Personne {i}: confiance {conf:.2f}")
        for j in range(17):
            x, y = kp_xy[j]
            c = kp_conf[j]
            print(f"  Keypoint {j}: ({x:.1f}, {y:.1f}) conf={c:.2f}")
```

### Format de sortie pour intégration mobile

```python
import json

def pose_to_json(result):
    keypoints = result.keypoints
    boxes = result.boxes
    persons = []
    
    for i in range(len(boxes)):
        kp_xy = keypoints.xy[i].cpu().numpy()
        kp_conf = keypoints.conf[i].cpu().numpy()
        
        persons.append({
            "bbox": boxes.xyxy[i].tolist(),
            "confidence": boxes.conf[i].item(),
            "keypoints": [
                {
                    "x": float(kp_xy[j][0]),
                    "y": float(kp_xy[j][1]),
                    "confidence": float(kp_conf[j])
                }
                for j in range(17)
            ]
        })
    
    return json.dumps(persons)
```

---

## 6. Commandes d'Export pour Mobiles

### Export universel (ONNX)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

# ONNX — format universel
model.export(format="onnx", simplify=True, opset=17)

# ONNX INT8
model.export(format="onnx", quantize=8, data="coco8-pose.yaml")
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
model.export(format="litert", quantize=8, data="coco8-pose.yaml")
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

## 7. Exemples de Code d'Inférence

### Inférence simple

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("image.jpg", conf=0.5)

for result in results:
    result.show()  # Afficher avec squelette
    result.save(filename="result_pose.jpg")
```

### Extraire les keypoints

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("image.jpg")

result = results[0]
keypoints = result.keypoints

if keypoints is not None:
    # Pour chaque personne
    for i in range(len(result.boxes)):
        kp_xy = keypoints.xy[i].cpu().numpy()    # (17, 2)
        kp_conf = keypoints.conf[i].cpu().numpy() # (17,)
        
        # Filtrer les keypoints visibles (confiance > 0.5)
        visible = kp_conf > 0.5
        print(f"Personne {i}: {visible.sum()}/17 keypoints visibles")
```

### Calculer l'angle articulaire

```python
import numpy as np

def calculate_angle(a, b, c):
    """Calcule l'angle en degrés entre 3 points (A → B → C où B est le sommet)"""
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))

# Exemple : angle du genou
kp = keypoints.xy[0].cpu().numpy()
knee_angle = calculate_angle(kp[11], kp[13], kp[15])  # hanche → genou → cheville
print(f"Angle du genou: {knee_angle:.1f}°")
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

results = model("video.mp4", stream=True, conf=0.5)

for i, result in enumerate(results):
    if result.keypoints is not None:
        print(f"Frame {i}: {len(result.boxes)} personnes détectées")
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
│  YOLO26-pose Model                                  │
│    ↓                                                 │
│  Results (keypoints, confidence, boxes)             │
│    ↓                                                 │
│  Skeleton Rendering / Overlay                       │
└─────────────────────────────────────────────────────┘
```

### Flutter — MethodChannel

```dart
import 'package:flutter/services.dart';

class YOLOPoseService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_pose');

  Future<List<Map<String, dynamic>>> detectPose(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('detectPose', {
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

const { YOLOPose } = NativeModules;

export async function detectPose(imagePath, confidence = 0.5) {
  const result = await YOLOPose.detectPose(imagePath, confidence);
  return result; // [{bbox, confidence, keypoints: [{x, y, confidence}]}]
}
```

### iOS — Swift (CoreML)

```swift
import CoreML
import Vision

class YOLOPoseDetector {
    private var model: VNCoreMLModel?
    
    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try! YOLO26n_pose(configuration: config)
        model = try! VNCoreMLModel(for: coreMLModel.model)
    }
    
    func detect(pixelBuffer: CVPixelBuffer, completion: @escaping ([PoseResult]) -> Void) {
        let request = VNCoreMLRequest(model: model!) { request, error in
            guard let results = request.results as? [VNRecognizedPointsObservation] else { return }
            let poses = results.map { obs in
                PoseResult(
                    keypoints: obs.allPoints,
                    confidence: obs.confidence
                )
            }
            completion(poses)
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

class YOLOPoseDetector(private val interpreter: Interpreter) {
    
    fun detect(imageData: ByteBuffer): List<PoseResult> {
        val output = Array(1) { Array(300) { FloatArray(56) } }  // 4 bbox + 17*3 keypoints
        interpreter.run(imageData, output)
        return parsePoses(output[0])
    }
    
    private fun parsePoses(output: Array<FloatArray>): List<PoseResult> {
        val poses = mutableListOf<PoseResult>()
        for (i in output.indices) {
            val conf = output[i][4]
            if (conf > 0.5f) {
                val keypoints = (0 until 17).map { j ->
                    Keypoint(
                        x = output[i][5 + j * 3],
                        y = output[i][6 + j * 3],
                        confidence = output[i][7 + j * 3]
                    )
                }
                poses.add(PoseResult(keypoints, conf))
            }
        }
        return poses
    }
}
```

---

## 9. Considérations Spécifiques au Mobile

### Gestion de la mémoire

| Conséquence | Solution |
|-------------|----------|
| Trop de personnes | Limiter `max_det=5` |
| Keypoints trop précis | Filtrer par confiance (> 0.5) |
| Images haute résolution | Redimensionner avant inférence |
| Fuites mémoire | Libérer les résultats après rendu |

### Optimisations temps réel

| Technique | Impact | Implémentation |
|-----------|--------|----------------|
| Réduire `imgsz` | +50% vitesse | `imgsz=320` |
| Quantification INT8 | -75% taille | `quantize=8` |
| Modèle Nano | +3x vitesse | `yolo26n-pose.pt` |
| GPU/NPU delegate | +5x vitesse | CoreML ANE / LiteRT GPU |
| Frame skipping | Réduire charge | Traiter 1 frame sur 2 |

### Rendu du squelette sur mobile

```dart
// Flutter — CustomPainter pour squelette
class PosePainter extends CustomPainter {
  final List<Map<String, dynamic>> poses;
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    
    for (final pose in poses) {
      final keypoints = pose['keypoints'] as List;
      
      // Dessiner les connexions du squelette
      for (final pair in COCO_SKELETON) {
        final kpA = keypoints[pair[0]];
        final kpB = keypoints[pair[1]];
        
        if (kpA['confidence'] > 0.5 && kpB['confidence'] > 0.5) {
          paint.color = Colors.green.withOpacity(0.8);
          canvas.drawLine(
            Offset(kpA['x'], kpA['y']),
            Offset(kpB['x'], kpB['y']),
            paint,
          );
        }
      }
      
      // Dessiner les keypoints
      for (final kp in keypoints) {
        if (kp['confidence'] > 0.5) {
          canvas.drawCircle(
            Offset(kp['x'], kp['y']),
            5,
            Paint()..color = Colors.red,
          );
        }
      }
    }
  }
}
```

---

## 10. Conseils de Performance

### Latence par modèle (mobile)

| Modèle | CPU (ms) | GPU/NPU (ms) | FPS estimé |
|--------|----------|--------------|------------|
| YOLO26n-pose | ~55 | ~11 | 18-90 |
| YOLO26s-pose | ~130 | ~28 | 8-38 |
| YOLO26m-pose | ~260 | ~52 | 4-20 |
| YOLO26l-pose | ~420 | ~82 | 2-12 |
| YOLO26x-pose | ~850 | ~155 | 1-7 |

### Checklist d'optimisation

- [ ] Utiliser `yolo26n-pose.pt` pour le mobile
- [ ] Quantifier en INT8 (`quantize=8`)
- [ ] Réduire `imgsz` à 320 si possible
- [ ] Filtrer les keypoints par confiance (> 0.5)
- [ ] Activer le delegate GPU/NPU
- [ ] Limiter `max_det=5`
- [ ] Utiliser `simplify=True` pour ONNX

### Code d'optimisation

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

# Configuration optimale mobile
results = model(
    "image.jpg",
    imgsz=320,
    conf=0.5,
    max_det=5,        # Limiter les personnes
    verbose=False
)
```

---

## 11. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Keypoints tremblants | Bruit de détection | Ajouter un filtre passe-bas |
| Personne non détectée | Éclairage/angle | Améliorer l'éclairage |
| Keypoints imprécis | Résolution trop basse | Augmenter `imgsz` à 640 |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-pose.pt` |
| Modèle trop volumineux | Pas de quantification | Ajouter `quantize=8` |
| Squelette incorrect | Mauvaises connexions | Vérifier `COCO_SKELETON` |
| Erreur mémoire | Trop de personnes | Réduire `max_det` |
| Résultats différents après export | Mode end2end | Vérifier les paramètres d'export |

---

## Références

- [Documentation YOLO26-pose Ultralytics](https://docs.ultralytics.com/fr/tasks/pose/)
- [Keypoints COCO](https://cocodataset.org/#keypoints-eval)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Intégration React Native](../frameworks/react-native-integration.md)
- [Déploiement iOS](../platforms/ios-coreml.md)
- [Déploiement Android](../platforms/android-litert.md)
