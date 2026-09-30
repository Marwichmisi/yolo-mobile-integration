# Estimation de Profondeur avec YOLO26-depth — Guide Complet

> Guide de référence pour l'estimation de profondeur monoculaire avec YOLO26-depth — architecture, intégration mobile et cas d'utilisation.

---

## Table des matières

1. [Description de la Tâche](#1-description-de-la-tâche)
2. [Variantes de Modèles](#2-variantes-de-modèles)
3. [Format des Résultats](#3-format-des-résultats)
4. [Commandes d'Export](#4-commandes-dexport)
5. [Exemples d'Inférence](#5-exemples-dinférence)
6. [Intégration Mobile](#6-intégration-mobile)
7. [Considérations Mobiles](#7-considérations-mobiles)
8. [Conseils de Performance](#8-conseils-de-performance)
9. [Dépannage](#9-dépannage)
10. [Références](#10-références)

---

## 1. Description de la Tâche

YOLO26-depth est la première tâche d'estimation de profondeur monoculaire de la famille YOLO26. Elle prédit la profondeur relative de chaque pixel en mètres, sans avoir besoin de capteur de profondeur matériel.

### Cas d'utilisation mobiles

- **Réalité augmentée (AR)** : Mesure de distances, placement d'objets virtuels
- **Photographie** : Effet de flou d'arrière-field (portrait mode)
- **Navigation** : Estimation de distances pour les applications de mobilité
- **Accessibilité** : Aide aux personnes malvoyantes pour estimer les distances
- **Sport** : Analyse de mouvements et de distances parcourues

### Avantages

- **Monoculaire** : Fonctionne avec une seule caméra, pas de capteur spécial
- **Temps réel** : Optimisé pour l'inférence sur mobile
- **Précision** : Prédiction de profondeur en mètres (pas seulement relative)
- **Léger** : Modèles nano et small adaptés aux appareils mobiles

---

## 2. Variantes de Modèles

| Modèle | Paramètres | FLOPs | mAP (NYUv2) | Cas d'utilisation mobile |
|--------|-----------|-------|-------------|--------------------------|
| **YOLO26n-depth** | ~3M | ~8G | 0.412 | Mobile temps réel, AR |
| **YOLO26s-depth** | ~11M | ~30G | 0.438 | Équilibre vitesse/précision |
| **YOLO26m-depth** | ~20M | ~55G | 0.451 | Précision améliorée |
| **YOLO26l-depth** | ~26M | ~80G | 0.463 | Haute précision |
| **YOLO26x-depth** | ~57M | ~170G | 0.471 | Précision maximale |

### Noms de fichiers

```
yolo26n-depth.pt
yolo26s-depth.pt
yolo26m-depth.pt
yolo26l-depth.pt
yolo26x-depth.pt
```

---

## 3. Format des Résultats

### Sortie du modèle

```python
from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")
results = model("image.jpg")

# Résultats de profondeur
depth_map = results[0].depth  # Tensor (H, W) en mètres
```

### Format de la carte de profondeur

| Propriété | Type | Description |
|-----------|------|-------------|
| `depth` | `torch.Tensor` | Carte de profondeur (H, W) en mètres |
| `depth.shape` | `tuple` | (hauteur, largeur) de l'image d'entrée |
| `depth.min()` | `float` | Profondeur minimale en mètres |
| `depth.max()` | `float` | Profondeur maximale en mètres |
| `depth.mean()` | `float` | Profondeur moyenne en mètres |

### Exemple d'utilisation

```python
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n-depth.pt")
results = model("image.jpg")

# Obtenir la carte de profondeur
depth_map = results[0].depth

# Statistiques
print(f"Profondeur min: {depth_map.min():.2f} m")
print(f"Profondeur max: {depth_map.max():.2f} m")
print(f"Profondeur moyenne: {depth_map.mean():.2f} m")

# Profondeur à un point spécifique (x, y)
x, y = 320, 240
point_depth = depth_map[y, x]
print(f"Profondeur au point ({x}, {y}): {point_depth:.2f} m")
```

---

## 4. Commandes d'Export

### Export pour mobile

```python
from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")

# CoreML (iOS)
model.export(format="coreml", quantize=8)

# LiteRT (Android)
model.export(format="litert", quantize="w8a32")

# ONNX (universel)
model.export(format="onnx", quantize=8)

# NCNN (mobile léger)
model.export(format="ncnn")
```

### Export avec taille personnalisée

```python
# Pour AR (entrée carrée)
model.export(format="coreml", imgsz=640)

# Pour mobile (entrée réduite)
model.export(format="litert", imgsz=480)
```

---

## 5. Exemples d'Inférence

### Inférence simple

```python
from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")
results = model("image.jpg")

# Afficher la carte de profondeur
results[0].show()
```

### Inférence vidéo

```python
from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")

# Inférence sur vidéo
results = model("video.mp4", stream=True)

for result in results:
    depth_map = result.depth
    # Traiter la carte de profondeur
    print(f"Profondeur moyenne: {depth_map.mean():.2f} m")
```

### Inférence avec seuil de confiance

```python
from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")

# Inférence avec seuil de confiance
results = model("image.jpg", conf=0.5)

# Filtrage par profondeur
depth_map = results[0].depth
mask = depth_map < 10.0  # Objets à moins de 10 mètres
```

---

## 6. Intégration Mobile

### Flutter

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class DepthScreen extends StatefulWidget {
  @override
  _DepthScreenState createState() => _DepthScreenState();
}

class _DepthScreenState extends State<DepthScreen> {
  CameraController? _controller;
  List<double> _depthValues = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _controller = CameraController(
      cameras[0],
      ResolutionPreset.medium,
    );
    await _controller!.initialize();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Column(
      children: [
        Expanded(
          child: CameraPreview(_controller!),
        ),
        // Overlay de profondeur
        Container(
          height: 100,
          child: CustomPaint(
            painter: DepthOverlayPainter(_depthValues),
          ),
        ),
      ],
    );
  }
}
```

### iOS (Swift)

```swift
import UIKit
import UltralyticsYOLO

class DepthViewController: UIViewController {
    private var yolo: YOLO?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadModel()
    }
    
    func loadModel() {
        yolo = YOLO("yolo26n-depth", task: .depth) { result in
            switch result {
            case .success(let model):
                print("Modèle de profondeur chargé")
            case .failure(let error):
                print("Erreur: \(error)")
            }
        }
    }
    
    func estimateDepth(image: UIImage) {
        guard let yolo = yolo else { return }
        
        let results = yolo(image)
        // Traiter la carte de profondeur
    }
}
```

### Android (Kotlin)

```kotlin
import com.ultralytics.lite.Interpreter

class DepthActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        loadModel()
    }
    
    fun loadModel() {
        val modelFile = loadModelFile("yolo26n-depth.tflite")
        interpreter = Interpreter(modelFile)
    }
    
    fun estimateDepth(bitmap: Bitmap): Array<FloatArray> {
        val input = preprocess(bitmap)
        val output = Array(1) { Array(640) { FloatArray(640) } }
        interpreter.run(input, output)
        return output[0]
    }
}
```

---

## 7. Considérations Mobiles

### Mémoire

| Modèle | Mémoire (inférence) | Recommandation |
|--------|---------------------|----------------|
| YOLO26n-depth | ~50 MB | Idéal pour mobile |
| YOLO26s-depth | ~120 MB | Acceptable |
| YOLO26m-depth | ~250 MB | Sur tablette |
| YOLO26l-depth | ~400 MB | Éviter sur mobile |
| YOLO26x-depth | ~800 MB | Serveur uniquement |

### Optimisations

1. **Réduire la résolution** : `imgsz=480` au lieu de 640
2. **Quantification INT8** : Réduit la taille et accélère l'inférence
3. **Frame throttling** : Traiter 1 frame sur 2 ou 3
4. **GPU delegation** : Utiliser le GPU mobile si disponible

---

## 8. Conseils de Performance

### Latence d'inférence (iPhone 17 Pro)

| Modèle | CPU (ms) | GPU (ms) |
|--------|----------|----------|
| YOLO26n-depth | 25.0 | 5.3 |
| YOLO26s-depth | 45.0 | 8.7 |
| YOLO26m-depth | 78.0 | 14.2 |

### Checklist de performance

- [ ] Utiliser YOLO26n-depth pour le temps réel
- [ ] Quantifier en INT8 pour CoreML
- [ ] Réduire la résolution si nécessaire
- [ ] Utiliser le GPU/ANE si disponible
- [ ] Limiter le taux d'images par seconde
- [ ] Libérer la mémoire entre les inférences

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Profondeur incorrecte | Modèle non calibré | Utiliser un modèle pré-entraîné |
| Latence élevée | Modèle trop grand | Passer à YOLO26n-depth |
| Mémoire insuffisante | Résolution trop élevée | Réduire `imgsz` |
| Export CoreML échoue | macOS requis | Exporter sur macOS |
| Résultats incohérents | Éclairage variable | Normaliser l'image d'entrée |

---

## 10. Références

- [Documentation YOLO26](https://docs.ultralytics.com/fr/models/yolo26)
- [Tâches Ultralytics](https://docs.ultralytics.com/fr/tasks)
- [Export CoreML](https://docs.ultralytics.com/fr/integrations/coreml)
- [Export LiteRT](https://docs.ultralytics.com/fr/integrations/litert)
