# Référence Complète : Mesure de Distance avec YOLO26

> Guide complet pour l'intégration d'un système de mesure de distance dans une application mobile Flutter utilisant YOLO26 pour la conversion pixel-monde réel, l'estimation de profondeur et la visualisation AR.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application de **mesure de distance / AR**
- Le projet nécessite la **conversion pixel → monde réel**
- Il faut de l'**estimation de profondeur** avec YOLO26
- L'application doit afficher un **overlay AR** de distances
- Le modèle **YOLO26** est utilisé pour la détection + mesure
- L'utilisateur mentionne : mesure, distance, AR, réalité augmentée, profondeur, dimension, taille

**Ne pas utiliser cette référence pour :**
- Sécurité/alarme (utiliser `security-alarm.md`)
- Gestion de parking (utiliser `parking-management.md`)
- Analyse de heatmap (utiliser `heatmap-analytics.md`)

---

## Table des matières

1. [Architecture de Mesure de Distance](#1-architecture-de-mesure-de-distance)
2. [Conversion Pixel vers Monde Réel](#2-conversion-pixel-vers-monde-réel)
3. [Estimation de Profondeur avec YOLO26](#3-estimation-de-profondeur-avec-yolo26)
4. [Overlay AR de Visualisation](#4-overlay-ar-de-visualisation)
5. [Exemple Complet Flutter](#5-exemple-complet-flutter)
6. [Optimisations Mobile](#6-optimisations-mobile)
7. [Dépannage](#7-dépannage)

---

## 1. Architecture de Mesure de Distance

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                   Application Flutter                     │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Camera  │→  │  YOLO26  │→ │  Distance Calculator │ │
│  │  Stream  │   │  Detect  │   │  (pixel → real world)│ │
│  └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                           │              │
│  ┌──────────┐   ┌──────────┐   ┌──────────▼───────────┐ │
│  │   AR     │←  │ Overlay  │← │  Depth Estimator     │ │
│  │   View   │   │ Renderer │   │  (YOLO26-depth)      │ │
│  └──────────┘   └──────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Détection + mesure | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

---

## 2. Conversion Pixel vers Monde Réel

### 2.1 Principe de conversion

La conversion pixel → monde réel utilise la géométrie projective et les paramètres intrinsèques de la caméra.

```
Équation de base:
    distance_réelle = (distance_pixels × distance_réelle_référence) / distance_pixels_référence

Ou avec la distance focale:
    Z = (f × W_réel) / W_pixels

Où:
    Z = distance de la caméra à l'objet
    f = distance focale en pixels
    W_réel = largeur réelle de l'objet (mètres)
    W_pixels = largeur de l'objet en pixels
```

### 2.2 Calculateur de distance

```dart
import 'dart:math' as math;

/// Paramètres intrinsèques de la caméra
class CameraIntrinsics {
  final double focalLengthX;  // fx en pixels
  final double focalLengthY;  // fy en pixels
  final double principalPointX; // cx
  final double principalPointY; // cy
  final double sensorWidth;   // Largeur du capteur en mm
  final double sensorHeight;  // Hauteur du capteur en mm
  final int imageWidth;       // Largeur de l'image en pixels
  final int imageHeight;      // Hauteur de l'image en pixels

  const CameraIntrinsics({
    required this.focalLengthX,
    required this.focalLengthY,
    required this.principalPointX,
    required this.principalPointY,
    required this.sensorWidth,
    required this.sensorHeight,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Calcule la distance focale en pixels à partir du FOV
  static CameraIntrinsics fromFov({
    required double fovX,     // FOV horizontal en degrés
    required double fovY,     // FOV vertical en degrés
    required int imageWidth,
    required int imageHeight,
  }) {
    final fx = imageWidth / (2 * math.tan(fovX * math.pi / 360));
    final fy = imageHeight / (2 * math.tan(fovY * math.pi / 360));

    return CameraIntrinsics(
      focalLengthX: fx,
      focalLengthY: fy,
      principalPointX: imageWidth / 2,
      principalPointY: imageHeight / 2,
      sensorWidth: 0, // Non utilisé dans cette méthode
      sensorHeight: 0,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );
  }
}

/// Objet de référence pour l'étalonnage
class ReferenceObject {
  final String label;
  final double realWidth;   // Largeur réelle en mètres
  final double realHeight;  // Hauteur réelle en mètres

  const ReferenceObject({
    required this.label,
    required this.realWidth,
    required this.realHeight,
  });
}

/// Calculateur de distance
class DistanceCalculator {
  final CameraIntrinsics intrinsics;
  final Map<String, ReferenceObject> referenceObjects;

  DistanceCalculator({
    required this.intrinsics,
    required this.referenceObjects,
  });

  /// Calcule la distance à un objet détecté
  /// Utilise la largeur ou hauteur réelle connue de l'objet
  double? calculateDistance(Detection detection) {
    final ref = referenceObjects[detection.label];
    if (ref == null) return null;

    // Largeur de l'objet en pixels
    final pixelWidth = detection.width * intrinsics.imageWidth;
    final pixelHeight = detection.height * intrinsics.imageHeight;

    // Calculer la distance avec la largeur et la hauteur
    final distanceFromWidth = (intrinsics.focalLengthX * ref.realWidth) / pixelWidth;
    final distanceFromHeight = (intrinsics.focalLengthY * ref.realHeight) / pixelHeight;

    // Moyenne des deux estimations
    return (distanceFromWidth + distanceFromHeight) / 2;
  }

  /// Calcule la distance entre deux objets détectés
  double? calculateDistanceBetween(Detection det1, Detection det2) {
    final dist1 = calculateDistance(det1);
    final dist2 = calculateDistance(det2);
    if (dist1 == null || dist2 == null) return null;

    // Distance euclidienne dans l'image
    final dx = (det1.centerX - det2.centerX) * intrinsics.imageWidth;
    final dy = (det1.centerY - det2.centerY) * intrinsics.imageHeight;

    // Distance réelle approximative (si les objets sont à peu près à la même distance)
    final avgDistance = (dist1 + dist2) / 2;
    final pixelDistance = math.sqrt(dx * dx + dy * dy);

    return (avgDistance * pixelDistance) / intrinsics.focalLengthX;
  }

  /// Calcule les dimensions réelles d'un objet
  RealWorldDimensions? calculateRealWorldSize(Detection detection) {
    final distance = calculateDistance(detection);
    if (distance == null) return null;

    final pixelWidth = detection.width * intrinsics.imageWidth;
    final pixelHeight = detection.height * intrinsics.imageHeight;

    final realWidth = (pixelWidth * distance) / intrinsics.focalLengthX;
    final realHeight = (pixelHeight * distance) / intrinsics.focalLengthY;

    return RealWorldDimensions(
      width: realWidth,
      height: realHeight,
      distance: distance,
    );
  }
}

/// Dimensions réelles d'un objet
class RealWorldDimensions {
  final double width;    // en mètres
  final double height;   // en mètres
  final double distance; // en mètres

  const RealWorldDimensions({
    required this.width,
    required this.height,
    required this.distance,
  });

  double get area => width * height;

  @override
  String toString() =>
      '${(width * 100).toStringAsFixed(1)}cm × ${(height * 100).toStringAsFixed(1)}cm '
      'à ${distance.toStringAsFixed(2)}m';
}
```

### 2.3 Objets de référence COCO

```dart
class ReferenceObjects {
  /// Dimensions réelles moyennes des objets COCO (en mètres)
  static const Map<String, ReferenceObject> cocoReferences = {
    'person': ReferenceObject(label: 'person', realWidth: 0.5, realHeight: 1.7),
    'car': ReferenceObject(label: 'car', realWidth: 1.8, realHeight: 1.5),
    'truck': ReferenceObject(label: 'truck', realWidth: 2.5, realHeight: 2.5),
    'bus': ReferenceObject(label: 'bus', realWidth: 2.5, realHeight: 12.0),
    'bicycle': ReferenceObject(label: 'bicycle', realWidth: 0.6, realHeight: 1.0),
    'motorcycle': ReferenceObject(label: 'motorcycle', realWidth: 0.8, realHeight: 1.2),
    'chair': ReferenceObject(label: 'chair', realWidth: 0.5, realHeight: 1.0),
    'table': ReferenceObject(label: 'dining table', realWidth: 1.5, realHeight: 0.75),
  };

  /// Récupère un objet de référence
  static ReferenceObject? get(String label) => cocoReferences[label];
}
```

---

## 3. Estimation de Profondeur avec YOLO26

### 3.1 Principe de l'estimation de profondeur

L'estimation de profondeur mono-caméra utilise un modèle entraîné pour prédire la profondeur relative à partir d'une seule image.

```
Image → YOLO26-depth → Carte de profondeur → Distance par pixel

┌─────────┐    ┌──────────────┐    ┌─────────────────┐
│  Image  │ →  │  YOLO26-depth │ →  │  Depth Map      │
│  RGB    │    │  (inférence)  │    │  (H × W × 1)    │
└─────────┘    └──────────────┘    └─────────────────┘
```

### 3.2 Estimateur de profondeur

```dart
/// Estimateur de profondeur
class DepthEstimator {
  final CameraIntrinsics intrinsics;
  bool _isInitialized = false;

  Future<void> initialize() async {
    // Charger le modèle de profondeur
    // _model = await Tflite.loadModel(
    //   model: 'assets/yolo26n-depth.tflite',
    // );
    _isInitialized = true;
  }

  /// Estime la profondeur à un point spécifique
  /// Retourne la distance en mètres
  double? estimateDepthAt(Offset point, List<Detection> detections) {
    if (!_isInitialized) return null;

    // Trouver la détection qui contient ce point
    for (final det in detections) {
      if (point.dx >= det.x &&
          point.dx <= det.x + det.width &&
          point.dy >= det.y &&
          point.dy <= det.y + det.height) {
        // Utiliser la distance de la détection comme approximation
        return _estimateFromDetection(det);
      }
    }

    return null;
  }

  /// Estime la profondeur à partir d'une détection
  double? _estimateFromDetection(Detection det) {
    // Approximation basée sur la taille de la boîte
    // Plus l'objet est grand, plus il est proche
    final pixelArea = det.width * det.height;
    if (pixelArea <= 0) return null;

    // Formule simplifiée: distance ∝ 1 / sqrt(aire)
    // Constante calibrée pour une caméra typique
    const calibrationConstant = 0.5;
    return calibrationConstant / math.sqrt(pixelArea);
  }

  /// Estime les distances pour toutes les détections
  Map<String, double> estimateAllDepths(List<Detection> detections) {
    final depths = <String, double>{};

    for (final det in detections) {
      final depth = _estimateFromDetection(det);
      if (depth != null) {
        depths['${det.label}_${det.hashCode}'] = depth;
      }
    }

    return depths;
  }

  void dispose() {
    // _model?.close();
    _isInitialized = false;
  }
}
```

---

## 4. Overlay AR de Visualisation

### 4.1 Rendu AR des distances

```dart
/// Overlay AR pour la visualisation des distances
class AROverlayPainter extends CustomPainter {
  final List<Detection> detections;
  final DistanceCalculator? distanceCalculator;
  final DepthEstimator? depthEstimator;
  final CameraIntrinsics? intrinsics;

  AROverlayPainter({
    required this.detections,
    this.distanceCalculator,
    this.depthEstimator,
    this.intrinsics,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (final det in detections) {
      _drawDetectionWithDistance(canvas, size, det);
    }
  }

  void _drawDetectionWithDistance(Canvas canvas, Size size, Detection det) {
    final rect = Rect.fromLTWH(
      det.x * size.width,
      det.y * size.height,
      det.width * size.width,
      det.height * size.height,
    );

    // Boîte de détection
    canvas.drawRect(
      rect,
      Paint()
        ..color = Colors.cyan
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Calculer la distance
    String? distanceText;
    if (distanceCalculator != null) {
      final distance = distanceCalculator!.calculateDistance(det);
      if (distance != null) {
        distanceText = '${distance.toStringAsFixed(2)}m';
      }
    }

    // Calculer les dimensions réelles
    String? sizeText;
    if (distanceCalculator != null) {
      final dims = distanceCalculator!.calculateRealWorldSize(det);
      if (dims != null) {
        sizeText = '${(dims.width * 100).toStringAsFixed(0)}×${(dims.height * 100).toStringAsFixed(0)}cm';
      }
    }

    // Dessiner les informations
    final infoLines = <String>[
      det.label,
      if (distanceText != null) distanceText,
      if (sizeText != null) sizeText,
    ];

    _drawInfoBox(canvas, rect, infoLines);
  }

  void _drawInfoBox(Canvas canvas, Rect rect, List<String> lines) {
    const padding = 8.0;
    const lineHeight = 16.0;

    final boxWidth = lines
            .map((l) => l.length * 7.0)
            .reduce((a, b) => math.max(a, b)) +
        padding * 2;
    final boxHeight = lines.length * lineHeight + padding * 2;

    final boxRect = Rect.fromLTWH(
      rect.left,
      rect.top - boxHeight,
      boxWidth,
      boxHeight,
    );

    // Fond
    canvas.drawRRect(
      RRect.fromRectAndRadius(boxRect, const Radius.circular(4)),
      Paint()..color = Colors.black.withOpacity(0.7),
    );

    // Bordure
    canvas.drawRRect(
      RRect.fromRectAndRadius(boxRect, const Radius.circular(4)),
      Paint()
        ..color = Colors.cyan
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1,
    );

    // Texte
    for (int i = 0; i < lines.length; i++) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: lines[i],
          style: TextStyle(
            color: i == 0 ? Colors.cyan : Colors.white,
            fontSize: 12,
            fontWeight: i == 0 ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          boxRect.left + padding,
          boxRect.top + padding + i * lineHeight,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant AROverlayPainter oldDelegate) => true;
}
```

---

## 5. Exemple Complet Flutter

### Structure du projet

```
lib/
├── main.dart
├── models/
│   ├── detection.dart
│   └── camera_intrinsics.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── distance_calculator.dart
│   └── depth_estimator.dart
├── screens/
│   └── distance_measurement_screen.dart
└── widgets/
    └── ar_overlay_painter.dart
```

### `screens/distance_measurement_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../models/camera_intrinsics.dart';
import '../services/yolo_detection_service.dart';
import '../services/distance_calculator.dart';
import '../services/depth_estimator.dart';
import '../widgets/ar_overlay_painter.dart';

class DistanceMeasurementScreen extends StatefulWidget {
  const DistanceMeasurementScreen({super.key});

  @override
  State<DistanceMeasurementScreen> createState() =>
      _DistanceMeasurementScreenState();
}

class _DistanceMeasurementScreenState extends State<DistanceMeasurementScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  late final DistanceCalculator _distanceCalculator;
  final DepthEstimator _depthEstimator = DepthEstimator();

  List<Detection> _detections = [];
  bool _isProcessing = false;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _yoloService.initialize();
    await _depthEstimator.initialize();

    // Configuration de la caméra (FOV typique pour un smartphone)
    final intrinsics = CameraIntrinsics.fromFov(
      fovX: 60,
      fovY: 45,
      imageWidth: 640,
      imageHeight: 480,
    );

    _distanceCalculator = DistanceCalculator(
      intrinsics: intrinsics,
      referenceObjects: ReferenceObjects.cocoReferences,
    );

    // Initialiser la caméra
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController!.initialize();
    _cameraController!.startImageStream(_processFrame);

    setState(() => _isInitialized = true);
  }

  Future<void> _processFrame(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final bytes = _imageToBytes(image);
      final detections = await _yoloService.detect(bytes);

      setState(() {
        _detections = detections;
      });
    } catch (e) {
      debugPrint('Erreur: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Uint8List _imageToBytes(CameraImage image) {
    return image.planes[0].bytes;
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mesure de Distance'),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          CustomPaint(
            painter: AROverlayPainter(
              detections: _detections,
              distanceCalculator: _distanceCalculator,
              depthEstimator: _depthEstimator,
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _yoloService.dispose();
    _depthEstimator.dispose();
    super.dispose();
  }
}
```

---

## 6. Optimisations Mobile

### Export du modèle

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export pour iOS (CoreML)
model.export(format="coreml", quantize="w8a16")

# Export pour Android (LiteRT)
model.export(format="litert", quantize="w8a32")
```

### Paramètres de performance

| Paramètre | Recommandation | Raison |
|-----------|---------------|--------|
| Résolution entrée | 640×480 | Équilibre précision/vitesse |
| FPS cible | ≥ 15 | Mesure fluide |
| Seuil confiance | 0.5 | Éviter les faux positifs |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

---

## 7. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Distances incorrectes | Objet de référence manquant | Ajouter l'objet dans `referenceObjects` |
| Distances instables | Détection qui osciller | Ajouter un filtre de lissage (moyenne mobile) |
| AR non aligné | Paramètres caméra incorrects | Recalibrer les intrinsèques |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Profondeur imprécise | Approximation simplifiée | Utiliser un modèle de profondeur dédié |
| Dimensions réelles erronées | Référence de taille incorrecte | Vérifier les dimensions réelles des objets |

### Commandes de debug

```dart
if (kDebugMode) {
  for (final det in _detections) {
    final dist = _distanceCalculator.calculateDistance(det);
    debugPrint('${det.label}: ${dist?.toStringAsFixed(2) ?? "N/A"}m');
  }
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Gestion de parking (similaire)](./parking-management.md)
