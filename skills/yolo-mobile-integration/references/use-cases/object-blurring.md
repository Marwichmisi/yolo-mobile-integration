# Référence Complète : Floutage d'Objets avec YOLO26

> Guide complet pour l'intégration d'un système de floutage d'objets dans une application mobile Flutter utilisant YOLO26 pour la génération de masques, l'application de flou gaussien et la protection de la vie privée en temps réel.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application de **protection de la vie privée**
- Le projet nécessite le **floutage de personnes** ou d'objets sensibles
- Il faut de la **redaction en temps réel** dans le flux vidéo
- L'application doit générer des **masques à partir des détections**
- Le modèle **YOLO26** est utilisé pour la détection + floutage
- L'utilisateur mentionne : floutage, masquage, redaction, vie privée, RGPD, anonymisation, flou

**Ne pas utiliser cette référence pour :**
- Sécurité/alarme (utiliser `security-alarm.md`)
- Analyse de heatmap (utiliser `heatmap-analytics.md`)
- Gestion de parking (utiliser `parking-management.md`)

---

## Table des matières

1. [Architecture du Floutage](#1-architecture-du-floutage)
2. [Génération de Masques à partir des Détections](#2-génération-de-masques-à-partir-des-détections)
3. [Application du Flou Gaussien](#3-application-du-flou-gaussien)
4. [Redaction en Temps Réel](#4-redaction-en-temps-réel)
5. [Patterns de Protection de la Vie Privée](#5-patterns-de-protection-de-la-vie-privée)
6. [Exemple Complet Flutter](#6-exemple-complet-flutter)
7. [Optimisations Mobile](#7-optimisations-mobile)
8. [Dépannage](#8-dépannage)

---

## 1. Architecture du Floutage

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                   Application Flutter                     │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Camera  │→  │  YOLO26  │→ │  Mask Generator      │ │
│  │  Stream  │   │  Detect  │   │  (from detections)    │ │
│  └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                           │              │
│  ┌──────────┐   ┌──────────┐   ┌──────────▼───────────┐ │
│  │   UI     │←  │ Blur     │←  │  Blur Engine         │ │
│  │  Preview │   │  Renderer │   │  (gaussian blur)     │ │
│  └──────────┘   └──────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Temps réel, CPU | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

### Classes COCO à flouter

| ID | Nom | Raison floutage |
|----|-----|-----------------|
| 0 | `person` | **Visage / personnes** — protection RGPD |
| 1 | `bicycle` | Plaques d'immatriculation potentielles |
| 2 | `car` | Plaques d'immatriculation |
| 3 | `motorcycle` | Plaques d'immatriculation |
| 5 | `bus` | Plaques + passagers |
| 7 | `truck` | Plaques d'immatriculation |

---

## 2. Génération de Masques à partir des Détections

### 2.1 Principe de génération de masque

Le masque est créé à partir des boîtes de détection. Seules les zones détectées sont floutées, le reste de l'image reste net.

```
Image originale:                    Masque généré:
┌─────────────────────┐             ┌─────────────────────┐
│                     │             │█████████████████████│
│   ┌───┐             │             │█████████████████████│
│   │   │  Personne   │      →      │█████████████████████│
│   └───┘             │             │█████████████████████│
│                     │             │█████████████████████│
│         ┌───────┐   │             │█████████████████████│
│         │       │   │             │█████████████████████│
│         └───────┘   │             │█████████████████████│
└─────────────────────┘             └─────────────────────┘
```

### 2.2 Générateur de masque

```dart
import 'dart:math' as math;

/// Type de masque
enum MaskType {
  boundingBox,    // Boîte de détection complète
  ellipse,        // Ellipse ajustée à la boîte
  tightBox,       // Boîte resserrée (80% de la détection)
  custom,         // Masque personnalisé
}

/// Masque de floutage
class BlurMask {
  final String detectionLabel;
  final double x;
  final double y;
  final double width;
  final double height;
  final MaskType type;
  final double padding; // Padding autour de la boîte

  const BlurMask({
    required this.detectionLabel,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    this.type = MaskType.boundingBox,
    this.padding = 0.0,
  });

  /// Crée un masque à partir d'une détection
  factory BlurMask.fromDetection(
    Detection detection, {
    MaskType type = MaskType.boundingBox,
    double padding = 0.0,
  }) {
    return BlurMask(
      detectionLabel: detection.label,
      x: detection.x - padding,
      y: detection.y - padding,
      width: detection.width + padding * 2,
      height: detection.height + padding * 2,
      type: type,
      padding: padding,
    );
  }

  /// Récupère le rectangle du masque (coordonnées normalisées)
  Rect get rect => Rect.fromLTWH(x, y, width, height);

  /// Récupère le rectangle ajusté selon le type de masque
  Rect getAdjustedRect() {
    switch (type) {
      case MaskType.boundingBox:
        return rect;
      case MaskType.ellipse:
        return rect;
      case MaskType.tightBox:
        final shrinkX = width * 0.1;
        final shrinkY = height * 0.1;
        return Rect.fromLTWH(
          x + shrinkX,
          y + shrinkY,
          width - shrinkX * 2,
          height - shrinkY * 2,
        );
      case MaskType.custom:
        return rect;
    }
  }
}

/// Générateur de masques
class MaskGenerator {
  final List<String> targetClasses;
  final MaskType defaultType;
  final double defaultPadding;

  MaskGenerator({
    required this.targetClasses,
    this.defaultType = MaskType.boundingBox,
    this.defaultPadding = 0.02,
  });

  /// Génère les masques à partir des détections
  List<BlurMask> generateMasks(List<Detection> detections) {
    final masks = <BlurMask>[];

    for (final det in detections) {
      if (!targetClasses.contains(det.label)) continue;

      // Adapter le type de masque selon la classe
      final maskType = _getMaskTypeForClass(det.label);
      final padding = _getPaddingForClass(det.label);

      masks.add(BlurMask.fromDetection(
        det,
        type: maskType,
        padding: padding,
      ));
    }

    return masks;
  }

  /// Détermine le type de masque selon la classe
  MaskType _getMaskTypeForClass(String label) {
    switch (label) {
      case 'person':
        return MaskType.ellipse; // Ellipse pour les personnes (plus naturel)
      case 'car':
      case 'truck':
      case 'motorcycle':
      case 'bus':
        return MaskType.boundingBox; // Boîte complète pour les véhicules
      default:
        return defaultType;
    }
  }

  /// Détermine le padding selon la classe
  double _getPaddingForClass(String label) {
    switch (label) {
      case 'person':
        return 0.05; // Plus de padding pour inclure le visage
      case 'car':
      return 0.03;
      default:
        return defaultPadding;
    }
  }
}
```

---

## 3. Application du Flou Gaussien

### 3.1 Moteur de floutage

```dart
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

/// Moteur de floutage
class BlurEngine {
  final double blurRadius;
  final bool useGaussian;

  BlurEngine({
    this.blurRadius = 15.0,
    this.useGaussian = true,
  });

  /// Applique le flou sur une zone spécifique de l'image
  Future<ui.Image> applyBlur(
    ui.Image sourceImage,
    List<BlurMask> masks,
  ) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Dessiner l'image originale
    canvas.drawImage(sourceImage, Offset.zero, Paint());

    // Appliquer le flou pour chaque masque
    for (final mask in masks) {
      await _applyBlurToMask(canvas, sourceImage, mask);
    }

    final picture = recorder.endRecording();
    return picture.toImage(sourceImage.width, sourceImage.height);
  }

  /// Applique le flou sur un masque spécifique
  Future<void> _applyBlurToMask(
    Canvas canvas,
    ui.Image sourceImage,
    BlurMask mask,
  ) async {
    final rect = mask.getAdjustedRect();

    // Convertir en coordonnées pixels
    final pixelRect = Rect.fromLTWH(
      rect.left * sourceImage.width,
      rect.top * sourceImage.height,
      rect.width * sourceImage.width,
      rect.height * sourceImage.height,
    );

    // Créer un canvas temporaire pour la zone à flouter
    final recorder = ui.PictureRecorder();
    final maskCanvas = Canvas(recorder);

    // Dessiner la zone de l'image
    maskCanvas.drawImageRect(
      sourceImage,
      pixelRect,
      Rect.fromLTWH(0, 0, pixelRect.width, pixelRect.height),
      Paint(),
    );

    // Appliquer le flou
    final blurPaint = Paint()
      ..imageFilter = ui.ImageFilter.blur(
        sigmaX: blurRadius,
        sigmaY: blurRadius,
      );

    // Dessiner la zone floutée sur le canvas principal
    canvas.save();
    canvas.clipRect(pixelRect);
    canvas.drawRect(pixelRect, blurPaint);
    canvas.restore();
  }

  /// Applique le flou sur toute l'image (mode maximal)
  Future<ui.Image> applyFullBlur(ui.Image sourceImage) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    final blurPaint = Paint()
      ..imageFilter = ui.ImageFilter.blur(
        sigmaX: blurRadius * 2,
        sigmaY: blurRadius * 2,
      );

    canvas.drawRect(
      Rect.fromLTWH(0, 0, sourceImage.width.toDouble(), sourceImage.height.toDouble()),
      blurPaint,
    );

    final picture = recorder.endRecording();
    return picture.toImage(sourceImage.width, sourceImage.height);
  }
}
```

### 3.2 Floutage avec CustomPainter (temps réel)

```dart
/// Peintre de floutage en temps réel
class BlurOverlayPainter extends CustomPainter {
  final List<BlurMask> masks;
  final double blurRadius;
  final ui.Image? blurredImage;

  BlurOverlayPainter({
    required this.masks,
    this.blurRadius = 15.0,
    this.blurredImage,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Dessiner l'image floutée en arrière-plan
    if (blurredImage != null) {
      canvas.drawImageRect(
        blurredImage!,
        Rect.fromLTWH(0, 0, blurredImage!.width.toDouble(), blurredImage!.height.toDouble()),
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint(),
      );
    }

    // Dessiner les zones non floutées (trous dans le masque)
    for (final mask in masks) {
      _drawUnblurredRegion(canvas, size, mask);
    }
  }

  void _drawUnblurredRegion(Canvas canvas, Size size, BlurMask mask) {
    final rect = Rect.fromLTWH(
      mask.x * size.width,
      mask.y * size.height,
      mask.width * size.width,
      mask.height * size.height,
    );

    // Créer un path avec un trou
    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)))
      ..fillType = PathFillType.evenOdd;

    // Dessiner le trou (zone non floutée)
    canvas.drawPath(
      path,
      Paint()
        ..color = Colors.transparent
        ..blendMode = BlendMode.clear,
    );
  }

  @override
  bool shouldRepaint(covariant BlurOverlayPainter oldDelegate) => true;
}
```

---

## 4. Redaction en Temps Réel

### 4.1 Pipeline de redaction

```dart
/// Pipeline de redaction en temps réel
class RedactionPipeline {
  final MaskGenerator maskGenerator;
  final BlurEngine blurEngine;
  final List<String> targetClasses;

  RedactionPipeline({
    required this.maskGenerator,
    required this.blurEngine,
    required this.targetClasses,
  });

  /// Traite une image et retourne l'image redactionnée
  Future<ui.Image> processFrame(ui.Image sourceImage) async {
    // 1. Détecter les objets (via YOLO26)
    // Note: La détection est faite par le service YOLO26

    // 2. Générer les masques
    // final masks = maskGenerator.generateMasks(detections);

    // 3. Appliquer le flou
    // final blurredImage = await blurEngine.applyBlur(sourceImage, masks);

    return sourceImage;
  }

  /// Traite une frame avec les détections
  Future<ui.Image> processFrameWithDetections(
    ui.Image sourceImage,
    List<Detection> detections,
  ) async {
    // 1. Générer les masques
    final masks = maskGenerator.generateMasks(detections);

    // 2. Appliquer le flou
    final blurredImage = await blurEngine.applyBlur(sourceImage, masks);

    return blurredImage;
  }
}
```

### 4.2 Widget de preview avec redaction

```dart
/// Widget de preview avec redaction en temps réel
class RedactionPreview extends StatefulWidget {
  final CameraController cameraController;
  final List<Detection> detections;
  final List<String> targetClasses;
  final double blurRadius;

  const RedactionPreview({
    super.key,
    required this.cameraController,
    required this.detections,
    required this.targetClasses,
    this.blurRadius = 15.0,
  });

  @override
  State<RedactionPreview> createState() => _RedactionPreviewState();
}

class _RedactionPreviewState extends State<RedactionPreview> {
  final MaskGenerator _maskGenerator = MaskGenerator(
    targetClasses: ['person', 'car', 'truck', 'motorcycle', 'bus'],
  );
  final BlurEngine _blurEngine = BlurEngine(blurRadius: 15.0);

  @override
  Widget build(BuildContext context) {
    return CameraPreview(
      widget.cameraController,
      child: Stack(
        children: [
          // Overlay de floutage
          CustomPaint(
            painter: BlurOverlayPainter(
              masks: _maskGenerator.generateMasks(widget.detections),
              blurRadius: widget.blurRadius,
            ),
            size: Size.infinite,
          ),
        ],
      ),
    );
  }
}
```

---

## 5. Patterns de Protection de la Vie Privée

### 5.1 Patterns courants

```dart
/// Patterns de protection de la vie privée
class PrivacyPatterns {
  /// Flouter uniquement les visages (personnes)
  static MaskGenerator faceOnly() => MaskGenerator(
        targetClasses: ['person'],
        defaultType: MaskType.tightBox,
        defaultPadding: 0.0,
      );

  /// Flouter les plaques d'immatriculation (véhicules)
  static MaskGenerator licensePlates() => MaskGenerator(
        targetClasses: ['car', 'truck', 'motorcycle', 'bus'],
        defaultType: MaskType.tightBox,
        defaultPadding: 0.01,
      );

  /// Flouter toutes les personnes et véhicules
  static MaskGenerator allPeopleAndVehicles() => MaskGenerator(
        targetClasses: ['person', 'car', 'truck', 'motorcycle', 'bus', 'bicycle'],
        defaultType: MaskType.boundingBox,
        defaultPadding: 0.02,
      );

  /// Flouter uniquement les personnes avec ellipse
  static MaskGenerator peopleEllipse() => MaskGenerator(
        targetClasses: ['person'],
        defaultType: MaskType.ellipse,
        defaultPadding: 0.05,
      );

  /// Flouter tout (mode maximal)
  static MaskGenerator everything() => MaskGenerator(
        targetClasses: [
          'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        ],
        defaultType: MaskType.boundingBox,
        defaultPadding: 0.0,
      );
}
```

### 5.2 Configuration RGPD

```dart
/// Configuration RGPD pour la protection des données
class RGPDConfig {
  /// Classes à flouter par défaut (conformité RGPD)
  static const List<String> defaultSensitiveClasses = [
    'person',  // Visage = donnée personnelle
  ];

  /// Classes à flouter pour les véhicules (plaque d'immatriculation)
  static const List<String> vehicleClasses = [
    'car', 'truck', 'motorcycle', 'bus', 'bicycle',
  ];

  /// Rayon de flou par défaut
  static const double defaultBlurRadius = 15.0;

  /// Rayon de flou maximal (pour les données sensibles)
  static const double maxBlurRadius = 30.0;

  /// Padding par défaut autour des détections
  static const double defaultPadding = 0.02;

  /// Vérifier si une classe est sensible
  static bool isSensitive(String label) {
    return defaultSensitiveClasses.contains(label) ||
        vehicleClasses.contains(label);
  }

  /// Obtenir le rayon de flou approprié
  static double getBlurRadius(String label) {
    if (defaultSensitiveClasses.contains(label)) {
      return maxBlurRadius;
    }
    return defaultBlurRadius;
  }
}
```

---

## 6. Exemple Complet Flutter

### Structure du projet

```
lib/
├── main.dart
├── models/
│   ├── detection.dart
│   └── blur_mask.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── mask_generator.dart
│   └── blur_engine.dart
├── screens/
│   └── object_blurring_screen.dart
└── widgets/
    ├── blur_overlay_painter.dart
    └── privacy_settings_widget.dart
```

### `screens/object_blurring_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../services/yolo_detection_service.dart';
import '../services/mask_generator.dart';
import '../services/blur_engine.dart';
import '../widgets/blur_overlay_painter.dart';
import '../widgets/privacy_settings_widget.dart';

class ObjectBlurringScreen extends StatefulWidget {
  const ObjectBlurringScreen({super.key});

  @override
  State<ObjectBlurringScreen> createState() => _ObjectBlurringScreenState();
}

class _ObjectBlurringScreenState extends State<ObjectBlurringScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  late final MaskGenerator _maskGenerator;
  final BlurEngine _blurEngine = BlurEngine(blurRadius: 15.0);

  List<Detection> _detections = [];
  List<BlurMask> _masks = [];
  bool _isProcessing = false;
  bool _isInitialized = false;
  bool _isRedactionEnabled = true;

  // Classes à flouter
  final Set<String> _targetClasses = {
    'person', 'car', 'truck', 'motorcycle', 'bus'
  };

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _yoloService.initialize();

    _maskGenerator = MaskGenerator(
      targetClasses: _targetClasses.toList(),
      defaultType: MaskType.boundingBox,
      defaultPadding: 0.02,
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

      // Générer les masques si la redaction est activée
      final masks = _isRedactionEnabled
          ? _maskGenerator.generateMasks(detections)
          : <BlurMask>[];

      setState(() {
        _detections = detections;
        _masks = masks;
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

  void _toggleRedaction() {
    setState(() => _isRedactionEnabled = !_isRedactionEnabled);
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      builder: (context) => PrivacySettingsWidget(
        targetClasses: _targetClasses,
        onClassesChanged: (classes) {
          setState(() {
            _targetClasses.clear();
            _targetClasses.addAll(classes);
            _maskGenerator = MaskGenerator(
              targetClasses: _targetClasses.toList(),
              defaultType: MaskType.boundingBox,
              defaultPadding: 0.02,
            );
          });
        },
      ),
    );
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
        title: const Text('Floutage d\'objets'),
        actions: [
          // Bouton settings
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettings,
          ),
          // Toggle redaction
          IconButton(
            icon: Icon(
              _isRedactionEnabled ? Icons.blur_on : Icons.blur_off,
              color: _isRedactionEnabled ? Colors.green : Colors.red,
            ),
            onPressed: _toggleRedaction,
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Caméra avec overlay de floutage
          CameraPreview(_cameraController!),
          if (_isRedactionEnabled)
            CustomPaint(
              painter: BlurOverlayPainter(
                masks: _masks,
                blurRadius: 15.0,
              ),
              size: Size.infinite,
            ),

          // Indicateur de redaction
          if (_isRedactionEnabled)
            Positioned(
              top: 16,
              right: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(0.8),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.blur_on, color: Colors.white, size: 16),
                    SizedBox(width: 4),
                    Text(
                      'Redaction active',
                      style: TextStyle(color: Colors.white, fontSize: 12),
                    ),
                  ],
                ),
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
    super.dispose();
  }
}
```

### `widgets/privacy_settings_widget.dart`

```dart
import 'package:flutter/material.dart';

class PrivacySettingsWidget extends StatefulWidget {
  final Set<String> targetClasses;
  final ValueChanged<Set<String>> onClassesChanged;

  const PrivacySettingsWidget({
    super.key,
    required this.targetClasses,
    required this.onClassesChanged,
  });

  @override
  State<PrivacySettingsWidget> createState() => _PrivacySettingsWidgetState();
}

class _PrivacySettingsWidgetState extends State<PrivacySettingsWidget> {
  late Set<String> _selectedClasses;

  static const Map<String, String> _classLabels = {
    'person': 'Personnes',
    'car': 'Voitures',
    'truck': 'Camions',
    'motorcycle': 'Motos',
    'bus': 'Bus',
    'bicycle': 'Vélos',
  };

  @override
  void initState() {
    super.initState();
    _selectedClasses = Set.from(widget.targetClasses);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Classes à flouter',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          ..._classLabels.entries.map((entry) {
            return CheckboxListTile(
              title: Text(entry.value),
              value: _selectedClasses.contains(entry.key),
              onChanged: (value) {
                setState(() {
                  if (value == true) {
                    _selectedClasses.add(entry.key);
                  } else {
                    _selectedClasses.remove(entry.key);
                  }
                });
                widget.onClassesChanged(_selectedClasses);
              },
            );
          }),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Fermer'),
            ),
          ),
        ],
      ),
    );
  }
}
```

---

## 7. Optimisations Mobile

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
| FPS cible | ≥ 15 | Redaction fluide |
| Rayon de flou | 10-20 | Bon équilibre visuel/performance |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

### Gestion de la mémoire

```dart
class BlurMemoryManager {
  static const int maxMasksPerFrame = 20;
  static const int maxBlurRadius = 50;

  /// Limite le nombre de masques par frame
  static List<BlurMask> limitMasks(List<BlurMask> masks) {
    if (masks.length > maxMasksPerFrame) {
      return masks.sublist(0, maxMasksPerFrame);
    }
    return masks;
  }

  /// Limite le rayon de flou
  static double clampBlurRadius(double radius) {
    return radius.clamp(5.0, maxBlurRadius.toDouble());
  }
}
```

---

## 8. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Flou non visible | Rayon trop faible | Augmenter `blurRadius` à 15+ |
| Zones floutées trop petites | Padding insuffisant | Augmenter `defaultPadding` |
| Zones floutées trop grandes | Padding excessif | Réduire `defaultPadding` |
| Performance dégradée | Trop de masques | Limiter à 20 masques par frame |
| Flou non uniforme | Gaussian non appliqué | Vérifier `useGaussian = true` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Masques mal alignés | Coordonnées incorrectes | Vérifier la normalisation des coordonnées |
| RGPD non conforme | Classes sensibles non floutées | Ajouter 'person' dans `targetClasses` |

### Commandes de debug

```dart
if (kDebugMode) {
  debugPrint('Détections: ${_detections.length}');
  debugPrint('Masques: ${_masks.length}');
  debugPrint('Classes cibles: $_targetClasses');
  debugPrint('Redaction: $_isRedactionEnabled');
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Analyse de heatmap (similaire)](./heatmap-analytics.md)
