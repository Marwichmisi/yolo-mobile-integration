# Référence Complète : Comptage d'Objets avec YOLO26

> Guide complet pour l'intégration d'un système de comptage d'objets dans une application mobile Flutter utilisant YOLO26 avec suivi de centroïde, détection de franchissement de ligne et comptage bidirectionnel.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer un système de **comptage d'objets** (personnes, véhicules, etc.)
- Le projet nécessite le **suivi d'objets** à travers les frames
- Il faut détecter le **franchissement de ligne** (entrée/sortie)
- L'application doit compter en **bidirectionnel** (IN/OUT)
- Le modèle **YOLO26** est utilisé pour la détection + tracking
- L'utilisateur mentionne : comptage, compteur, entrée/sortie, IN/OUT, passage, tracking, suivi

**Ne pas utiliser cette référence pour :**
- Sécurité/alarme (utiliser `security-alarm.md`)
- Gestion de parking (utiliser `parking-management.md`)
- Analyse de heatmap (utiliser `heatmap-analytics.md`)

---

## Table des matières

1. [Architecture du Comptage d'Objets](#1-architecture-du-comptage-dobjets)
2. [Algorithme de Suivi de Centroïde](#2-algorithme-de-suivi-de-centroïde)
3. [Détection de Franchissement de Ligne](#3-détection-de-franchissement-de-ligne)
4. [Comptage Bidirectionnel IN/OUT](#4-comptage-bidirectionnel-inout)
5. [Support des Zones Polygonales](#5-support-des-zones-polygonales)
6. [Exemple Complet Flutter](#6-exemple-complet-flutter)
7. [Optimisations Mobile](#7-optimisations-mobile)
8. [Dépannage](#8-dépannage)

---

## 1. Architecture du Comptage d'Objets

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────┐
│                  Application Flutter                  │
│                                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  Camera  │→  │  YOLO26  │→ │  Tracker         │ │
│  │  Stream  │   │  Detect  │   │  (centroid)      │ │
│  └──────────┘   └──────────┘   └────────┬─────────┘ │
│                                         │            │
│  ┌──────────┐   ┌──────────┐   ┌───────▼─────────┐ │
│  │   UI     │←  │ Counter  │←  │  Line Crossing   │ │
│  │   IN/OUT │   │  State   │   │  Detector        │ │
│  └──────────┘   └──────────┘   └─────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Temps réel, CPU | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

### Classes COCO pour comptage

| ID | Nom | Usage |
|----|-----|-------|
| 0 | `person` | Comptage de personnes |
| 1 | `bicycle` | Comptage de vélos |
| 2 | `car` | Comptage de voitures |
| 3 | `motorcycle` | Comptage de motos |
| 5 | `bus` | Comptage de bus |
| 7 | `truck` | Comptage de camions |

---

## 2. Algorithme de Suivi de Centroïde

### 2.1 Principe du suivi par centroïde

Le suivi de centroïde associe les détections entre frames consécutives en calculant la distance euclidienne entre les centroïdes des boîtes de détection.

```
Frame N:                    Frame N+1:
┌─────────┐                 ┌─────────┐
│  ● A    │  ─── associer ─→ │  ● A'   │
│  ● B    │  ─── associer ─→ │  ● B'   │
│  ● C    │  ─── associer ─→ │  ● C'   │
└─────────┘                 └─────────┘

Distance(A, A') < Distance(A, B') → A associé à A'
```

### 2.2 Implémentation du tracker

```dart
import 'dart:math';

/// Objet suivi
class TrackedObject {
  final int id;
  double centroidX;
  double centroidY;
  double width;
  double height;
  String label;
  double confidence;
  int age;                    // Nombre de frames depuis la première détection
  int invisibleCount;         // Nombre de frames sans détection
  List<Offset> trajectory;    // Historique des positions
  bool counted;               // Si l'objet a déjà été compté
  String? direction;          // 'IN' ou 'OUT' si compté

  TrackedObject({
    required this.id,
    required this.centroidX,
    required this.centroidY,
    required this.width,
    required this.height,
    required this.label,
    required this.confidence,
    this.age = 0,
    this.invisibleCount = 0,
    this.counted = false,
    this.direction,
  }) : trajectory = [Offset(centroidX, centroidY)];

  Offset get centroid => Offset(centroidX, centroidY);

  void update({
    required double centroidX,
    required double centroidY,
    required double width,
    required double height,
    required String label,
    required double confidence,
  }) {
    this.centroidX = centroidX;
    this.centroidY = centroidY;
    this.width = width;
    this.height = height;
    this.label = label;
    this.confidence = confidence;
    this.age++;
    this.invisibleCount = 0;
    trajectory.add(Offset(centroidX, centroidY));

    // Limiter la trajectoire
    if (trajectory.length > 30) {
      trajectory.removeAt(0);
    }
  }

  void markInvisible() {
    invisibleCount++;
  }
}

/// Tracker de centroïde
class CentroidTracker {
  final Map<int, TrackedObject> _objects = {};
  int _nextId = 0;
  final double maxDistance;       // Distance max d'association (normalisée)
  final int maxInvisible;         // Frames max avant suppression
  final int minHits;              // Détections min avant de valider

  CentroidTracker({
    this.maxDistance = 0.15,     // 15% de la dimension de l'image
    this.maxInvisible = 10,       // 10 frames sans détection
    this.minHits = 3,             // 3 détections pour valider
  });

  Map<int, TrackedObject> get objects => Map.unmodifiable(_objects);

  /// Met à jour le tracker avec les nouvelles détections
  List<TrackedObject> update(List<Detection> detections) {
    // 1. Calculer les centroïdes des nouvelles détections
    final detectionsData = detections.map((d) => {
          'centroidX': d.centerX,
          'centroidY': d.centerY,
          'width': d.width,
          'height': d.height,
          'label': d.label,
          'confidence': d.confidence,
        }).toList();

    // 2. Calculer la matrice de distances (détections × objets existants)
    final costMatrix = _computeCostMatrix(detectionsData);

    // 3. Associer par hongrois (ou glouton simplifié)
    final assignments = _hungarianAssignment(costMatrix);

    // 4. Mettre à jour les objets associés
    final updatedObjects = <TrackedObject>{};
    final usedDetections = <int>{};
    final usedObjectIds = <int>{};

    for (final assignment in assignments) {
      final detIdx = assignment['detection'] as int;
      final objId = assignment['object'] as int;
      final distance = assignment['distance'] as double;

      if (distance > maxDistance) continue;

      final det = detectionsData[detIdx];
      final obj = _objects[objId]!;

      obj.update(
        centroidX: det['centroidX'] as double,
        centroidY: det['centroidY'] as double,
        width: det['width'] as double,
        height: det['height'] as double,
        label: det['label'] as String,
        confidence: det['confidence'] as double,
      );

      updatedObjects.add(obj);
      usedDetections.add(detIdx);
      usedObjectIds.add(objId);
    }

    // 5. Créer les nouveaux objets pour les détections non associées
    for (int i = 0; i < detectionsData.length; i++) {
      if (usedDetections.contains(i)) continue;

      final det = detectionsData[i];
      final newObj = TrackedObject(
        id: _nextId++,
        centroidX: det['centroidX'] as double,
        centroidY: det['centroidY'] as double,
        width: det['width'] as double,
        height: det['height'] as double,
        label: det['label'] as String,
        confidence: det['confidence'] as double,
      );
      _objects[newObj.id] = newObj;
      updatedObjects.add(newObj);
    }

    // 6. Marquer les objets non détectés comme invisibles
    for (final entry in _objects.entries) {
      if (!usedObjectIds.contains(entry.key)) {
        entry.value.markInvisible();
      }
    }

    // 7. Supprimer les objets invisibles trop longtemps
    _objects.removeWhere((id, obj) => obj.invisibleCount > maxInvisible);

    return updatedObjects.toList();
  }

  /// Calcule la matrice de coûts (distances euclidiennes)
  List<List<double>> _computeCostMatrix(List<Map<String, dynamic>> detections) {
    final matrix = <List<double>>[];

    for (final det in detections) {
      final row = <double>[];
      for (final obj in _objects.values) {
        final dx = (det['centroidX'] as double) - obj.centroidX;
        final dy = (det['centroidY'] as double) - obj.centroidY;
        final distance = sqrt(dx * dx + dy * dy);
        row.add(distance);
      }
      matrix.add(row);
    }

    return matrix;
  }

  /// Algorithme d'association hongrois simplifié (glouton)
  List<Map<String, dynamic>> _hungarianAssignment(List<List<double>> costMatrix) {
    final assignments = <Map<String, dynamic>>[];

    if (costMatrix.isEmpty || costMatrix[0].isEmpty) return assignments;

    // Pour chaque détection, trouver l'objet le plus proche
    for (int i = 0; i < costMatrix.length; i++) {
      double minDist = double.infinity;
      int minObjIdx = -1;

      for (int j = 0; j < costMatrix[i].length; j++) {
        if (costMatrix[i][j] < minDist) {
          minDist = costMatrix[i][j];
          minObjIdx = j;
        }
      }

      if (minObjIdx >= 0) {
        final objId = _objects.keys.elementAt(minObjIdx);
        assignments.add({
          'detection': i,
          'object': objId,
          'distance': minDist,
        });
      }
    }

    // Résoudre les conflits (un objet assigné à plusieurs détections)
    final objAssigned = <int>{};
    final resolved = <Map<String, dynamic>>[];

    // Trier par distance (les plus proches d'abord)
    assignments.sort((a, b) => (a['distance'] as double).compareTo(b['distance'] as double));

    for (final a in assignments) {
      final objId = a['object'] as int;
      if (!objAssigned.contains(objId)) {
        resolved.add(a);
        objAssigned.add(objId);
      }
    }

    return resolved;
  }

  void reset() {
    _objects.clear();
    _nextId = 0;
  }
}
```

---

## 3. Détection de Franchissement de Ligne

### 3.1 Principe

Une ligne de comptage est définie par deux points. Un franchissement est détecté lorsque la trajectoire d'un objet passe d'un côté à l'autre de la ligne.

```
Côté A          Ligne          Côté B
   ●──────────────│
                  │
   ●──────────────│
                  │
   ●──────────────│
                  │
   ●──────────────│
```

### 3.2 Implémentation

```dart
/// Ligne de comptage
class CountingLine {
  final String id;
  final Offset startPoint;    // Normalisé (0-1)
  final Offset endPoint;      // Normalisé (0-1)
  final String name;

  const CountingLine({
    required this.id,
    required this.startPoint,
    required this.endPoint,
    required this.name,
  });

  /// Calcule le côté d'un point par rapport à la ligne
  /// Retourne > 0 pour un côté, < 0 pour l'autre, 0 sur la ligne
  double sideOfPoint(Offset point) {
    // Produit vectoriel (2D)
    final dx = endPoint.dx - startPoint.dx;
    final dy = endPoint.dy - startPoint.dy;
    final px = point.dx - startPoint.dx;
    final py = point.dy - startPoint.dy;
    return dx * py - dy * px;
  }

  /// Vérifie si un segment traverse la ligne
  bool isCrossed(Offset from, Offset to) {
    final sideA = sideOfPoint(from);
    final sideB = sideOfPoint(to);

    // Si les deux points sont du même côté, pas de franchissement
    if (sideA * sideB > 0) return false;

    // Si un point est exactement sur la ligne
    if (sideA == 0 || sideB == 0) return true;

    // Les points sont de part et d'autre → franchissement
    return true;
  }

  /// Calcule la direction du franchissement
  /// Retourne 'IN' ou 'OUT' selon la direction de référence
  String getCrossingDirection(Offset from, Offset to) {
    final sideA = sideOfPoint(from);
    final sideB = sideOfPoint(to);

    // Déterminer la direction selon le signe du produit vectoriel
    // Si on passe du côté positif au négatif → IN
    // Si on passe du côté négatif au positif → OUT
    if (sideA > 0 && sideB < 0) return 'IN';
    if (sideA < 0 && sideB > 0) return 'OUT';

    // Cas limite : un point sur la ligne
    if (sideA == 0) return sideB > 0 ? 'IN' : 'OUT';
    if (sideB == 0) return sideA > 0 ? 'OUT' : 'IN';

    return 'UNKNOWN';
  }
}

/// Détecteur de franchissement de ligne
class LineCrossingDetector {
  final List<CountingLine> _lines = [];
  final Map<String, String> _lastSides = {}; // Dernier côté connu par objet

  List<CountingLine> get lines => List.unmodifiable(_lines);

  void addLine(CountingLine line) => _lines.add(line);

  void removeLine(String lineId) {
    _lines.removeWhere((l) => l.id == lineId);
    _lastSides.remove(lineId);
  }

  /// Vérifie les franchissements pour un objet suivi
  /// Retourne les lignes franchies avec leur direction
  List<LineCrossingEvent> checkCrossings(TrackedObject object) {
    final events = <LineCrossingEvent>[];

    for (final line in _lines) {
      if (object.trajectory.length < 2) continue;

      final lastPos = object.trajectory[object.trajectory.length - 2];
      final currentPos = object.trajectory.last;

      if (line.isCrossed(lastPos, currentPos)) {
        final direction = line.getCrossingDirection(lastPos, currentPos);

        events.add(LineCrossingEvent(
          lineId: line.id,
          lineName: line.name,
          objectId: object.id,
          objectLabel: object.label,
          direction: direction,
          timestamp: DateTime.now(),
        ));
      }
    }

    return events;
  }

  void reset() {
    _lines.clear();
    _lastSides.clear();
  }
}

/// Événement de franchissement de ligne
class LineCrossingEvent {
  final String lineId;
  final String lineName;
  final int objectId;
  final String objectLabel;
  final String direction; // 'IN' ou 'OUT'
  final DateTime timestamp;

  const LineCrossingEvent({
    required this.lineId,
    required this.lineName,
    required this.objectId,
    required this.objectLabel,
    required this.direction,
    required this.timestamp,
  });
}
```

---

## 4. Comptage Bidirectionnel IN/OUT

### 4.1 Compteur bidirectionnel

```dart
/// Compteur bidirectionnel
class BidirectionalCounter {
  final String lineId;
  final String lineName;
  int _inCount = 0;
  int _outCount = 0;
  final Map<int, String> _countedObjects = {}; // ID objet → direction

  BidirectionalCounter({
    required this.lineId,
    required this.lineName,
  });

  int get inCount => _inCount;
  int get outCount => _outCount;
  int get totalCount => _inCount - _outCount; // Solde (personnes dans la zone)

  /// Enregistre un franchissement
  void recordCrossing(LineCrossingEvent event) {
    // Éviter le double comptage
    if (_countedObjects.containsKey(event.objectId)) {
      // L'objet a déjà été compté — vérifier si c'est un retour
      final lastDirection = _countedObjects[event.objectId]!;
      if (lastDirection == event.direction) return; // Même direction, ignorer

      // Retour : décrémenter l'ancien compteur
      if (lastDirection == 'IN') {
        _inCount--;
      } else {
        _outCount--;
      }
    }

    // Enregistrer le nouveau franchissement
    if (event.direction == 'IN') {
      _inCount++;
    } else {
      _outCount++;
    }
    _countedObjects[event.objectId] = event.direction;
  }

  /// Réinitialise le compteur
  void reset() {
    _inCount = 0;
    _outCount = 0;
    _countedObjects.clear();
  }

  Map<String, dynamic> toJson() => {
        'lineId': lineId,
        'lineName': lineName,
        'in': _inCount,
        'out': _outCount,
        'total': totalCount,
      };
}
```

### 4.2 Gestionnaire de comptage global

```dart
/// Gestionnaire de comptage global
class CountingManager {
  final CentroidTracker _tracker;
  final LineCrossingDetector _lineDetector;
  final Map<String, BidirectionalCounter> _counters = {};

  CountingManager({
    CentroidTracker? tracker,
    LineCrossingDetector? lineDetector,
  })  : _tracker = tracker ?? CentroidTracker(),
        _lineDetector = lineDetector ?? LineCrossingDetector();

  CentroidTracker get tracker => _tracker;
  LineCrossingDetector get lineDetector => _lineDetector;

  /// Ajoute une ligne de comptage avec son compteur
  void addCountingLine(CountingLine line) {
    _lineDetector.addLine(line);
    _counters[line.id] = BidirectionalCounter(
      lineId: line.id,
      lineName: line.name,
    );
  }

  /// Traite une frame de détections
  List<LineCrossingEvent> processFrame(List<Detection> detections) {
    // 1. Mettre à jour le tracker
    final trackedObjects = _tracker.update(detections);

    // 2. Vérifier les franchissements de ligne
    final events = <LineCrossingEvent>[];
    for (final obj in trackedObjects) {
      final crossings = _lineDetector.checkCrossings(obj);
      for (final crossing in crossings) {
        // Mettre à jour le compteur
        final counter = _counters[crossing.lineId];
        if (counter != null) {
          counter.recordCrossing(crossing);
          obj.counted = true;
          obj.direction = crossing.direction;
        }
      }
      events.addAll(crossings);
    }

    return events;
  }

  /// Récupère un compteur par ID de ligne
  BidirectionalCounter? getCounter(String lineId) => _counters[lineId];

  /// Récupère tous les compteurs
  List<BidirectionalCounter> get allCounters => _counters.values.toList();

  /// Réinitialise tout
  void reset() {
    _tracker.reset();
    _lineDetector.reset();
    for (final counter in _counters.values) {
      counter.reset();
    }
  }
}
```

---

## 5. Support des Zones Polygonales

### 5.1 Zone de comptage polygonale

```dart
/// Zone de comptage polygonale
class CountingZone {
  final String id;
  final String name;
  final List<Offset> polygon; // Points normalisés (0-1)
  final List<String> targetClasses;
  int _count = 0;
  final Set<int> _countedObjectIds = {};

  CountingZone({
    required this.id,
    required this.name,
    required this.polygon,
    required this.targetClasses,
  });

  int get count => _count;

  /// Vérifie si un point est dans le polygone
  bool containsPoint(Offset point) {
    if (polygon.length < 3) return false;

    bool inside = false;
    int j = polygon.length - 1;

    for (int i = 0; i < polygon.length; i++) {
      final pi = polygon[i];
      final pj = polygon[j];

      if (((pi.dy > point.dy) != (pj.dy > point.dy)) &&
          (point.dx < (pj.dx - pi.dx) * (point.dy - pi.dy) / (pj.dy - pi.dy) + pi.dx)) {
        inside = !inside;
      }
      j = i;
    }
    return inside;
  }

  /// Met à jour le comptage avec les objets suivis
  void updateCount(List<TrackedObject> objects) {
    final currentIds = <int>{};

    for (final obj in objects) {
      if (!targetClasses.contains(obj.label)) continue;
      if (!containsPoint(obj.centroid)) continue;

      currentIds.add(obj.id);

      // Nouvel objet dans la zone
      if (!_countedObjectIds.contains(obj.id)) {
        _countedObjectIds.add(obj.id);
        _count++;
      }
    }

    // Supprimer les objets qui ont quitté la zone
    _countedObjectIds.removeWhere((id) {
      if (!currentIds.contains(id)) {
        _count--;
        return true;
      }
      return false;
    });
  }

  void reset() {
    _count = 0;
    _countedObjectIds.clear();
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
│   ├── tracked_object.dart
│   └── counting_line.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── centroid_tracker.dart
│   ├── line_crossing_detector.dart
│   └── counting_manager.dart
├── screens/
│   └── object_counting_screen.dart
└── widgets/
    ├── counting_overlay_painter.dart
    └── counter_display_widget.dart
```

### `screens/object_counting_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../models/counting_line.dart';
import '../services/yolo_detection_service.dart';
import '../services/counting_manager.dart';
import '../widgets/counting_overlay_painter.dart';
import '../widgets/counter_display_widget.dart';

class ObjectCountingScreen extends StatefulWidget {
  const ObjectCountingScreen({super.key});

  @override
  State<ObjectCountingScreen> createState() => _ObjectCountingScreenState();
}

class _ObjectCountingScreenState extends State<ObjectCountingScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  final CountingManager _countingManager = CountingManager();

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

    // Configurer une ligne de comptage horizontale au milieu
    _countingManager.addCountingLine(CountingLine(
      id: 'main_line',
      name: 'Entrée/Sortie',
      startPoint: const Offset(0.0, 0.5),
      endPoint: const Offset(1.0, 0.5),
    ));

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

      // Filtrer pour ne garder que les personnes et véhicules
      final filtered = detections
          .where((d) => ['person', 'car', 'truck', 'motorcycle', 'bus', 'bicycle']
              .contains(d.label))
          .toList();

      // Traiter le comptage
      _countingManager.processFrame(filtered);

      setState(() {
        _detections = filtered;
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
        title: const Text('Comptage d\'objets'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => _countingManager.reset(),
          ),
        ],
      ),
      body: Column(
        children: [
          // Zone caméra avec overlay
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                CustomPaint(
                  painter: CountingOverlayPainter(
                    countingManager: _countingManager,
                    detections: _detections,
                  ),
                ),
              ],
            ),
          ),

          // Affichage des compteurs
          Expanded(
            flex: 1,
            child: CounterDisplayWidget(
              counters: _countingManager.allCounters,
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

### `widgets/counting_overlay_painter.dart`

```dart
import 'package:flutter/material.dart';
import '../models/detection.dart';
import '../services/counting_manager.dart';

class CountingOverlayPainter extends CustomPainter {
  final CountingManager countingManager;
  final List<Detection> detections;

  CountingOverlayPainter({
    required this.countingManager,
    required this.detections,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Dessiner les lignes de comptage
    for (final line in countingManager.lineDetector.lines) {
      _drawCountingLine(canvas, size, line);
    }

    // Dessiner les objets suivis
    for (final obj in countingManager.tracker.objects.values) {
      _drawTrackedObject(canvas, size, obj);
    }

    // Dessiner les détections
    for (final det in detections) {
      _drawDetection(canvas, size, det);
    }
  }

  void _drawCountingLine(Canvas canvas, Size size, CountingLine line) {
    final start = Offset(line.startPoint.dx * size.width, line.startPoint.dy * size.height);
    final end = Offset(line.endPoint.dx * size.width, line.endPoint.dy * size.height);

    // Ligne principale
    canvas.drawLine(
      start,
      end,
      Paint()
        ..color = Colors.yellow
        ..strokeWidth = 3
        ..style = PaintingStyle.stroke,
    );

    // Flèches de direction
    _drawArrow(canvas, start, end, Colors.green.withOpacity(0.5));
    _drawArrow(canvas, end, start, Colors.red.withOpacity(0.5));

    // Nom de la ligne
    final textPainter = TextPainter(
      text: TextSpan(
        text: line.name,
        style: const TextStyle(
          color: Colors.yellow,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black54,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, start + const Offset(5, -20));
  }

  void _drawTrackedObject(Canvas canvas, Size size, TrackedObject obj) {
    final pos = Offset(obj.centroidX * size.width, obj.centroidY * size.height);

    // Point du centroïde
    final color = obj.counted
        ? (obj.direction == 'IN' ? Colors.green : Colors.red)
        : Colors.yellow;

    canvas.drawCircle(
      pos,
      6,
      Paint()..color = color,
    );

    // Trajectoire
    if (obj.trajectory.length > 1) {
      final path = Path();
      path.moveTo(
        obj.trajectory.first.dx * size.width,
        obj.trajectory.first.dy * size.height,
      );
      for (int i = 1; i < obj.trajectory.length; i++) {
        path.lineTo(
          obj.trajectory[i].dx * size.width,
          obj.trajectory[i].dy * size.height,
        );
      }
      canvas.drawPath(
        path,
        Paint()
          ..color = color.withOpacity(0.4)
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke,
      );
    }

    // ID de l'objet
    final textPainter = TextPainter(
      text: TextSpan(
        text: '#${obj.id}${obj.counted ? " ${obj.direction}" : ""}',
        style: TextStyle(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, pos + const Offset(8, -8));
  }

  void _drawDetection(Canvas canvas, Size size, Detection det) {
    final rect = Rect.fromLTWH(
      det.x * size.width,
      det.y * size.height,
      det.width * size.width,
      det.height * size.height,
    );

    canvas.drawRect(
      rect,
      Paint()
        ..color = Colors.cyan.withOpacity(0.5)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1,
    );
  }

  void _drawArrow(Canvas canvas, Offset from, Offset to, Color color) {
    final direction = (to - from).direction;
    const arrowSize = 10.0;

    final path = Path();
    path.moveTo(to.dx, to.dy);
    path.lineTo(
      to.dx - arrowSize * cos(direction - 0.5),
      to.dy - arrowSize * sin(direction - 0.5),
    );
    path.moveTo(to.dx, to.dy);
    path.lineTo(
      to.dx - arrowSize * cos(direction + 0.5),
      to.dy - arrowSize * sin(direction + 0.5),
    );

    canvas.drawPath(path, Paint()..color = color..strokeWidth = 2);
  }

  @override
  bool shouldRepaint(covariant CountingOverlayPainter oldDelegate) => true;
}
```

### `widgets/counter_display_widget.dart`

```dart
import 'package:flutter/material.dart';
import '../services/counting_manager.dart';

class CounterDisplayWidget extends StatelessWidget {
  final List<BidirectionalCounter> counters;

  const CounterDisplayWidget({
    super.key,
    required this.counters,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: counters.map((counter) {
          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                counter.lineName,
                style: Theme.of(context).textTheme.titleSmall,
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _CounterBadge(
                    label: 'IN',
                    count: counter.inCount,
                    color: Colors.green,
                  ),
                  const SizedBox(width: 16),
                  _CounterBadge(
                    label: 'OUT',
                    count: counter.outCount,
                    color: Colors.red,
                  ),
                  const SizedBox(width: 16),
                  _CounterBadge(
                    label: 'TOTAL',
                    count: counter.totalCount,
                    color: Colors.blue,
                  ),
                ],
              ),
            ],
          );
        }).toList(),
      ),
    );
  }
}

class _CounterBadge extends StatelessWidget {
  final String label;
  final int count;
  final Color color;

  const _CounterBadge({
    required this.label,
    required this.count,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: color.withOpacity(0.2),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: color),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
              fontSize: 12,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          '$count',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                color: color,
                fontWeight: FontWeight.bold,
              ),
        ),
      ],
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
| FPS cible | ≥ 15 | Suivi fluide |
| Distance max association | 0.15 (normalisé) | Éviter les sauts d'ID |
| Frames max invisibles | 10 | Éviter les objets fantômes |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

### Gestion de la mémoire

```dart
class CountingMemoryManager {
  static const int maxTrajectoryLength = 30;
  static const int maxTrackedObjects = 100;
  static const int maxCounters = 10;

  /// Limite la trajectoire des objets
  static void limitTrajectory(TrackedObject obj) {
    if (obj.trajectory.length > maxTrajectoryLength) {
      obj.trajectory.removeRange(0, obj.trajectory.length - maxTrajectoryLength);
    }
  }

  /// Nettoie les objets inactifs
  static void cleanupInactiveObjects(Map<int, TrackedObject> objects) {
    objects.removeWhere((id, obj) => obj.invisibleCount > 30);
  }
}
```

---

## 8. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Objets non suivis | Distance max trop faible | Augmenter `maxDistance` à 0.2+ |
| IDs qui changent | Association incorrecte | Réduire `maxDistance` ou augmenter `minHits` |
| Double comptage | Objet qui traverse la ligne plusieurs fois | Vérifier la logique `_countedObjects` |
| Comptage inversé | Direction de référence incorrecte | Inverser la logique `getCrossingDirection` |
| Objets fantômes | `maxInvisible` trop élevé | Réduire à 5-8 frames |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Perte de suivi aux chevauchements | Centroïdes trop proches | Utiliser l'IoU en plus de la distance |
| Comptage négatif | Retour d'objet non géré | Vérifier la logique de décrémentation |

### Commandes de debug

```dart
if (kDebugMode) {
  debugPrint('Objets suivis: ${_countingManager.tracker.objects.length}');
  debugPrint('Détections: ${_detections.length}');
  for (final counter in _countingManager.allCounters) {
    debugPrint('${counter.lineName}: IN=${counter.inCount} OUT=${counter.outCount}');
  }
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Alarme de sécurité (similaire)](./security-alarm.md)
