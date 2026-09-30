# Référence Complète : Gestion de Parking avec YOLO26

> Guide complet pour l'intégration d'un système de gestion de parking dans une application mobile Flutter utilisant YOLO26 pour la détection d'occupation des places, le guidage et l'affichage du statut.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application de **gestion de parking**
- Le projet nécessite la **détection d'occupation** des places de parking
- Il faut un **système de guidage** vers les places disponibles
- L'application doit afficher le **statut en temps réel** du parking
- Le modèle **YOLO26** est utilisé pour la détection de véhicules
- L'utilisateur mentionne : parking, place, stationnement, guidage, occupation, véhicule

**Ne pas utiliser cette référence pour :**
- Comptage d'objets (utiliser `object-counting.md`)
- Sécurité/alarme (utiliser `security-alarm.md`)
- Mesure de distance (utiliser `distance-measurement.md`)

---

## Table des matières

1. [Architecture du Système de Parking](#1-architecture-du-système-de-parking)
2. [Annotation des Places de Parking](#2-annotation-des-places-de-parking)
3. [Détection d'Occupation](#3-détection-doccupation)
4. [Système de Guidage](#4-système-de-guidage)
5. [Affichage du Statut UI](#5-affichage-du-statut-ui)
6. [Exemple Complet Flutter](#6-exemple-complet-flutter)
7. [Optimisations Mobile](#7-optimisations-mobile)
8. [Dépannage](#8-dépannage)

---

## 1. Architecture du Système de Parking

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                   Application Flutter                     │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Camera  │→  │  YOLO26  │→ │  Parking Analyzer    │ │
│  │  Stream  │   │  Detect  │   │  (occupation + guidage)│ │
│  └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                           │              │
│  ┌──────────┐   ┌──────────┐   ┌──────────▼───────────┐ │
│  │   UI     │←  │ Guidance │← │  Parking State       │ │
│  │  Status  │   │  Engine  │   │  Manager             │ │
│  └──────────┘   └──────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Temps réel, CPU | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

### Classes COCO pertinentes

| ID | Nom | Usage parking |
|----|-----|---------------|
| 2 | `car` | **Voiture** — occupation standard |
| 3 | `motorcycle` | **Moto** — place moto |
| 5 | `bus` | **Bus** — grande place |
| 7 | `truck` | **Camion** — grande place |
| 1 | `bicycle` | **Vélo** — place vélo |

---

## 2. Annotation des Places de Parking

### 2.1 Modèle de place de parking

```dart
/// Statut d'une place de parking
enum SpotStatus {
  available,   // Libre
  occupied,    // Occupée
  reserved,    // Réservée
  disabled,    // PMR / handicapé
  electric,    // Borne de recharge
  unknown,     // Indéterminé
}

/// Type de place
enum SpotType {
  standard,
  compact,     // Petite voiture
  large,       // Bus / camion
  disabled,    // PMR
  electric,    // Recharge électrique
  motorcycle,  // Moto / scooter
}

/// Place de parking
class ParkingSpot {
  final String id;
  final String label;           // Ex: "A1", "B3"
  final List<Offset> polygon;   // Points normalisés (0-1)
  final SpotType type;
  SpotStatus status;
  final double confidence;      // Confiance de la détection
  DateTime? lastUpdated;
  String? occupiedByLabel;      // Classe de véhicule détecté

  ParkingSpot({
    required this.id,
    required this.label,
    required this.polygon,
    required this.type,
    this.status = SpotStatus.available,
    this.confidence = 0.0,
    this.lastUpdated,
    this.occupiedByLabel,
  });

  /// Calcule le centroïde de la place
  Offset get centroid {
    double sumX = 0, sumY = 0;
    for (final pt in polygon) {
      sumX += pt.dx;
      sumY += pt.dy;
    }
    return Offset(sumX / polygon.length, sumY / polygon.length);
  }

  /// Vérifie si un point est dans la place
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

  /// Met à jour le statut
  void updateStatus(SpotStatus newStatus, {String? vehicleLabel, double? conf}) {
    status = newStatus;
    lastUpdated = DateTime.now();
    occupiedByLabel = vehicleLabel;
    if (conf != null) confidence = conf;
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'label': label,
        'type': type.name,
        'status': status.name,
        'confidence': confidence,
        'lastUpdated': lastUpdated?.toIso8601String(),
        'occupiedByLabel': occupiedByLabel,
      };
}
```

### 2.2 Gestionnaire de places

```dart
/// Gestionnaire des places de parking
class ParkingLot {
  final String id;
  final String name;
  final List<ParkingSpot> spots;
  final Map<String, List<String>> adjacency; // ID place → places adjacentes

  ParkingLot({
    required this.id,
    required this.name,
    required this.spots,
    Map<String, List<String>>? adjacency,
  }) : adjacency = adjacency ?? {};

  int get totalSpots => spots.length;

  int get availableSpots => spots.where((s) => s.status == SpotStatus.available).length;

  int get occupiedSpots => spots.where((s) => s.status == SpotStatus.occupied).length;

  int get reservedSpots => spots.where((s) => s.status == SpotStatus.reserved).length;

  double get occupancyRate => totalSpots > 0 ? occupiedSpots / totalSpots : 0.0;

  /// Trouve une place par son ID
  ParkingSpot? getSpot(String spotId) {
    try {
      return spots.firstWhere((s) => s.id == spotId);
    } catch (_) {
      return null;
    }
  }

  /// Trouve les places disponibles par type
  List<ParkingSpot> getAvailableSpots({SpotType? type}) {
    return spots.where((s) {
      if (s.status != SpotStatus.available) return false;
      if (type != null && s.type != type) return false;
      return true;
    }).toList();
  }

  /// Trouve la place disponible la plus proche d'un point
  ParkingSpot? findNearestAvailable(Offset fromPoint, {SpotType? type}) {
    final available = getAvailableSpots(type: type);
    if (available.isEmpty) return null;

    ParkingSpot? nearest;
    double minDistance = double.infinity;

    for (final spot in available) {
      final dx = spot.centroid.dx - fromPoint.dx;
      final dy = spot.centroid.dy - fromPoint.dy;
      final distance = dx * dx + dy * dy;

      if (distance < minDistance) {
        minDistance = distance;
        nearest = spot;
      }
    }

    return nearest;
  }

  /// Met à jour le statut d'une place
  void updateSpotStatus(String spotId, SpotStatus status, {String? vehicleLabel, double? conf}) {
    final spot = getSpot(spotId);
    if (spot != null) {
      spot.updateStatus(status, vehicleLabel: vehicleLabel, conf: conf);
    }
  }

  /// Réinitialise toutes les places
  void reset() {
    for (final spot in spots) {
      spot.status = SpotStatus.available;
      spot.occupiedByLabel = null;
      spot.confidence = 0.0;
    }
  }
}
```

### 2.3 Exemple de configuration de parking

```dart
class ParkingLotFactory {
  /// Crée un parking simple en rangées
  static ParkingLot createSimpleLot({
    required String id,
    required String name,
    required int rows,
    required int columns,
    double startX = 0.1,
    double startY = 0.1,
    double spotWidth = 0.08,
    double spotHeight = 0.15,
    double gapX = 0.02,
    double gapY = 0.05,
  }) {
    final spots = <ParkingSpot>[];
    final rowLabels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        final x = startX + col * (spotWidth + gapX);
        final y = startY + row * (spotHeight + gapY);

        final polygon = [
          Offset(x, y),
          Offset(x + spotWidth, y),
          Offset(x + spotWidth, y + spotHeight),
          Offset(x, y + spotHeight),
        ];

        spots.add(ParkingSpot(
          id: '${rowLabels[row]}${col + 1}',
          label: '${rowLabels[row]}${col + 1}',
          polygon: polygon,
          type: SpotType.standard,
        ));
      }
    }

    return ParkingLot(id: id, name: name, spots: spots);
  }

  /// Crée un parking avec des places PMR et électriques
  static ParkingLot createMixedLot({
    required String id,
    required String name,
  }) {
    final spots = <ParkingSpot>[];

    // Rangée A : places standard
    for (int i = 1; i <= 8; i++) {
      spots.add(ParkingSpot(
        id: 'A$i',
        label: 'A$i',
        polygon: _rect(0.05 + (i - 1) * 0.11, 0.1, 0.09, 0.15),
        type: SpotType.standard,
      ));
    }

    // Rangée B : places PMR + électriques
    spots.add(ParkingSpot(
      id: 'B1',
      label: 'B1',
      polygon: _rect(0.05, 0.4, 0.12, 0.18),
      type: SpotType.disabled,
    ));
    spots.add(ParkingSpot(
      id: 'B2',
      label: 'B2',
      polygon: _rect(0.19, 0.4, 0.12, 0.18),
      type: SpotType.disabled,
    ));
    spots.add(ParkingSpot(
      id: 'B3',
      label: 'B3',
      polygon: _rect(0.33, 0.4, 0.10, 0.15),
      type: SpotType.electric,
    ));
    spots.add(ParkingSpot(
      id: 'B4',
      label: 'B4',
      polygon: _rect(0.45, 0.4, 0.10, 0.15),
      type: SpotType.electric,
    ));

    // Rangée C : places standard
    for (int i = 1; i <= 6; i++) {
      spots.add(ParkingSpot(
        id: 'C$i',
        label: 'C$i',
        polygon: _rect(0.05 + (i - 1) * 0.13, 0.7, 0.11, 0.15),
        type: SpotType.standard,
      ));
    }

    return ParkingLot(id: id, name: name, spots: spots);
  }

  static List<Offset> _rect(double x, double y, double w, double h) {
    return [
      Offset(x, y),
      Offset(x + w, y),
      Offset(x + w, y + h),
      Offset(x, y + h),
    ];
  }
}
```

---

## 3. Détection d'Occupation

### 3.1 Analyseur d'occupation

```dart
/// Résultat d'analyse d'une place
class SpotAnalysis {
  final String spotId;
  final SpotStatus status;
  final double confidence;
  final String? vehicleLabel;
  final DateTime timestamp;

  const SpotAnalysis({
    required this.spotId,
    required this.status,
    required this.confidence,
    this.vehicleLabel,
    required this.timestamp,
  });
}

/// Analyseur d'occupation des places
class OccupancyAnalyzer {
  final ParkingLot parkingLot;
  final Map<String, SpotAnalysis> _lastAnalyses = {};
  final Duration _confirmationDelay;
  final double _occupancyThreshold;
  final double _vacancyThreshold;

  OccupancyAnalyzer({
    required this.parkingLot,
    this._confirmationDelay = const Duration(seconds: 2),
    this._occupancyThreshold = 0.3,  // 30% de la place couverte
    this._vacancyThreshold = 0.1,   // 10% de la place couverte
  });

  /// Analyse les détections et met à jour le statut des places
  List<SpotAnalysis> analyze(List<Detection> detections) {
    final analyses = <SpotAnalysis>[];
    final now = DateTime.now();

    for (final spot in parkingLot.spots) {
      // Calculer le chevauchement entre chaque détection et la place
      double maxOverlap = 0;
      String? detectedLabel;

      for (final det in detections) {
        // Vérifier si le centre de la détection est dans la place
        if (!spot.containsPoint(Offset(det.centerX, det.centerY))) continue;

        // Calculer l'IoU (Intersection over Union) simplifié
        final overlap = _calculateOverlap(spot, det);
        if (overlap > maxOverlap) {
          maxOverlap = overlap;
          detectedLabel = det.label;
        }
      }

      // Déterminer le statut selon le chevauchement
      SpotStatus newStatus;
      if (maxOverlap >= _occupancyThreshold) {
        newStatus = SpotStatus.occupied;
      } else if (maxOverlap <= _vacancyThreshold) {
        newStatus = SpotStatus.available;
      } else {
        // Zone intermédiaire — garder le statut précédent
        final lastAnalysis = _lastAnalyses[spot.id];
        newStatus = lastAnalysis?.status ?? SpotStatus.available;
      }

      // Appliquer un délai de confirmation pour éviter les oscillations
      final lastAnalysis = _lastAnalyses[spot.id];
      if (lastAnalysis != null && lastAnalysis.status != newStatus) {
        if (now.difference(lastAnalysis.timestamp) < _confirmationDelay) {
          newStatus = lastAnalysis.status; // Garder l'ancien statut
        }
      }

      final analysis = SpotAnalysis(
        spotId: spot.id,
        status: newStatus,
        confidence: maxOverlap,
        vehicleLabel: detectedLabel,
        timestamp: now,
      );

      _lastAnalyses[spot.id] = analysis;
      analyses.add(analysis);

      // Mettre à jour la place
      spot.updateStatus(newStatus, vehicleLabel: detectedLabel, conf: maxOverlap);
    }

    return analyses;
  }

  /// Calcule le chevauchement entre une place et une détection
  double _calculateOverlap(ParkingSpot spot, Detection det) {
    // Convertir la place en bounding box
    double minX = double.infinity, minY = double.infinity;
    double maxX = 0, maxY = 0;

    for (final pt in spot.polygon) {
      if (pt.dx < minX) minX = pt.dx;
      if (pt.dy < minY) minY = pt.dy;
      if (pt.dx > maxX) maxX = pt.dx;
      if (pt.dy > maxY) maxY = pt.dy;
    }

    // Bounding box de la détection
    final detX = det.x;
    final detY = det.y;
    final detW = det.width;
    final detH = det.height;

    // Intersection
    final interX = math.max(0, math.min(maxX, detX + detW) - math.max(minX, detX));
    final interY = math.max(0, math.min(maxY, detY + detH) - math.max(minY, detY));
    final intersection = interX * interY;

    // Aire de la place
    final spotArea = (maxX - minX) * (maxY - minY);

    return spotArea > 0 ? intersection / spotArea : 0;
  }

  /// Réinitialise l'analyseur
  void reset() {
    _lastAnalyses.clear();
  }
}
```

---

## 4. Système de Guidage

### 4.1 Moteur de guidage

```dart
/// Instruction de guidage
class GuidanceInstruction {
  final String spotId;
  final String spotLabel;
  final Offset spotPosition;
  final double distance;        // Distance normalisée
  final String direction;       // 'left', 'right', 'straight', 'back'
  final String instruction;     // Texte d'instruction
  final SpotType spotType;

  const GuidanceInstruction({
    required this.spotId,
    required this.spotLabel,
    required this.spotPosition,
    required this.distance,
    required this.direction,
    required this.instruction,
    required this.spotType,
  });
}

/// Moteur de guidage vers les places disponibles
class GuidanceEngine {
  final ParkingLot parkingLot;
  final Offset entrancePosition; // Position de l'entrée (normalisée)

  GuidanceEngine({
    required this.parkingLot,
    required this.entrancePosition,
  });

  /// Trouve la meilleure place disponible et génère les instructions
  GuidanceInstruction? findBestSpot({
    SpotType? preferredType,
    Offset? fromPosition,
  }) {
    final from = fromPosition ?? entrancePosition;
    final available = parkingLot.getAvailableSpots(type: preferredType);

    if (available.isEmpty) return null;

    // Trouver la place la plus proche
    ParkingSpot? bestSpot;
    double minDistance = double.infinity;

    for (final spot in available) {
      final dx = spot.centroid.dx - from.dx;
      final dy = spot.centroid.dy - from.dy;
      final distance = dx * dx + dy * dy;

      if (distance < minDistance) {
        minDistance = distance;
        bestSpot = spot;
      }
    }

    if (bestSpot == null) return null;

    // Générer les instructions
    final distance = math.sqrt(minDistance);
    final direction = _calculateDirection(from, bestSpot.centroid);
    final instruction = _generateInstruction(bestSpot, direction, distance);

    return GuidanceInstruction(
      spotId: bestSpot.id,
      spotLabel: bestSpot.label,
      spotPosition: bestSpot.centroid,
      distance: distance,
      direction: direction,
      instruction: instruction,
      spotType: bestSpot.type,
    );
  }

  /// Calcule la direction relative
  String _calculateDirection(Offset from, Offset to) {
    final dx = to.dx - from.dx;
    final dy = to.dy - from.dy;

    // Angle en degrés
    final angle = math.atan2(dy, dx) * 180 / math.pi;

    if (angle >= -45 && angle < 45) return 'right';
    if (angle >= 45 && angle < 135) return 'straight';
    if (angle >= 135 || angle < -135) return 'left';
    return 'back';
  }

  /// Génère une instruction textuelle
  String _generateInstruction(ParkingSpot spot, String direction, double distance) {
    final directionText = switch (direction) {
      'right' => 'à droite',
      'left' => 'à gauche',
      'straight' => 'tout droit',
      'back' => 'en arrière',
      _ => '',
    };

    final distanceText = distance < 0.1
        ? 'proche'
        : distance < 0.3
            ? 'à moyenne distance'
            : 'au fond';

    final typeText = switch (spot.type) {
      SpotType.standard => 'place standard',
      SpotType.disabled => 'place PMR',
      SpotType.electric => 'place avec borne de recharge',
      SpotType.motorcycle => 'place moto',
      SpotType.compact => 'place compacte',
      SpotType.large => 'grande place',
    };

    return 'Place $directionText $distanceText — $typeText (${spot.label})';
  }

  /// Récupère le résumé du parking
  ParkingSummary getSummary() {
    return ParkingSummary(
      total: parkingLot.totalSpots,
      available: parkingLot.availableSpots,
      occupied: parkingLot.occupiedSpots,
      reserved: parkingLot.reservedSpots,
      occupancyRate: parkingLot.occupancyRate,
    );
  }
}

/// Résumé du parking
class ParkingSummary {
  final int total;
  final int available;
  final int occupied;
  final int reserved;
  final double occupancyRate;

  const ParkingSummary({
    required this.total,
    required this.available,
    required this.occupied,
    required this.reserved,
    required this.occupancyRate,
  });

  bool get isFull => available == 0;
  bool get isNearlyFull => occupancyRate >= 0.9;
}
```

---

## 5. Affichage du Statut UI

### 5.1 Widget d'affichage du statut

```dart
/// Widget d'affichage du statut du parking
class ParkingStatusWidget extends StatelessWidget {
  final ParkingSummary summary;
  final GuidanceInstruction? guidance;

  const ParkingStatusWidget({
    super.key,
    required this.summary,
    this.guidance,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // En-tête avec taux d'occupation
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Parking',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              _OccupancyBadge(rate: summary.occupancyRate),
            ],
          ),
          const SizedBox(height: 12),

          // Statistiques
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _StatItem(
                label: 'Total',
                value: '${summary.total}',
                color: Colors.blue,
              ),
              _StatItem(
                label: 'Disponibles',
                value: '${summary.available}',
                color: Colors.green,
              ),
              _StatItem(
                label: 'Occupées',
                value: '${summary.occupied}',
                color: Colors.red,
              ),
              _StatItem(
                label: 'Réservées',
                value: '${summary.reserved}',
                color: Colors.orange,
              ),
            ],
          ),

          // Instruction de guidage
          if (guidance != null) ...[
            const SizedBox(height: 12),
            const Divider(),
            const SizedBox(height: 8),
            _GuidancePanel(instruction: guidance!),
          ],
        ],
      ),
    );
  }
}

class _OccupancyBadge extends StatelessWidget {
  final double rate;

  const _OccupancyBadge({required this.rate});

  @override
  Widget build(BuildContext context) {
    final color = rate >= 0.9
        ? Colors.red
        : rate >= 0.7
            ? Colors.orange
            : Colors.green;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color),
      ),
      child: Text(
        '${(rate * 100).toStringAsFixed(0)}%',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _StatItem({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                color: color,
                fontWeight: FontWeight.bold,
              ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _GuidancePanel extends StatelessWidget {
  final GuidanceInstruction instruction;

  const _GuidancePanel({required this.instruction});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.green.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.green.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(_directionIcon(instruction.direction), color: Colors.green),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Place ${instruction.spotLabel}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.green,
                  ),
                ),
                Text(
                  instruction.instruction,
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  IconData _directionIcon(String direction) {
    switch (direction) {
      case 'right':
        return Icons.arrow_forward;
      case 'left':
        return Icons.arrow_back;
      case 'straight':
        return Icons.arrow_upward;
      case 'back':
        return Icons.arrow_downward;
      default:
        return Icons.help;
    }
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
│   ├── parking_spot.dart
│   └── parking_lot.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── occupancy_analyzer.dart
│   └── guidance_engine.dart
├── screens/
│   └── parking_screen.dart
└── widgets/
    ├── parking_overlay_painter.dart
    └── parking_status_widget.dart
```

### `screens/parking_screen.dart`

```dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../models/parking_lot.dart';
import '../services/yolo_detection_service.dart';
import '../services/occupancy_analyzer.dart';
import '../services/guidance_engine.dart';
import '../widgets/parking_overlay_painter.dart';
import '../widgets/parking_status_widget.dart';

class ParkingScreen extends StatefulWidget {
  const ParkingScreen({super.key});

  @override
  State<ParkingScreen> createState() => _ParkingScreenState();
}

class _ParkingScreenState extends State<ParkingScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  late final ParkingLot _parkingLot;
  late final OccupancyAnalyzer _analyzer;
  late final GuidanceEngine _guidanceEngine;

  List<Detection> _detections = [];
  bool _isProcessing = false;
  bool _isInitialized = false;
  GuidanceInstruction? _currentGuidance;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _yoloService.initialize();

    // Créer le parking
    _parkingLot = ParkingLotFactory.createMixedLot(
      id: 'main',
      name: 'Parking Principal',
    );

    // Initialiser l'analyseur et le guidage
    _analyzer = OccupancyAnalyzer(parkingLot: _parkingLot);
    _guidanceEngine = GuidanceEngine(
      parkingLot: _parkingLot,
      entrancePosition: const Offset(0.5, 0.95),
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

      // Filtrer pour ne garder que les véhicules
      final vehicleDetections = detections
          .where((d) => ['car', 'truck', 'motorcycle', 'bus', 'bicycle']
              .contains(d.label))
          .toList();

      // Analyser l'occupation
      _analyzer.analyze(vehicleDetections);

      // Mettre à jour le guidage
      final guidance = _guidanceEngine.findBestSpot();

      setState(() {
        _detections = vehicleDetections;
        _currentGuidance = guidance;
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

    final summary = _guidanceEngine.getSummary();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Gestion de Parking'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              _parkingLot.reset();
              _analyzer.reset();
            },
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
                  painter: ParkingOverlayPainter(
                    parkingLot: _parkingLot,
                    detections: _detections,
                    guidance: _currentGuidance,
                  ),
                ),
              ],
            ),
          ),

          // Statut du parking
          Expanded(
            flex: 2,
            child: SingleChildScrollView(
              child: ParkingStatusWidget(
                summary: summary,
                guidance: _currentGuidance,
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

### `widgets/parking_overlay_painter.dart`

```dart
import 'package:flutter/material.dart';
import '../models/detection.dart';
import '../models/parking_lot.dart';
import '../services/guidance_engine.dart';

class ParkingOverlayPainter extends CustomPainter {
  final ParkingLot parkingLot;
  final List<Detection> detections;
  final GuidanceInstruction? guidance;

  ParkingOverlayPainter({
    required this.parkingLot,
    required this.detections,
    this.guidance,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Dessiner les places
    for (final spot in parkingLot.spots) {
      _drawSpot(canvas, size, spot);
    }

    // Dessiner les détections
    for (final det in detections) {
      _drawDetection(canvas, size, det);
    }

    // Dessiner le chemin de guidage
    if (guidance != null) {
      _drawGuidancePath(canvas, size, guidance!);
    }
  }

  void _drawSpot(Canvas canvas, Size size, ParkingSpot spot) {
    if (spot.polygon.length < 3) return;

    final path = Path();
    final first = spot.polygon.first;
    path.moveTo(first.dx * size.width, first.dy * size.height);

    for (int i = 1; i < spot.polygon.length; i++) {
      final pt = spot.polygon[i];
      path.lineTo(pt.dx * size.width, pt.dy * size.height);
    }
    path.close();

    // Couleur selon le statut
    final color = _statusColor(spot.status);

    // Remplissage
    canvas.drawPath(path, Paint()..color = color.withOpacity(0.2));

    // Bordure
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Label de la place
    final centroid = spot.centroid;
    final textPainter = TextPainter(
      text: TextSpan(
        text: spot.label,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black.withOpacity(0.5),
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        centroid.dx * size.width - textPainter.width / 2,
        centroid.dy * size.height - textPainter.height / 2,
      ),
    );
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
        ..color = Colors.red.withOpacity(0.5)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  void _drawGuidancePath(Canvas canvas, Size size, GuidanceInstruction guidance) {
    final start = Offset(size.width / 2, size.height * 0.95);
    final end = Offset(
      guidance.spotPosition.dx * size.width,
      guidance.spotPosition.dy * size.height,
    );

    // Ligne pointillée
    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    _drawDashedLine(canvas, start, end, paint);

    // Flèche vers la place
    final direction = (end - start).direction;
    const arrowSize = 15.0;

    final path = Path();
    path.moveTo(end.dx, end.dy);
    path.lineTo(
      end.dx - arrowSize * cos(direction - 0.5),
      end.dy - arrowSize * sin(direction - 0.5),
    );
    path.moveTo(end.dx, end.dy);
    path.lineTo(
      end.dx - arrowSize * cos(direction + 0.5),
      end.dy - arrowSize * sin(direction + 0.5),
    );

    canvas.drawPath(path, paint..strokeWidth = 3);
  }

  void _drawDashedLine(Canvas canvas, Offset start, Offset end, Paint paint) {
    const dashLength = 10.0;
    const gapLength = 5.0;

    final totalDistance = (end - start).distance;
    final direction = (end - start) / totalDistance;

    double drawn = 0;
    while (drawn < totalDistance) {
      final dashEnd = math.min(drawn + dashLength, totalDistance);
      final startPt = start + direction * drawn;
      final endPt = start + direction * dashEnd;

      canvas.drawLine(startPt, endPt, paint);

      drawn = dashEnd + gapLength;
    }
  }

  Color _statusColor(SpotStatus status) {
    switch (status) {
      case SpotStatus.available:
        return Colors.green;
      case SpotStatus.occupied:
        return Colors.red;
      case SpotStatus.reserved:
        return Colors.orange;
      case SpotStatus.disabled:
        return Colors.blue;
      case SpotStatus.electric:
        return Colors.purple;
      case SpotStatus.unknown:
        return Colors.grey;
    }
  }

  @override
  bool shouldRepaint(covariant ParkingOverlayPainter oldDelegate) => true;
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
| FPS cible | ≥ 10 | Suffisant pour la détection d'occupation |
| Seuil occupation | 0.3 (30%) | Éviter les faux négatifs |
| Délai confirmation | 2s | Éviter les oscillations |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

---

## 8. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Places non détectées | Polygones mal définis | Vérifier les coordonnées normalisées |
| Statut instable | Délai de confirmation trop court | Augmenter `_confirmationDelay` |
| Guidage incorrect | Position d'entrée erronée | Vérifier `entrancePosition` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Faux positifs occupation | Seuil trop bas | Augmenter `_occupancyThreshold` |
| Places jamais disponibles | Véhicules stationnés longtemps | Normal — comportement attendu |
| Caméra ne s'initialise pas | Permissions manquantes | Vérifier Info.plist / AndroidManifest |

### Commandes de debug

```dart
if (kDebugMode) {
  debugPrint('Places totales: ${_parkingLot.totalSpots}');
  debugPrint('Disponibles: ${_parkingLot.availableSpots}');
  debugPrint('Occupées: ${_parkingLot.occupiedSpots}');
  debugPrint('Taux: ${_parkingLot.occupancyRate}');
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Comptage d'objets (similaire)](./object-counting.md)
