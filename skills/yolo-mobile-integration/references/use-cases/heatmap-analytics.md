# Référence Complète : Analyse de Heatmap avec YOLO26

> Guide complet pour l'intégration d'un système d'analyse de heatmap dans une application mobile Flutter utilisant YOLO26 pour l'accumulation sur grille, la décroissance temporelle et la visualisation.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application d'**analyse de heatmap / densité**
- Le projet nécessite l'**accumulation de présence** sur une grille
- Il faut de la **décroissance temporelle** pour les données anciennes
- L'application doit afficher une **carte de chaleur** en overlay
- Le modèle **YOLO26** est utilisé pour la détection + accumulation
- L'utilisateur mentionne : heatmap, densité, affluence, analyse, fréquentation, zone chaude

**Ne pas utiliser cette référence pour :**
- Sécurité/alarme (utiliser `security-alarm.md`)
- Comptage d'objets (utiliser `object-counting.md`)
- Gestion de parking (utiliser `parking-management.md`)

---

## Table des matières

1. [Architecture de la Heatmap](#1-architecture-de-la-heatmap)
2. [Accumulation sur Grille](#2-accumulation-sur-grille)
3. [Décroissance Temporelle](#3-décroissance-temporelle)
4. [Visualisation Overlay](#4-visualisation-overlay)
5. [Exemple Complet Flutter](#5-exemple-complet-flutter)
6. [Optimisations Mobile](#6-optimisations-mobile)
7. [Dépannage](#7-dépannage)

---

## 1. Architecture de la Heatmap

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────────┐
│                   Application Flutter                     │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Camera  │→  │  YOLO26  │→ │  Grid Accumulator    │ │
│  │  Stream  │   │  Detect  │   │  (heatmap grid)      │ │
│  └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                           │              │
│  ┌──────────┐   ┌──────────┐   ┌──────────▼───────────┐ │
│  │   UI     │←  │ Overlay  │← │  Temporal Decay      │ │
│  │  Heatmap │   │ Renderer │   │  Engine              │ │
│  └──────────┘   └──────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Temps réel, CPU | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

---

## 2. Accumulation sur Grille

### 2.1 Principe de la grille de heatmap

L'image est divisée en une grille de cellules. Chaque détection incrémente la valeur de la cellule correspondante.

```
Image 640×480 divisée en grille 16×12:

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

### 2.2 Implémentation de la grille

```dart
import 'dart:math' as math;

/// Grille de heatmap
class HeatmapGrid {
  final int rows;
  final int columns;
  final List<List<double>> _grid;
  final List<List<double>> _decayGrid; // Pour le lissage temporel

  HeatmapGrid({
    required this.rows,
    required this.columns,
  })  : _grid = List.generate(rows, (_) => List.filled(columns, 0.0)),
        _decayGrid = List.generate(rows, (_) => List.filled(columns, 0.0));

  double get maxValue {
    double maxVal = 0;
    for (final row in _grid) {
      for (final val in row) {
        if (val > maxVal) maxVal = val;
      }
    }
    return maxVal;
  }

  /// Accumule une détection dans la grille
  void accumulate(Detection detection, {double weight = 1.0}) {
    // Convertir le centroïde en coordonnées de grille
    final col = (detection.centerX * columns).floor().clamp(0, columns - 1);
    final row = (detection.centerY * rows).floor().clamp(0, rows - 1);

    // Accumuler avec un noyau gaussien pour lisser
    _applyGaussianKernel(row, col, weight);
  }

  /// Applique un noyau gaussien autour du point
  void _applyGaussianKernel(int centerRow, int centerCol, double weight) {
    const radius = 2;
    const sigma = 1.0;

    for (int r = -radius; r <= radius; r++) {
      for (int c = -radius; c <= radius; c++) {
        final row = centerRow + r;
        final col = centerCol + c;

        if (row < 0 || row >= rows || col < 0 || col >= columns) continue;

        final distance = math.sqrt(r * r + c * c);
        final gaussianWeight = math.exp(-(distance * distance) / (2 * sigma * sigma));

        _grid[row][col] += weight * gaussianWeight;
      }
    }
  }

  /// Accumule plusieurs détections
  void accumulateAll(List<Detection> detections, {double weight = 1.0}) {
    for (final det in detections) {
      accumulate(det, weight: weight);
    }
  }

  /// Récupère la valeur d'une cellule
  double getValue(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= columns) return 0;
    return _grid[row][col];
  }

  /// Récupère la valeur normalisée (0-1)
  double getNormalizedValue(int row, int col) {
    final maxVal = maxValue;
    if (maxVal == 0) return 0;
    return getValue(row, col) / maxVal;
  }

  /// Récupère les coordonnées de grille d'un point normalisé
  (int row, int col) pointToGrid(Offset point) {
    final col = (point.dx * columns).floor().clamp(0, columns - 1);
    final row = (point.dy * rows).floor().clamp(0, rows - 1);
    return (row, col);
  }

  /// Récupère le centroïde d'une cellule en coordonnées normalisées
  Offset gridToPoint(int row, int col) {
    return Offset(
      (col + 0.5) / columns,
      (row + 0.5) / rows,
    );
  }

  /// Récupère les zones chaudes (cellules au-dessus d'un seuil)
  List<Hotspot> getHotspots({double threshold = 0.5}) {
    final hotspots = <Hotspot>[];
    final maxVal = maxValue;

    if (maxVal == 0) return hotspots;

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        final normalized = _grid[r][c] / maxVal;
        if (normalized >= threshold) {
          hotspots.add(Hotspot(
            row: r,
            col: c,
            position: gridToPoint(r, c),
            intensity: normalized,
            value: _grid[r][c],
          ));
        }
      }
    }

    // Trier par intensité décroissante
    hotspots.sort((a, b) => b.intensity.compareTo(a.intensity));
    return hotspots;
  }

  /// Réinitialise la grille
  void reset() {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        _grid[r][c] = 0.0;
      }
    }
  }

  /// Récupère une copie de la grille
  List<List<double>> get grid => _grid.map((row) => List<double>.from(row)).toList();
}

/// Zone chaude
class Hotspot {
  final int row;
  final int col;
  final Offset position;
  final double intensity; // 0-1
  final double value;

  const Hotspot({
    required this.row,
    required this.col,
    required this.position,
    required this.intensity,
    required this.value,
  });
}
```

---

## 3. Décroissance Temporelle

### 3.1 Principe de la décroissance

La décroissance temporelle réduit progressivement l'intensité des cellules pour que les données anciennes perdent de l'importance.

```
Valeur au temps t: V(t) = V0 × e^(-λt)

Où:
    V0 = valeur initiale
    λ = taux de décroissance
    t = temps écoulé
```

### 3.2 Moteur de décroissance

```dart
/// Moteur de décroissance temporelle
class TemporalDecayEngine {
  final HeatmapGrid grid;
  final double decayRate;        // λ (lambda)
  final Duration updateInterval; // Intervalle de mise à jour
  DateTime _lastUpdate;

  TemporalDecayEngine({
    required this.grid,
    this.decayRate = 0.01,        // 1% de décroissance par seconde
    this.updateInterval = const Duration(seconds: 1),
  }) : _lastUpdate = DateTime.now();

  /// Applique la décroissance à toute la grille
  void applyDecay() {
    final now = DateTime.now();
    final elapsed = now.difference(_lastUpdate).inMilliseconds / 1000.0;

    if (elapsed < updateInterval.inMilliseconds / 1000.0) return;

    // Facteur de décroissance: e^(-λ × Δt)
    final decayFactor = math.exp(-decayRate * elapsed);

    for (int r = 0; r < grid.rows; r++) {
      for (int c = 0; c < grid.columns; c++) {
        final currentValue = grid.getValue(r, c);
        if (currentValue > 0) {
          final newValue = currentValue * decayFactor;
          // Éviter les valeurs trop petites
          if (newValue < 0.001) {
            // Réinitialiser à 0
            grid._grid[r][c] = 0.0;
          } else {
            grid._grid[r][c] = newValue;
          }
        }
      }
    }

    _lastUpdate = now;
  }

  /// Applique une décroissance partielle (pour les zones récemment actives)
  void applyPartialDecay({double factor = 0.95}) {
    for (int r = 0; r < grid.rows; r++) {
      for (int c = 0; c < grid.columns; c++) {
        final currentValue = grid.getValue(r, c);
        if (currentValue > 0) {
          grid._grid[r][c] = currentValue * factor;
        }
      }
    }
  }

  /// Réinitialise le moteur
  void reset() {
    _lastUpdate = DateTime.now();
  }
}
```

---

## 4. Visualisation Overlay

### 4.1 Rendu de la heatmap

```dart
/// Peintre de heatmap
class HeatmapOverlayPainter extends CustomPainter {
  final HeatmapGrid grid;
  final double opacity;
  final bool showGrid;
  final bool showHotspots;

  HeatmapOverlayPainter({
    required this.grid,
    this.opacity = 0.6,
    this.showGrid = false,
    this.showHotspots = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final cellWidth = size.width / grid.columns;
    final cellHeight = size.height / grid.rows;

    // Dessiner les cellules
    for (int r = 0; r < grid.rows; r++) {
      for (int c = 0; c < grid.columns; c++) {
        final value = grid.getNormalizedValue(r, c);
        if (value <= 0) continue;

        final rect = Rect.fromLTWH(
          c * cellWidth,
          r * cellHeight,
          cellWidth,
          cellHeight,
        );

        // Couleur selon l'intensité
        final color = _intensityColor(value);

        canvas.drawRect(
          rect,
          Paint()..color = color.withOpacity(value * opacity),
        );
      }
    }

    // Dessiner la grille (optionnel)
    if (showGrid) {
      _drawGrid(canvas, size, cellWidth, cellHeight);
    }

    // Dessiner les zones chaudes
    if (showHotspots) {
      _drawHotspots(canvas, size);
    }
  }

  void _drawGrid(Canvas canvas, Size size, double cellWidth, double cellHeight) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.1)
      ..strokeWidth = 0.5;

    // Lignes verticales
    for (int c = 0; c <= grid.columns; c++) {
      canvas.drawLine(
        Offset(c * cellWidth, 0),
        Offset(c * cellWidth, size.height),
        paint,
      );
    }

    // Lignes horizontales
    for (int r = 0; r <= grid.rows; r++) {
      canvas.drawLine(
        Offset(0, r * cellHeight),
        Offset(size.width, r * cellHeight),
        paint,
      );
    }
  }

  void _drawHotspots(Canvas canvas, Size size) {
    final hotspots = grid.getHotspots(threshold: 0.6);

    for (final hotspot in hotspots) {
      final center = Offset(
        hotspot.position.dx * size.width,
        hotspot.position.dy * size.height,
      );

      // Cercle de zone chaude
      final radius = 20.0 + hotspot.intensity * 30.0;
      canvas.drawCircle(
        center,
        radius,
        Paint()
          ..color = _intensityColor(hotspot.intensity).withOpacity(0.3)
          ..style = PaintingStyle.fill,
      );

      canvas.drawCircle(
        center,
        radius,
        Paint()
          ..color = _intensityColor(hotspot.intensity)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );

      // Label
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${(hotspot.intensity * 100).toStringAsFixed(0)}%',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            backgroundColor: Colors.black54,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          center.dx - textPainter.width / 2,
          center.dy - textPainter.height / 2,
        ),
      );
    }
  }

  /// Couleur selon l'intensité (bleu → vert → jaune → rouge)
  Color _intensityColor(double intensity) {
    if (intensity < 0.25) {
      return Color.lerp(Colors.blue, Colors.cyan, intensity * 4)!;
    } else if (intensity < 0.5) {
      return Color.lerp(Colors.cyan, Colors.green, (intensity - 0.25) * 4)!;
    } else if (intensity < 0.75) {
      return Color.lerp(Colors.green, Colors.yellow, (intensity - 0.5) * 4)!;
    } else {
      return Color.lerp(Colors.yellow, Colors.red, (intensity - 0.75) * 4)!;
    }
  }

  @override
  bool shouldRepaint(covariant HeatmapOverlayPainter oldDelegate) => true;
}
```

---

## 5. Exemple Complet Flutter

### Structure du projet

```
lib/
├── main.dart
├── models/
│   └── detection.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── heatmap_grid.dart
│   └── temporal_decay_engine.dart
├── screens/
│   └── heatmap_screen.dart
└── widgets/
    ├── heatmap_overlay_painter.dart
    └── heatmap_stats_widget.dart
```

### `screens/heatmap_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../services/yolo_detection_service.dart';
import '../services/heatmap_grid.dart';
import '../services/temporal_decay_engine.dart';
import '../widgets/heatmap_overlay_painter.dart';
import '../widgets/heatmap_stats_widget.dart';

class HeatmapScreen extends StatefulWidget {
  const HeatmapScreen({super.key});

  @override
  State<HeatmapScreen> createState() => _HeatmapScreenState();
}

class _HeatmapScreenState extends State<HeatmapScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  late final HeatmapGrid _heatmapGrid;
  late final TemporalDecayEngine _decayEngine;

  List<Detection> _detections = [];
  bool _isProcessing = false;
  bool _isInitialized = false;
  List<Hotspot> _hotspots = [];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _yoloService.initialize();

    // Créer la grille de heatmap (16×12 pour un bon équilibre)
    _heatmapGrid = HeatmapGrid(rows: 12, columns: 16);
    _decayEngine = TemporalDecayEngine(
      grid: _heatmapGrid,
      decayRate: 0.005, // Décroissance lente
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

      // Filtrer pour ne garder que les personnes
      final peopleDetections = detections
          .where((d) => d.label == 'person')
          .toList();

      // Accumuler dans la heatmap
      _heatmapGrid.accumulateAll(peopleDetections);

      // Appliquer la décroissance temporelle
      _decayEngine.applyDecay();

      // Mettre à jour les zones chaudes
      final hotspots = _heatmapGrid.getHotspots(threshold: 0.4);

      setState(() {
        _detections = peopleDetections;
        _hotspots = hotspots;
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
        title: const Text('Analyse de Heatmap'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              _heatmapGrid.reset();
              _decayEngine.reset();
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Zone caméra avec overlay heatmap
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                CustomPaint(
                  painter: HeatmapOverlayPainter(
                    grid: _heatmapGrid,
                    opacity: 0.5,
                    showHotspots: true,
                  ),
                ),
              ],
            ),
          ),

          // Statistiques de la heatmap
          Expanded(
            flex: 1,
            child: HeatmapStatsWidget(
              hotspots: _hotspots,
              maxValue: _heatmapGrid.maxValue,
              detectionCount: _detections.length,
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

### `widgets/heatmap_stats_widget.dart`

```dart
import 'package:flutter/material.dart';
import '../services/heatmap_grid.dart';

class HeatmapStatsWidget extends StatelessWidget {
  final List<Hotspot> hotspots;
  final double maxValue;
  final int detectionCount;

  const HeatmapStatsWidget({
    super.key,
    required this.hotspots,
    required this.maxValue,
    required this.detectionCount,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Statistiques générales
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _StatItem(
                label: 'Détections',
                value: '$detectionCount',
                icon: Icons.person,
                color: Colors.blue,
              ),
              _StatItem(
                label: 'Zones chaudes',
                value: '${hotspots.length}',
                icon: Icons.local_fire_department,
                color: Colors.orange,
              ),
              _StatItem(
                label: 'Intensité max',
                value: '${maxValue.toStringAsFixed(1)}',
                icon: Icons.trending_up,
                color: Colors.red,
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Liste des zones chaudes
          if (hotspots.isNotEmpty) ...[
            Text(
              'Zones chaudes',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: hotspots.length.clamp(0, 10),
                itemBuilder: (context, index) {
                  final hotspot = hotspots[index];
                  return _HotspotCard(hotspot: hotspot, index: index);
                },
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _StatItem({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: color),
        const SizedBox(height: 4),
        Text(
          value,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
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

class _HotspotCard extends StatelessWidget {
  final Hotspot hotspot;
  final int index;

  const _HotspotCard({
    required this.hotspot,
    required this.index,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 100,
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: _intensityColor(hotspot.intensity).withOpacity(0.2),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _intensityColor(hotspot.intensity),
        ),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            '#${index + 1}',
            style: TextStyle(
              color: _intensityColor(hotspot.intensity),
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '${(hotspot.intensity * 100).toStringAsFixed(0)}%',
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            'Intensité',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Color _intensityColor(double intensity) {
    if (intensity < 0.25) return Colors.blue;
    if (intensity < 0.5) return Colors.cyan;
    if (intensity < 0.75) return Colors.green;
    if (intensity < 0.9) return Colors.yellow;
    return Colors.red;
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
| FPS cible | ≥ 10 | Accumulation fluide |
| Taille grille | 16×12 | Bon équilibre précision/performance |
| Taux décroissance | 0.005-0.01 | Adapté à la durée de la session |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

### Gestion de la mémoire

```dart
class HeatmapMemoryManager {
  static const int maxGridRows = 24;
  static const int maxGridColumns = 32;
  static const int maxHotspots = 50;

  /// Limite la taille de la grille
  static (int, int) clampGridSize(int rows, int cols) {
    return (
      rows.clamp(4, maxGridRows),
      cols.clamp(4, maxGridColumns),
    );
  }

  /// Limite le nombre de zones chaudes
  static List<Hotspot> limitHotspots(List<Hotspot> hotspots) {
    if (hotspots.length > maxHotspots) {
      return hotspots.sublist(0, maxHotspots);
    }
    return hotspots;
  }
}
```

---

## 7. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Heatmap vide | Aucune détection | Vérifier que le modèle détecte bien les objets |
| Zones chaudes trop grandes | Noyau gaussien trop large | Réduire le rayon du noyau |
| Décroissance trop rapide | Taux λ trop élevé | Réduire `decayRate` |
| Décroissance trop lente | Taux λ trop bas | Augmenter `decayRate` |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Grille trop fine | Trop de cellules | Réduire `rows` et `columns` |
| Grille trop grossière | Pas assez de cellules | Augmenter `rows` et `columns` |
| Zones chaudes instables | Pas de lissage | Ajouter un filtre de lissage temporel |

### Commandes de debug

```dart
if (kDebugMode) {
  debugPrint('Grille: ${_heatmapGrid.rows}×${_heatmapGrid.columns}');
  debugPrint('Valeur max: ${_heatmapGrid.maxValue}');
  debugPrint('Zones chaudes: ${_hotspots.length}');
  debugPrint('Détections: ${_detections.length}');
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Comptage d'objets (similaire)](./object-counting.md)
