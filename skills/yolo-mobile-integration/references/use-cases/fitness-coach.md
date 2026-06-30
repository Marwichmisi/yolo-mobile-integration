# Référence Complète : Coach Fitness avec YOLO26-pose

> Guide complet pour l'intégration d'un coach fitness dans une application mobile Flutter utilisant YOLO26-pose pour la détection de pose, le comptage de répétitions et l'analyse de forme.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application de **fitness / sport / exercice**
- Le projet nécessite la **détection de pose** en temps réel sur mobile
- Il faut compter des **répétitions d'exercices** (squats, pompes, curls)
- L'application doit **analyser la forme** et donner du **feedback vocal**
- Le modèle **YOLO26-pose** est utilisé (variantes `-pose` de YOLO26)
- L'utilisateur mentionne : coach sportif, compteur de reps, analyse de forme, exercice, fitness, workout, gym

**Ne pas utiliser cette référence pour :**
- Détection d'objets simples (utiliser `detection.md`)
- Segmentation d'instances (utiliser `segmentation.md`)
- Comptage d'objets inanimés (utiliser `object-counting.md`)

---

## Table des matières

1. [Informations YOLO26-pose et Keypoints COCO](#1-informations-yolo26-pose-et-keypoints-coco)
2. [Calcul des Angles Articulaires](#2-calcul-des-angles-articulaires)
3. [Algorithmes de Détection d'Exercices](#3-algorithmes-de-detection-dexercices)
4. [Logique de Comptage de Répétitions](#4-logique-de-comptage-de-répétitions)
5. [Algorithme de Scoring de Forme (0-100%)](#5-algorithme-de-scoring-de-forme-0-100)
6. [Intégration du Feedback Vocal](#6-intégration-du-feedback-vocal)
7. [Exemple Complet Flutter](#7-exemple-complet-flutter)
8. [Optimisations Mobile](#8-optimisations-mobile)
9. [Dépannage](#9-dépannage)

---

## 1. Informations YOLO26-pose et Keypoints COCO

### Modèles disponibles

| Modèle | Fichier | Params | Idéal pour |
|--------|---------|--------|------------|
| YOLO26n-pose | `yolo26n-pose.pt` | ~3M | Mobile temps réel, CPU |
| YOLO26s-pose | `yolo26s-pose.pt` | ~11M | Équilibre vitesse/précision |
| YOLO26m-pose | `yolo26m-pose.pt` | ~20M | Haute précision mobile |
| YOLO26l-pose | `yolo26l-pose.pt` | ~26M | Précision maximale |
| YOLO26x-pose | `yolo26x-pose.pt` | ~57M | Serveur/recherche |

### Format de sortie YOLO26-pose

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("image.jpg")

# Structure de résultats :
# results[0].keypoints.shape = (N, 17, 3)  → N personnes, 17 keypoints, (x, y, visibility)
# results[0].boxes.shape = (N, 6)          → N personnes, (x1, y1, x2, y2, conf, class)
```

### Les 17 Keypoints COCO (indices et noms)

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

### Skelettes COCO (connexions pour le dessin)

```dart
// Indices des paires de keypoints pour dessiner le squelette
const List<List<int>> cocoSkeleton = [
  [0, 1],   // nez → œil gauche
  [0, 2],   // nez → œil droit
  [1, 3],   // œil gauche → oreille gauche
  [2, 4],   // œil droit → oreille droite
  [5, 6],   // épaule gauche → épaule droite
  [5, 7],   // épaule gauche → coude gauche
  [7, 9],   // coude gauche → poignet gauche
  [6, 8],   // épaule droite → coude droit
  [8, 10],  // coude droit → poignet droit
  [5, 11],  // épaule gauche → hanche gauche
  [6, 12],  // épaule droite → hanche droite
  [11, 12], // hanche gauche → hanche droite
  [11, 13], // hanche gauche → genou gauche
  [13, 15], // genou gauche → cheville gauche
  [12, 14], // hanche droite → genou droit
  [14, 16], // genou droit → cheville droite
];
```

### Constantes d'indices pour les calculs

```dart
// Indices des keypoints pour les calculs d'exercices
class KeypointIndex {
  static const int nose = 0;
  static const int leftEye = 1;
  static const int rightEye = 2;
  static const int leftEar = 3;
  static const int rightEar = 4;
  static const int leftShoulder = 5;
  static const int rightShoulder = 6;
  static const int leftElbow = 7;
  static const int rightElbow = 8;
  static const int leftWrist = 9;
  static const int rightWrist = 10;
  static const int leftHip = 11;
  static const int rightHip = 12;
  static const int leftKnee = 13;
  static const int rightKnee = 14;
  static const int leftAnkle = 15;
  static const int rightAnkle = 16;
}
```

---

## 2. Calcul des Angles Articulaires

### Algorithme de calcul d'angle entre 3 points

L'angle est calculé entre trois points : le point central est le sommet de l'angle (articulation).

```dart
import 'dart:math';

class AngleCalculator {
  /// Calcule l'angle en degrés entre 3 points (A → B → C où B est le sommet)
  static double calculateAngle(Point a, Point b, Point c) {
    final ba = Point(a.x - b.x, a.y - b.y);
    final bc = Point(c.x - b.x, c.y - b.y);

    final dotProduct = ba.x * bc.x + ba.y * bc.y;
    final magnitudeBA = sqrt(ba.x * ba.x + ba.y * ba.y);
    final magnitudeBC = sqrt(bc.x * bc.x + bc.y * bc.y);

    if (magnitudeBA == 0 || magnitudeBC == 0) return 0;

    final cosAngle = dotProduct / (magnitudeBA * magnitudeBC);
    // Clamp pour éviter les erreurs d'arrondi
    final clampedCos = cosAngle.clamp(-1.0, 1.0);
    return acos(clampedCos) * (180 / pi);
  }

  /// Calcule l'angle du coude (épaule → coude → poignet)
  static double elbowAngle(List<Point> keypoints) {
    return calculateAngle(
      Point(keypoints[KeypointIndex.leftShoulder].x, keypoints[KeypointIndex.leftShoulder].y),
      Point(keypoints[KeypointIndex.leftElbow].x, keypoints[KeypointIndex.leftElbow].y),
      Point(keypoints[KeypointIndex.leftWrist].x, keypoints[KeypointIndex.leftWrist].y),
    );
  }

  /// Calcule l'angle du genou (hanche → genou → cheville)
  static double kneeAngle(List<Point> keypoints) {
    return calculateAngle(
      Point(keypoints[KeypointIndex.leftHip].x, keypoints[KeypointIndex.leftHip].y),
      Point(keypoints[KeypointIndex.leftKnee].x, keypoints[KeypointIndex.leftKnee].y),
      Point(keypoints[KeypointIndex.leftAnkle].x, keypoints[KeypointIndex.leftAnkle].y),
    );
  }

  /// Calcule l'angle de la hanche (épaule → hanche → genou)
  static double hipAngle(List<Point> keypoints) {
    return calculateAngle(
      Point(keypoints[KeypointIndex.leftShoulder].x, keypoints[KeypointIndex.leftShoulder].y),
      Point(keypoints[KeypointIndex.leftHip].x, keypoints[KeypointIndex.leftHip].y),
      Point(keypoints[KeypointIndex.leftKnee].x, keypoints[KeypointIndex.leftKnee].y),
    );
  }
}
```

### Angles moyens au repos vs en mouvement

| Exercice | Angle au repos (debout) | Angle en contraction |
|----------|------------------------|---------------------|
| Squat | Genou ~180° | Genou ~90° ou moins |
| Pompes | Coude ~160° | Coude ~90° |
| Curl biceps | Coude ~170° | Coude ~45° |

---

## 3. Algorithmes de Détection d'Exercices

### 3.1 Détection de Squat

**Keypoints utilisés :** hanche (11,12), genou (13,14), cheville (15,16)

**Logique :**
1. Calculer l'angle du genou (hanche → genou → cheville)
2. État DEBOUT : angle genou > 150°
3. État ACCROUPI : angle genou < 110°
4. Transition DEBOUT → ACCROUPI → DEBOUT = 1 rep

```dart
class SquatDetector {
  static const double standingThreshold = 150.0;
  static const double squattingThreshold = 110.0;

  /// Détermine l'état actuel du squat
  static ExercisePhase detectPhase(List<Point> keypoints) {
    final angle = AngleCalculator.kneeAngle(keypoints);

    if (angle > standingThreshold) {
      return ExercisePhase.standing;
    } else if (angle < squattingThreshold) {
      return ExercisePhase.squatting;
    }
    return ExercisePhase.transitioning;
  }

  /// Vérifie la qualité de la position accroupie
  static double formScore(List<Point> keypoints) {
    double score = 100.0;
    final kneeAngleVal = AngleCalculator.kneeAngle(keypoints);
    final hipAngleVal = AngleCalculator.hipAngle(keypoints);

    // Pénalité si les genoux dépassent excessivement les pieds
    final kneeX = keypoints[KeypointIndex.leftKnee].x;
    final ankleX = keypoints[KeypointIndex.leftAnkle].x;
    final kneeOverAnkle = (kneeX - ankleX).abs();
    if (kneeOverAnkle > 50) {
      score -= 15; // Genoux trop en avant
    }

    // Pénalité si le dos n'est pas droit (angle hanche trop fermé)
    if (hipAngleVal < 70) {
      score -= 20; // Dos trop penché en avant
    }

    // Pénalité si pas assez profond (angle genou > 110)
    if (kneeAngleVal > 120) {
      score -= 10; // Squat trop superficiel
    }

    // Bonus si profond (angle genou < 90)
    if (kneeAngleVal < 90) {
      score += 5; // Profondeur excellente
    }

    return score.clamp(0.0, 100.0);
  }
}
```

### 3.2 Détection de Pompes (Push-ups)

**Keypoints utilisés :** épaule (5,6), coude (7,8), poignet (9,10)

**Logique :**
1. Calculer l'angle du coude (épaule → coude → poignet)
2. État HAUT : angle coude > 150°
3. État BAS : angle coude < 90°
4. Vérifier que le corps est droit (distance épaule-hanche stable)
5. Transition HAUT → BAS → HAUT = 1 rep

```dart
class PushupDetector {
  static const double extendedThreshold = 150.0;
  static const double loweredThreshold = 90.0;

  static ExercisePhase detectPhase(List<Point> keypoints) {
    final angle = AngleCalculator.elbowAngle(keypoints);

    if (angle > extendedThreshold) {
      return ExercisePhase.standing; // position haute
    } else if (angle < loweredThreshold) {
      return ExercisePhase.squatting; // position basse
    }
    return ExercisePhase.transitioning;
  }

  static double formScore(List<Point> keypoints) {
    double score = 100.0;
    final elbowAngleVal = AngleCalculator.elbowAngle(keypoints);

    // Vérifier l'alignement du corps (épaule-hanche-cheville en ligne)
    final shoulderY = (keypoints[KeypointIndex.leftShoulder].y +
            keypoints[KeypointIndex.rightShoulder].y) /
        2;
    final hipY = (keypoints[KeypointIndex.leftHip].y +
            keypoints[KeypointIndex.rightHip].y) /
        2;
    final ankleY = (keypoints[KeypointIndex.leftAnkle].y +
            keypoints[KeypointIndex.rightAnkle].y) /
        2;

    final bodyAlignment = (shoulderY - hipY).abs() + (hipY - ankleY).abs();
    if (bodyAlignment > 80) {
      score -= 25; // Corps pas droit
    }

    // Pénalité si coude pas assez plié (pompe incomplète)
    if (elbowAngleVal > 110) {
      score -= 20; // Amplitude insuffisante
    }

    // Vérifier la symétrie (les deux coudes plient de la même façon)
    final rightElbowAngle = _rightElbowAngle(keypoints);
    final asymmetry = (elbowAngleVal - rightElbowAngle).abs();
    if (asymmetry > 15) {
      score -= 10; // Asymétrie
    }

    return score.clamp(0.0, 100.0);
  }

  static double _rightElbowAngle(List<Point> keypoints) {
    return AngleCalculator.calculateAngle(
      Point(keypoints[KeypointIndex.rightShoulder].x, keypoints[KeypointIndex.rightShoulder].y),
      Point(keypoints[KeypointIndex.rightElbow].x, keypoints[KeypointIndex.rightElbow].y),
      Point(keypoints[KeypointIndex.rightWrist].x, keypoints[KeypointIndex.rightWrist].y),
    );
  }
}
```

### 3.3 Détection de Curl Biceps

**Keypoints utilisés :** épaule (5,6), coude (7,8), poignet (9,10)

**Logique :**
1. Calculer l'angle du coude (épaule → coude → poignet)
2. État BAS : angle coude > 150° (bras tendu)
3. État HAUT : angle coude < 50° (contraction complète)
4. Vérifier que le coude reste stable (pas de mouvement du coude)
5. Transition BAS → HAUT → BAS = 1 rep

```dart
class BicepCurlDetector {
  static const double extendedThreshold = 150.0;
  static const double contractedThreshold = 50.0;

  static ExercisePhase detectPhase(List<Point> keypoints) {
    final angle = AngleCalculator.elbowAngle(keypoints);

    if (angle > extendedThreshold) {
      return ExercisePhase.standing; // bras tendu
    } else if (angle < contractedThreshold) {
      return ExercisePhase.squatting; // contraction complète
    }
    return ExercisePhase.transitioning;
  }

  static double formScore(List<Point> keypoints) {
    double score = 100.0;
    final elbowAngleVal = AngleCalculator.elbowAngle(keypoints);

    // Vérifier que le coude ne bouge pas (stabilité)
    final elbowY = keypoints[KeypointIndex.leftElbow].y;
    final shoulderY = keypoints[KeypointIndex.leftShoulder].y;
    final elbowDrift = (elbowY - shoulderY).abs();

    if (elbowDrift > 30) {
      score -= 20; // Coude qui dérive
    }

    // Vérifier amplitude complète
    if (elbowAngleVal > 60) {
      score -= 15; // Contraction incomplète
    }

    // Vérifier que le bras descend complètement
    if (elbowAngleVal < 160) {
      score -= 10; // Extension incomplète
    }

    return score.clamp(0.0, 100.0);
  }
}
```

---

## 4. Logique de Comptage de Répétitions

### Machine à états pour le comptage

```
┌─────────────┐
│  INACTIF    │ ← État initial
└──────┬──────┘
       │ Détection exercice
       ▼
┌─────────────┐
│  DEBOUT     │ ← Position de départ
└──────┬──────┘
       │ Angle < seuil bas
       ▼
┌─────────────┐
│ DESCENDANT  │ ← En mouvement vers le bas
└──────┬──────┘
       │ Angle < seuil bas atteint
       ▼
┌─────────────┐
│  ACCROUPI   │ ← Position basse atteinte
└──────┬──────┘
       │ Angle > seuil haut
       ▼
┌─────────────┐
│ MONTANT     │ ← En mouvement vers le haut
└──────┬──────┘
       │ Angle > seuil haut atteint
       ▼
   ★ REP COMPTÉE ★ → retour à DEBOUT
```

### Implémentation de la machine à états

```dart
enum ExercisePhase {
  inactive,
  standing,
  descending,
  bottom,
  ascending,
  transitioning,
}

enum ExerciseType { squat, pushup, bicepCurl }

class RepCounter {
  final ExerciseType exerciseType;
  int _reps = 0;
  ExercisePhase _currentPhase = ExercisePhase.inactive;
  int _frameCount = 0;
  static const int minFramesPerPhase = 3; // Stabilité : 3 frames minimum

  RepCounter(this.exerciseType);

  int get reps => _reps;
  ExercisePhase get currentPhase => _currentPhase;

  /// Met à jour le compteur avec un nouveau jeu de keypoints
  /// Retourne true si une rep a été complétée
  bool update(List<Point> keypoints) {
    _frameCount++;
    if (_frameCount < minFramesPerPhase) return false;

    final newPhase = _detectPhase(keypoints);
    final completedRep = _transition(newPhase);
    return completedRep;
  }

  ExercisePhase _detectPhase(List<Point> keypoints) {
    switch (exerciseType) {
      case ExerciseType.squat:
        return SquatDetector.detectPhase(keypoints);
      case ExerciseType.pushup:
        return PushupDetector.detectPhase(keypoints);
      case ExerciseType.bicepCurl:
        return BicepCurlDetector.detectPhase(keypoints);
    }
  }

  /// Gère les transitions entre états
  /// Retourne true si une rep est complétée
  bool _transition(ExercisePhase newPhase) {
    switch (_currentPhase) {
      case ExercisePhase.inactive:
        if (newPhase == ExercisePhase.standing) {
          _currentPhase = ExercisePhase.standing;
          _frameCount = 0;
        }
        break;

      case ExercisePhase.standing:
        if (newPhase == ExercisePhase.squatting ||
            newPhase == ExercisePhase.transitioning) {
          _currentPhase = ExercisePhase.descending;
          _frameCount = 0;
        }
        break;

      case ExercisePhase.descending:
        if (newPhase == ExercisePhase.squatting) {
          _currentPhase = ExercisePhase.bottom;
          _frameCount = 0;
        } else if (newPhase == ExercisePhase.standing) {
          // Annulation : retour en position debout
          _currentPhase = ExercisePhase.standing;
          _frameCount = 0;
        }
        break;

      case ExercisePhase.bottom:
        if (newPhase == ExercisePhase.standing ||
            newPhase == ExercisePhase.transitioning) {
          _currentPhase = ExercisePhase.ascending;
          _frameCount = 0;
        }
        break;

      case ExercisePhase.ascending:
        if (newPhase == ExercisePhase.standing) {
          // REP COMPLÉTÉE !
          _reps++;
          _currentPhase = ExercisePhase.standing;
          _frameCount = 0;
          return true;
        }
        break;

      case ExercisePhase.transitioning:
        // Transitoire, pas d'action
        break;
    }
    return false;
  }

  void reset() {
    _reps = 0;
    _currentPhase = ExercisePhase.inactive;
    _frameCount = 0;
  }
}
```

### Seuils par exercice

| Exercice | Seuil bas (position basse) | Seuil haut (position haute) |
|----------|---------------------------|----------------------------|
| Squat | Genou < 110° | Genou > 150° |
| Pompes | Coude < 90° | Coude > 150° |
| Curl biceps | Coude < 50° | Coude > 150° |

---

## 5. Algorithme de Scoring de Forme (0-100%)

### Système de scoring global

```dart
class FormScorer {
  static const Map<ExerciseType, List<ScoringCriterion>> criteria = {
    ExerciseType.squat: [
      ScoringCriterion('profondeur', 'Profondeur du squat', 25),
      ScoringCriterion('alignement', 'Alignement genoux/pieds', 25),
      ScoringCriterion('dos', 'Dos droit', 30),
      ScoringCriterion('stabilite', 'Stabilité', 20),
    ],
    ExerciseType.pushup: [
      ScoringCriterion('amplitude', 'Amplitude du mouvement', 30),
      ScoringCriterion('alignement', 'Alignement du corps', 35),
      ScoringCriterion('symetrie', 'Symétrie des bras', 20),
      ScoringCriterion('vitesse', 'Régularité', 15),
    ],
    ExerciseType.bicepCurl: [
      ScoringCriterion('amplitude', 'Amplitude complète', 30),
      ScoringCriterion('stabilite', 'Stabilité du coude', 35),
      ScoringCriterion('vitesse', 'Tempo contrôlé', 20),
      ScoringCriterion('symetrie', 'Symétrie', 15),
    ],
  };

  /// Score global sur les N dernières répétitions
  static double calculateOverallScore(
    ExerciseType type,
    List<double> recentScores,
  ) {
    if (recentScores.isEmpty) return 0;

    // Moyenne des 5 dernières reps
    final lastFive = recentScores.length > 5
        ? recentScores.sublist(recentScores.length - 5)
        : recentScores;

    final average = lastFive.reduce((a, b) => a + b) / lastFive.length;

    // Pénalité pour la fatigue (score diminue avec plus de reps)
    final fatiguePenalty = (recentScores.length > 10)
        ? (recentScores.length - 10) * 0.5
        : 0.0;

    return (average - fatiguePenalty).clamp(0.0, 100.0);
  }

  /// Retourne un label de performance
  static String getPerformanceLabel(double score) {
    if (score >= 90) return 'Excellent';
    if (score >= 75) return 'Bon';
    if (score >= 60) return 'Moyen';
    if (score >= 40) return 'À améliorer';
    return 'Mauvais';
  }

  /// Retourne des conseils basés sur le score par critère
  static List<String> getFeedback(
    ExerciseType type,
    Map<String, double> criterionScores,
  ) {
    final feedback = <String>[];

    criterionScores.forEach((criterion, score) {
      if (score < 50) {
        switch (criterion) {
          case 'profondeur':
            feedback.add('Descendez plus bas, vos cuisses doivent être parallèles au sol');
            break;
          case 'alignement':
            feedback.add('Gardez vos genoux alignés avec vos pieds');
            break;
          case 'dos':
            feedback.add('Gardez le dos droit, ne penchez pas en avant');
            break;
          case 'amplitude':
            feedback.add('Faites le mouvement complet');
            break;
          case 'stabilite':
            feedback.add('Contrôlez le mouvement, pas de balancement');
            break;
          case 'symetrie':
            feedback.add('Utilisez les deux côtés de la même façon');
            break;
          case 'vitesse':
            feedback.add('Ralentissez, contrôlez le tempo');
            break;
        }
      }
    });

    return feedback;
  }
}

class ScoringCriterion {
  final String id;
  final String label;
  final int weight; // pourcentage du score total

  const ScoringCriterion(this.id, this.label, this.weight);
}
```

### Pondération des critères par exercice

#### Squat
| Critère | Poids | Seuil excellent | Seuil mauvais |
|---------|-------|-----------------|---------------|
| Profondeur (genou < 90°) | 25% | < 90° | > 120° |
| Alignement genoux/pieds | 25% | Décalage < 20px | Décalage > 50px |
| Dos droit (angle hanche) | 30% | Angle > 80° | Angle < 60° |
| Stabilité | 20% | Variance < 5° | Variance > 15° |

#### Pompes
| Critère | Poids | Seuil excellent | Seuil mauvais |
|---------|-------|-----------------|---------------|
| Amplitude (coude < 90°) | 30% | < 80° | > 110° |
| Alignement corps | 35% | Décalage < 30px | Décalage > 80px |
| Symétrie bras | 20% | Différence < 10° | Différence > 20° |
| Régularité tempo | 15% | Variance < 0.3s | Variance > 1s |

#### Curl Biceps
| Critère | Poids | Seuil excellent | Seuil mauvais |
|---------|-------|-----------------|---------------|
| Amplitude complète | 30% | 40°-160° | 60°-140° |
| Stabilité coude | 35% | Dérive < 15px | Dérive > 40px |
| Tempo contrôlé | 20% | 1-2s concentrique | > 3s |
| Symétrie | 15% | Différence < 10° | Différence > 20° |

---

## 6. Intégration du Feedback Vocal

### Service de synthèse vocale Flutter

```dart
import 'package:flutter_tts/flutter_tts.dart';

class VoiceFeedbackService {
  final FlutterTts _tts = FlutterTts();
  bool _isEnabled = true;
  DateTime _lastFeedback = DateTime.now();
  static const Duration _feedbackCooldown = Duration(seconds: 3);

  VoiceFeedbackService() {
    _initTts();
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('fr-FR');
    await _tts.setSpeechRate(0.5); // Vitesse modérée
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  void toggle() => _isEnabled = !_isEnabled;

  /// Annonce le nombre de répétitions
  Future<void> announceReps(int reps) async {
    if (!_isEnabled) return;
    await _speak('$reps répétitions');
  }

  /// Feedback quand une rep est complétée
  Future<void> onRepCompleted({
    required int totalReps,
    required double score,
  }) async {
    if (!_isEnabled) return;
    if (!_canGiveFeedback()) return;

    final label = FormScorer.getPerformanceLabel(score);

    if (score >= 90) {
      await _speak('Excellente forme ! $totalReps répétitions');
    } else if (score >= 75) {
      await _speak('Bon ! $totalReps répétitions');
    } else if (score < 50) {
      await _speak('Attention à votre forme');
    }

    _lastFeedback = DateTime.now();
  }

  /// Donne un conseil technique
  Future<void> giveCorrection(String correction) async {
    if (!_isEnabled) return;
    if (!_canGiveFeedback()) return;

    await _speak(correction);
    _lastFeedback = DateTime.now();
  }

  /// Début de séance
  Future<void> startSession(String exerciseName) async {
    if (!_isEnabled) return;
    await _speak('Début des $exerciseName. Allez-y !');
  }

  /// Fin de séance avec résumé
  Future<void> endSession({
    required int totalReps,
    required double avgScore,
  }) async {
    if (!_isEnabled) return;
    final label = FormScorer.getPerformanceLabel(avgScore);
    await _speak(
      'Séance terminée. $totalReps répétitions. Score moyen : $label.',
    );
  }

  bool _canGiveFeedback() {
    return DateTime.now().difference(_lastFeedback) > _feedbackCooldown;
  }

  Future<void> _speak(String text) async {
    await _tts.speak(text);
  }

  Future<void> dispose() async {
    await _tts.stop();
  }
}
```

---

## 7. Exemple Complet Flutter

### Structure du projet

```
lib/
├── main.dart
├── models/
│   ├── keypoint.dart
│   └── exercise_data.dart
├── services/
│   ├── pose_detection_service.dart
│   ├── exercise_analyzer.dart
│   └── voice_feedback_service.dart
├── screens/
│   └── fitness_coach_screen.dart
└── widgets/
    ├── pose_overlay_painter.dart
    └── exercise_stats_widget.dart
```

### `main.dart`

```dart
import 'package:flutter/material.dart';
import 'screens/fitness_coach_screen.dart';

void main() => runApp(const FitnessCoachApp());

class FitnessCoachApp extends StatelessWidget {
  const FitnessCoachApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Coach Fitness',
      theme: ThemeData(
        colorSchemeSeed: Colors.green,
        useMaterial3: true,
      ),
      home: const FitnessCoachScreen(),
    );
  }
}
```

### `models/keypoint.dart`

```dart
import 'dart:math';

class Keypoint {
  final double x;
  final double y;
  final double confidence;
  final int index;

  const Keypoint({
    required this.x,
    required this.y,
    required this.confidence,
    required this.index,
  });

  Point get toPoint => Point(x, y);

  bool get isVisible => confidence > 0.5;

  static const List<String> names = [
    'nez', 'œil_gauche', 'œil_droit', 'oreille_gauche', 'oreille_droite',
    'épaule_gauche', 'épaule_droite', 'coude_gauche', 'coude_droit',
    'poignet_gauche', 'poignet_droit', 'hanche_gauche', 'hanche_droite',
    'genou_gauche', 'genou_droit', 'cheville_gauche', 'cheville_droite',
  ];
}
```

### `services/pose_detection_service.dart`

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart';

class PoseDetectionService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_pose');
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    await _channel.invokeMethod('initializePoseModel');
    _isInitialized = true;
  }

  /// Détecte les poses dans une image
  /// Retourne une liste de personnes, chaque personne étant une liste de 17 keypoints
  Future<List<List<Keypoint>>> detectPose(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('detectPose', {
      'image': imageBytes,
    });

    final List<List<Keypoint>> persons = [];

    for (final person in result['persons']) {
      final keypoints = <Keypoint>[];
      for (int i = 0; i < person.length; i++) {
        keypoints.add(Keypoint(
          x: person[i]['x'].toDouble(),
          y: person[i]['y'].toDouble(),
          confidence: person[i]['confidence'].toDouble(),
          index: i,
        ));
      }
      persons.add(keypoints);
    }

    return persons;
  }

  void dispose() {
    _channel.invokeMethod('disposePoseModel');
    _isInitialized = false;
  }
}
```

### `services/exercise_analyzer.dart`

```dart
import 'dart:math';
import '../models/keypoint.dart';

class ExerciseAnalyzer {
  final ExerciseType exerciseType;
  final RepCounter _repCounter;
  final VoiceFeedbackService _voiceService;
  final List<double> _scores = [];

  ExerciseAnalyzer({
    required this.exerciseType,
    required VoiceFeedbackService voiceService,
  })  : _repCounter = RepCounter(exerciseType),
        _voiceService = voiceService;

  int get reps => _repCounter.reps;
  ExercisePhase get phase => _repCounter.currentPhase;
  double get overallScore => _scores.isEmpty
      ? 0
      : _scores.reduce((a, b) => a + b) / _scores.length;

  /// Analyse une frame et retourne les données d'exercice
  Future<ExerciseData> analyze(List<Keypoint> keypoints) async {
    final points = keypoints.map((k) => k.toPoint).toList();

    // Calculer l'angle pertinent
    double angle;
    switch (exerciseType) {
      case ExerciseType.squat:
        angle = AngleCalculator.kneeAngle(points);
        break;
      case ExerciseType.pushup:
        angle = AngleCalculator.elbowAngle(points);
        break;
      case ExerciseType.bicepCurl:
        angle = AngleCalculator.elbowAngle(points);
        break;
    }

    // Calculer le score de forme
    double score;
    switch (exerciseType) {
      case ExerciseType.squat:
        score = SquatDetector.formScore(points);
        break;
      case ExerciseType.pushup:
        score = PushupDetector.formScore(points);
        break;
      case ExerciseType.bicepCurl:
        score = BicepCurlDetector.formScore(points);
        break;
    }

    // Mettre à jour le compteur de reps
    final repCompleted = _repCounter.update(points);

    if (repCompleted) {
      _scores.add(score);
      await _voiceService.onRepCompleted(
        totalReps: _repCounter.reps,
        score: score,
      );
    }

    return ExerciseData(
      reps: _repCounter.reps,
      currentAngle: angle,
      phase: _repCounter.currentPhase,
      score: score,
      repCompleted: repCompleted,
    );
  }

  void reset() {
    _repCounter.reset();
    _scores.clear();
  }
}

class ExerciseData {
  final int reps;
  final double currentAngle;
  final ExercisePhase phase;
  final double score;
  final bool repCompleted;

  const ExerciseData({
    required this.reps,
    required this.currentAngle,
    required this.phase,
    required this.score,
    required this.repCompleted,
  });
}
```

### `screens/fitness_coach_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/pose_detection_service.dart';
import '../services/exercise_analyzer.dart';
import '../services/voice_feedback_service.dart';
import '../widgets/pose_overlay_painter.dart';
import '../widgets/exercise_stats_widget.dart';

class FitnessCoachScreen extends StatefulWidget {
  const FitnessCoachScreen({super.key});

  @override
  State<FitnessCoachScreen> createState() => _FitnessCoachScreenState();
}

class _FitnessCoachScreenState extends State<FitnessCoachScreen> {
  CameraController? _cameraController;
  final PoseDetectionService _poseService = PoseDetectionService();
  final VoiceFeedbackService _voiceService = VoiceFeedbackService();
  ExerciseAnalyzer? _analyzer;

  List<Keypoint>? _currentKeypoints;
  ExerciseData? _exerciseData;
  bool _isProcessing = false;
  bool _isInitialized = false;
  ExerciseType _selectedExercise = ExerciseType.squat;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _poseService.initialize();
    await _voiceService.startSession(_exerciseLabel);

    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController!.initialize();
    _cameraController!.startImageStream(_processFrame);

    _analyzer = ExerciseAnalyzer(
      exerciseType: _selectedExercise,
      voiceService: _voiceService,
    );

    setState(() => _isInitialized = true);
  }

  String get _exerciseLabel {
    switch (_selectedExercise) {
      case ExerciseType.squat:
        return 'squats';
      case ExerciseType.pushup:
        return 'pompes';
      case ExerciseType.bicepCurl:
        return 'curls biceps';
    }
  }

  Future<void> _processFrame(CameraImage image) async {
    if (_isProcessing || _analyzer == null) return;
    _isProcessing = true;

    try {
      // Convertir l'image en bytes pour YOLO
      final bytes = _imageToBytes(image);
      final persons = await _poseService.detectPose(bytes);

      if (persons.isNotEmpty) {
        final keypoints = persons.first; // Première personne détectée
        final data = await _analyzer!.analyze(keypoints);

        setState(() {
          _currentKeypoints = keypoints;
          _exerciseData = data;
        });
      }
    } catch (e) {
      debugPrint('Erreur de traitement: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Uint8List _imageToBytes(CameraImage image) {
    // Conversion YUV → bytes pour l'inférence
    final buffer = image.planes[0].bytes;
    return buffer;
  }

  void _changeExercise(ExerciseType type) {
    setState(() {
      _selectedExercise = type;
      _analyzer?.reset();
      _analyzer = ExerciseAnalyzer(
        exerciseType: type,
        voiceService: _voiceService,
      );
    });
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
        title: const Text('Coach Fitness'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              _analyzer?.reset();
              setState(() => _exerciseData = null);
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Sélecteur d'exercice
          ExerciseSelector(
            selected: _selectedExercise,
            onChanged: _changeExercise,
          ),

          // Zone caméra avec overlay de pose
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                if (_currentKeypoints != null)
                  CustomPaint(
                    painter: PoseOverlayPainter(
                      keypoints: _currentKeypoints!,
                      exerciseType: _selectedExercise,
                    ),
                  ),
              ],
            ),
          ),

          // Statistiques
          Expanded(
            flex: 1,
            child: ExerciseStatsWidget(
              data: _exerciseData,
              exerciseType: _selectedExercise,
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _poseService.dispose();
    _voiceService.dispose();
    super.dispose();
  }
}

class ExerciseSelector extends StatelessWidget {
  final ExerciseType selected;
  final ValueChanged<ExerciseType> onChanged;

  const ExerciseSelector({
    super.key,
    required this.selected,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: SegmentedButton<ExerciseType>(
        segments: const [
          ButtonSegment(
            value: ExerciseType.squat,
            label: Text('Squats'),
            icon: Icon(Icons.accessibility_new),
          ),
          ButtonSegment(
            value: ExerciseType.pushup,
            label: Text('Pompes'),
            icon: Icon(Icons.fitness_center),
          ),
          ButtonSegment(
            value: ExerciseType.bicepCurl,
            label: Text('Curls'),
            icon: Icon(Icons.sports_gymnastics),
          ),
        ],
        selected: {selected},
        onSelectionChanged: (selection) {
          if (selection.isNotEmpty) onChanged(selection.first);
        },
      ),
    );
  }
}
```

### `widgets/pose_overlay_painter.dart`

```dart
import 'dart:math';
import 'package:flutter/material.dart';
import '../models/keypoint.dart';

class PoseOverlayPainter extends CustomPainter {
  final List<Keypoint> keypoints;
  final ExerciseType exerciseType;

  PoseOverlayPainter({
    required this.keypoints,
    required this.exerciseType,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    // Dessiner le squelette
    for (final pair in cocoSkeleton) {
      final kpA = keypoints[pair[0]];
      final kpB = keypoints[pair[1]];

      if (kpA.isVisible && kpB.isVisible) {
        // Couleur verte si visible, rouge si non visible
        paint.color = Colors.green.withOpacity(0.8);
        canvas.drawLine(
          Offset(kpA.x, kpA.y),
          Offset(kpB.x, kpB.y),
          paint,
        );
      }
    }

    // Dessiner les keypoints
    for (final kp in keypoints) {
      if (kp.isVisible) {
        final circlePaint = Paint()
          ..color = _getColorForKeypoint(kp.index)
          ..style = PaintingStyle.fill;
        canvas.drawCircle(Offset(kp.x, kp.y), 6, circlePaint);

        // Bordure blanche
        canvas.drawCircle(
          Offset(kp.x, kp.y),
          6,
          Paint()
            ..color = Colors.white
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2,
        );
      }
    }

    // Dessiner l'angle en surbrillance pour l'exercice actuel
    _drawAngleArc(canvas, size);
  }

  Color _getColorForKeypoint(int index) {
    // Couleurs par groupe articulaire
    if (index <= 4) return Colors.blue; // Tête
    if (index <= 6) return Colors.orange; // Épaules
    if (index <= 10) return Colors.red; // Bras
    if (index <= 12) return Colors.purple; // Hanches
    return Colors.green; // Jambes
  }

  void _drawAngleArc(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.yellow.withOpacity(0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    Point? a, b, c; // A → B (sommet) → C

    switch (exerciseType) {
      case ExerciseType.squat:
        a = keypoints[KeypointIndex.leftHip].toPoint;
        b = keypoints[KeypointIndex.leftKnee].toPoint;
        c = keypoints[KeypointIndex.leftAnkle].toPoint;
        break;
      case ExerciseType.pushup:
      case ExerciseType.bicepCurl:
        a = keypoints[KeypointIndex.leftShoulder].toPoint;
        b = keypoints[KeypointIndex.leftElbow].toPoint;
        c = keypoints[KeypointIndex.leftWrist].toPoint;
        break;
    }

    if (a != null && b != null && c != null) {
      final angle = AngleCalculator.calculateAngle(a, b, c);
      final radius = 30.0;

      canvas.drawArc(
        Rect.fromCircle(center: Offset(b.x, b.y), radius: radius),
        _angleFromPoint(a, b),
        _sweepAngle(a, b, c),
        false,
        paint,
      );

      // Afficher l'angle en texte
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${angle.toStringAsFixed(0)}°',
          style: const TextStyle(
            color: Colors.yellow,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(b.x + 15, b.y - 15),
      );
    }
  }

  double _angleFromPoint(Point a, Point b) {
    return atan2(a.y - b.y, a.x - b.x);
  }

  double _sweepAngle(Point a, Point b, Point c) {
    final startAngle = _angleFromPoint(a, b);
    final endAngle = _angleFromPoint(c, b);
    return endAngle - startAngle;
  }

  @override
  bool shouldRepaint(covariant PoseOverlayPainter oldDelegate) => true;
}
```

### `widgets/exercise_stats_widget.dart`

```dart
import 'package:flutter/material.dart';

class ExerciseStatsWidget extends StatelessWidget {
  final ExerciseData? data;
  final ExerciseType exerciseType;

  const ExerciseStatsWidget({
    super.key,
    required this.data,
    required this.exerciseType,
  });

  @override
  Widget build(BuildContext context) {
    final reps = data?.reps ?? 0;
    final angle = data?.currentAngle ?? 0;
    final score = data?.score ?? 0;
    final phase = data?.phase ?? ExercisePhase.inactive;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
      ),
      child: Column(
        children: [
          // Compteur de reps
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _StatCard(
                label: 'Répétitions',
                value: '$reps',
                icon: Icons.repeat,
              ),
              _StatCard(
                label: 'Angle',
                value: '${angle.toStringAsFixed(0)}°',
                icon: Icons.architecture,
              ),
              _StatCard(
                label: 'Score',
                value: '${score.toStringAsFixed(0)}%',
                icon: Icons.star,
                color: _scoreColor(score),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Indicateur de phase
          LinearProgressIndicator(
            value: _phaseProgress(phase),
            backgroundColor: Colors.grey[300],
            color: _phaseColor(phase),
          ),
          const SizedBox(height: 4),
          Text(
            _phaseLabel(phase),
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Color _scoreColor(double score) {
    if (score >= 80) return Colors.green;
    if (score >= 60) return Colors.orange;
    return Colors.red;
  }

  double _phaseProgress(ExercisePhase phase) {
    switch (phase) {
      case ExercisePhase.inactive:
        return 0.0;
      case ExercisePhase.standing:
        return 0.2;
      case ExercisePhase.descending:
        return 0.4;
      case ExercisePhase.bottom:
        return 0.6;
      case ExercisePhase.ascending:
        return 0.8;
      case ExercisePhase.transitioning:
        return 0.5;
    }
  }

  Color _phaseColor(ExercisePhase phase) {
    switch (phase) {
      case ExercisePhase.standing:
        return Colors.green;
      case ExercisePhase.descending:
        return Colors.blue;
      case ExercisePhase.bottom:
        return Colors.orange;
      case ExercisePhase.ascending:
        return Colors.purple;
      default:
        return Colors.grey;
    }
  }

  String _phaseLabel(ExercisePhase phase) {
    switch (phase) {
      case ExercisePhase.inactive:
        return 'En attente...';
      case ExercisePhase.standing:
        return 'Position de départ';
      case ExercisePhase.descending:
        return 'Descente';
      case ExercisePhase.bottom:
        return 'Position basse';
      case ExercisePhase.ascending:
        return 'Montée';
      case ExercisePhase.transitioning:
        return 'Transition';
    }
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color? color;

  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color ?? Theme.of(context).colorScheme.primary),
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
```

---

## 8. Optimisations Mobile

### Export du modèle pour mobile

```python
from ultralytics import YOLO

# Charger le modèle pose
model = YOLO("yolo26n-pose.pt")

# Export pour iOS (CoreML)
model.export(format="coreml", quantize="w8a16")

# Export pour Android (LiteRT)
model.export(format="litert", quantize="w8a32")

# Export cross-platform (NCNN)
model.export(format="ncnn")
```

### Paramètres de performance

| Paramètre | Recommandation | Raison |
|-----------|---------------|--------|
| Résolution entrée | 640×480 | Bon équilibre précision/vitesse |
| FPS cible | ≥ 15 | Suffisant pour le comptage de reps |
| Confiance keypoints | > 0.5 | Filtrer les keypoints peu fiables |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

### Gestion de la mémoire

```dart
class MemoryManager {
  static const int maxKeypointHistory = 30; // 1 seconde à 30fps
  static const int maxScoreHistory = 50;

  /// Limite l'historique pour éviter les fuites mémoire
  static List<T> limitHistory<T>(List<T> history, int maxSize) {
    if (history.length > maxSize) {
      return history.sublist(history.length - maxSize);
    }
    return history;
  }
}
```

---

## 9. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Keypoints tremblants | Bruit de détection | Augmenter `minFramesPerPhase` ou ajouter un filtre passe-bas |
| Faux positifs (reps comptées alors qu'aucun mouvement) | Seuils trop sensibles | Augmenter les seuils ou ajouter un filtre de bruit |
| Reps non comptées | Seuils trop stricts | Diminuer les seuils ou vérifier l'angle de la caméra |
| Détection lente | Modèle trop gros | Utiliser `yolo26n-pose.pt` + quantification INT8 |
| Feedback vocal en retard | Cooldown trop long | Réduire `_feedbackCooldown` |
| Personne non détectée | Éclairage/angle | Améliorer l'éclairage, positionner la caméra face à l'utilisateur |
| Modèle ne charge pas | Format incorrect | Vérifier l'export avec le bon format pour la plateforme |
| Erreur Platform Channel | Mauvaise config native | Vérifier AppDelegate (iOS) / MainActivity (Android) |

### Commandes de debug

```dart
// Activer les logs de debug
import 'package:flutter/foundation.dart';

if (kDebugMode) {
  debugPrint('Angle genou: $kneeAngle');
  debugPrint('Phase: $currentPhase');
  debugPrint('Score: $score');
}

// Afficher les keypoints sur l'image
if (kDebugMode) {
  for (final kp in keypoints) {
    debugPrint('KP[${kp.index}] ${kp.name}: '
        'x=${kp.x.toStringAsFixed(1)}, '
        'y=${kp.y.toStringAsFixed(1)}, '
        'conf=${kp.confidence.toStringAsFixed(2)}');
  }
}
```

---

## Références

- [Documentation YOLO26-pose Ultralytics](https://docs.ultralytics.com/fr/tasks/pose/)
- [Keypoints COCO](https://cocodataset.org/#keypoints-eval)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
