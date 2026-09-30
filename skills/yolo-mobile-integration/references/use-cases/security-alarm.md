# Référence Complète : Alarme de Sécurité avec YOLO26

> Guide complet pour l'intégration d'un système d'alarme de sécurité dans une application mobile Flutter utilisant YOLO26 pour la détection d'intrusion, la surveillance de zones et le déclenchement d'alertes.

---

## QUAND UTILISER CETTE RÉFÉRENCE

**Utilisez cette référence quand :**
- L'utilisateur veut créer une application de **sécurité / surveillance / alarme**
- Le projet nécessite la **détection d'intrusion** en temps réel
- Il faut des **zones de détection** personnalisables (polygones, rectangles)
- L'application doit **déclencher des alertes** basées sur des objets détectés
- Le modèle **YOLO26** est utilisé pour la détection de personnes, véhicules, etc.
- L'utilisateur mentionne : sécurité, alarme, intrusion, surveillance, caméra, gardien, monitoring

**Ne pas utiliser cette référence pour :**
- Comptage d'objets (utiliser `object-counting.md`)
- Analyse de heatmap (utiliser `heatmap-analytics.md`)
- Floutage de personnes (utiliser `object-blurring.md`)

---

## Table des matières

1. [Architecture du Système de Sécurité](#1-architecture-du-système-de-sécurité)
2. [Détection de Mouvement par Zones](#2-détection-de-mouvement-par-zones)
3. [Déclenchement d'Alertes](#3-déclenchement-dalertes)
4. [Système de Notification](#4-système-de-notification)
5. [Journalisation des Événements](#5-journalisation-des-événements)
6. [Exemple Complet Flutter](#6-exemple-complet-flutter)
7. [Optimisations Mobile](#7-optimisations-mobile)
8. [Dépannage](#8-dépannage)

---

## 1. Architecture du Système de Sécurité

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────┐
│                  Application Flutter                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Camera  │→ │  YOLO26  │→ │  Zone Manager    │  │
│  │  Stream  │  │  Detect  │  │  (polygones)     │  │
│  └──────────┘  └──────────┘  └────────┬──────────┘  │
│                                       │              │
│  ┌──────────┐  ┌──────────┐  ┌───────▼──────────┐  │
│  │  Alert   │← │  Event   │← │  Alert Engine    │  │
│  │  UI      │  │  Logger  │  │  (règles + cooldown)│ │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│         │              │                             │
│  ┌──────▼──────┐  ┌───▼──────────┐                  │
│  │ Notification│  │  Local DB    │                  │
│  │ Service     │  │  (events)    │                  │
│  └─────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────┘
```

### Modèles YOLO26 recommandés

| Modèle | Fichier | Usage | Latence mobile |
|--------|---------|-------|----------------|
| YOLO26n | `yolo26n.pt` | Temps réel, CPU | ~15ms |
| YOLO26s | `yolo26s.pt` | Précision + vitesse | ~30ms |
| YOLO26m | `yolo26m.pt` | Haute précision | ~55ms |

### Classes COCO pertinentes pour la sécurité

| ID Classe | Nom | Usage sécurité |
|-----------|-----|----------------|
| 0 | `person` | **Intrusion humaine** |
| 1 | `bicycle` | Vélo suspect |
| 2 | `car` | Véhicule non autorisé |
| 3 | `motorcycle` | Moto suspecte |
| 5 | `bus` | Gros véhicule |
| 7 | `truck` | Camion suspect |
| 9 | `traffic light` | État du feu (contexte) |

---

## 2. Détection de Mouvement par Zones

### 2.1 Définition des zones de détection

Les zones sont définies comme des polygones normalisés (coordonnées 0.0-1.0) superposés au flux vidéo.

```dart
import 'dart:math';

/// Type de zone de détection
enum ZoneType {
  intrusion,      // Zone interdite — toute personne déclenche une alerte
  passage,        // Zone de passage — alerte si présence prolongée
  surveillance,   // Zone surveillée — logging seulement
  interdite,      // Zone totalement interdite
}

/// Niveau de sécurité de la zone
enum SecurityLevel {
  low,      // Notification seulement
  medium,   // Notification + son
  high,     // Notification + son + enregistrement
  critical, // Notification + son + enregistrement + appel
}

/// Point normalisé (0.0 - 1.0)
class NormalizedPoint {
  final double x;
  final double y;

  const NormalizedPoint(this.x, this.y);

  Offset toOffset(Size size) => Offset(x * size.width, y * size.height);
}

/// Zone de détection polygonale
class DetectionZone {
  final String id;
  final String name;
  final List<NormalizedPoint> polygon;
  final ZoneType type;
  final SecurityLevel level;
  final List<String> triggerClasses; // Classes qui déclenchent l'alerte
  final Duration minDwellTime;       // Temps minimum dans la zone pour alerter
  final bool isActive;

  const DetectionZone({
    required this.id,
    required this.name,
    required this.polygon,
    required this.type,
    required this.level,
    required this.triggerClasses,
    this.minDwellTime = const Duration(seconds: 2),
    this.isActive = true,
  });

  /// Vérifie si un point est à l'intérieur du polygone (algorithme ray-casting)
  bool containsPoint(NormalizedPoint point) {
    if (polygon.length < 3) return false;

    bool inside = false;
    int j = polygon.length - 1;

    for (int i = 0; i < polygon.length; i++) {
      final pi = polygon[i];
      final pj = polygon[j];

      if (((pi.y > point.y) != (pj.y > point.y)) &&
          (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)) {
        inside = !inside;
      }
      j = i;
    }
    return inside;
  }

  /// Vérifie si un centre de boîte de détection est dans la zone
  bool containsDetection(double centerX, double centerY) {
    return containsPoint(NormalizedPoint(centerX, centerY));
  }

  /// Calcule l'aire relative du polygone (pour debug/validation)
  double get area {
    if (polygon.length < 3) return 0;
    double sum = 0;
    for (int i = 0; i < polygon.length; i++) {
      final j = (i + 1) % polygon.length;
      sum += polygon[i].x * polygon[j].y;
      sum -= polygon[j].x * polygon[i].y;
    }
    return (sum / 2).abs();
  }
}
```

### 2.2 Gestionnaire de zones

```dart
class ZoneManager {
  final List<DetectionZone> _zones = [];
  final Map<String, DateTime> _zoneEntryTimes = {};

  List<DetectionZone> get zones => List.unmodifiable(_zones);

  void addZone(DetectionZone zone) {
    _zones.add(zone);
  }

  void removeZone(String zoneId) {
    _zones.removeWhere((z) => z.id == zoneId);
    _zoneEntryTimes.remove(zoneId);
  }

  void clearZones() {
    _zones.clear();
    _zoneEntryTimes.clear();
  }

  /// Détecte quelles zones sont traversées par une détection
  /// Retourne les zones actives où le centre de la boîte est à l'intérieur
  List<DetectionZone> checkDetections(List<Detection> detections) {
    final triggeredZones = <DetectionZone>[];

    for (final zone in _zones) {
      if (!zone.isActive) continue;

      for (final det in detections) {
        // Vérifier si la classe de la détection est dans les déclencheurs
        if (!zone.triggerClasses.contains(det.label)) continue;

        // Calculer le centre de la boîte (normalisé)
        final centerX = det.x + det.width / 2;
        final centerY = det.y + det.height / 2;

        if (zone.containsPoint(NormalizedPoint(centerX, centerY))) {
          triggeredZones.add(zone);
          break; // Une seule détection suffit par zone
        }
      }
    }

    return triggeredZones;
  }

  /// Met à jour les temps de présence et retourne les zones où le temps
  // minimum est dépassé
  List<DetectionZone> updateDwellTimes(
    List<DetectionZone> currentlyTriggered,
    DateTime now,
  ) {
    final alertedZones = <DetectionZone>[];

    for (final zone in _zones) {
      if (currentlyTriggered.contains(zone)) {
        // L'objet est dans la zone
        if (!_zoneEntryTimes.containsKey(zone.id)) {
          _zoneEntryTimes[zone.id] = now;
        } else {
          final entryTime = _zoneEntryTimes[zone.id]!;
          final dwellTime = now.difference(entryTime);
          if (dwellTime >= zone.minDwellTime) {
            alertedZones.add(zone);
          }
        }
      } else {
        // L'objet a quitté la zone — réinitialiser
        _zoneEntryTimes.remove(zone.id);
      }
    }

    return alertedZones;
  }
}
```

### 2.3 Exemples de zones prédéfinies

```dart
class SecurityZones {
  /// Zone d'entrée principale (porte d'entrée)
  static DetectionZone entranceZone() => DetectionZone(
        id: 'entrance',
        name: 'Entrée principale',
        polygon: [
          NormalizedPoint(0.3, 0.7),
          NormalizedPoint(0.7, 0.7),
          NormalizedPoint(0.8, 1.0),
          NormalizedPoint(0.2, 1.0),
        ],
        type: ZoneType.intrusion,
        level: SecurityLevel.high,
        triggerClasses: ['person'],
        minDwellTime: const Duration(seconds: 1),
      );

  /// Zone de passage (couloir)
  static DetectionZone corridorZone() => DetectionZone(
        id: 'corridor',
        name: 'Couloir',
        polygon: [
          NormalizedPoint(0.0, 0.3),
          NormalizedPoint(0.5, 0.3),
          NormalizedPoint(0.5, 0.7),
          NormalizedPoint(0.0, 0.7),
        ],
        type: ZoneType.passage,
        level: SecurityLevel.medium,
        triggerClasses: ['person'],
        minDwellTime: const Duration(seconds: 5),
      );

  /// Zone de stationnement (véhicules)
  static DetectionZone parkingZone() => DetectionZone(
        id: 'parking',
        name: 'Parking',
        polygon: [
          NormalizedPoint(0.5, 0.0),
          NormalizedPoint(1.0, 0.0),
          NormalizedPoint(1.0, 0.5),
          NormalizedPoint(0.5, 0.5),
        ],
        type: ZoneType.surveillance,
        level: SecurityLevel.low,
        triggerClasses: ['car', 'truck', 'motorcycle', 'bus'],
        minDwellTime: const Duration(seconds: 10),
      );

  /// Zone interdite (périmètre sensible)
  static DetectionZone restrictedZone() => DetectionZone(
        id: 'restricted',
        name: 'Zone restreinte',
        polygon: [
          NormalizedPoint(0.1, 0.1),
          NormalizedPoint(0.4, 0.1),
          NormalizedPoint(0.4, 0.4),
          NormalizedPoint(0.1, 0.4),
        ],
        type: ZoneType.interdite,
        level: SecurityLevel.critical,
        triggerClasses: ['person', 'bicycle', 'motorcycle'],
        minDwellTime: Duration.zero, // Alerte immédiate
      );
}
```

---

## 3. Déclenchement d'Alertes

### 3.1 Moteur d'alertes basé sur des règles

```dart
/// Type d'alerte
enum AlertType {
  intrusion,
  presence,
  vehicule,
  mouvement,
  inconnu,
}

/// Alerte de sécurité
class SecurityAlert {
  final String id;
  final AlertType type;
  final SecurityLevel level;
  final String zoneId;
  final String zoneName;
  final String detectionLabel;
  final double confidence;
  final DateTime timestamp;
  final String? snapshotPath;

  const SecurityAlert({
    required this.id,
    required this.type,
    required this.level,
    required this.zoneId,
    required this.zoneName,
    required this.detectionLabel,
    required this.confidence,
    required this.timestamp,
    this.snapshotPath,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'type': type.name,
        'level': level.name,
        'zoneId': zoneId,
        'zoneName': zoneName,
        'detectionLabel': detectionLabel,
        'confidence': confidence,
        'timestamp': timestamp.toIso8601String(),
        'snapshotPath': snapshotPath,
      };
}

/// Règle de déclenchement d'alerte
class AlertRule {
  final String id;
  final String name;
  final AlertType type;
  final SecurityLevel minLevel;
  final List<String> triggerClasses;
  final double minConfidence;
  final Duration cooldown; // Éviter les alertes répétées
  final bool requireDwellTime;

  const AlertRule({
    required this.id,
    required this.name,
    required this.type,
    required this.minLevel,
    required this.triggerClasses,
    this.minConfidence = 0.5,
    this.cooldown = const Duration(seconds: 10),
    this.requireDwellTime = true,
  });
}

/// Moteur d'alertes
class AlertEngine {
  final List<AlertRule> _rules = [];
  final Map<String, DateTime> _lastAlertTimes = {};
  final List<SecurityAlert> _alerts = [];

  List<SecurityAlert> get alerts => List.unmodifiable(_alerts);

  void addRule(AlertRule rule) => _rules.add(rule);

  void removeRule(String ruleId) => _rules.removeWhere((r) => r.id == ruleId);

  /// Traite les détections et déclenche les alertes appropriées
  List<SecurityAlert> processDetections(
    List<Detection> detections,
    List<DetectionZone> triggeredZones,
    List<DetectionZone> dwellTimeZones,
  ) {
    final newAlerts = <SecurityAlert>[];
    final now = DateTime.now();

    for (final rule in _rules) {
      // Vérifier le cooldown
      final lastAlert = _lastAlertTimes[rule.id];
      if (lastAlert != null && now.difference(lastAlert) < rule.cooldown) {
        continue;
      }

      // Vérifier chaque détection
      for (final det in detections) {
        if (!rule.triggerClasses.contains(det.label)) continue;
        if (det.confidence < rule.minConfidence) continue;

        // Trouver la zone correspondante
        final zone = triggeredZones.firstWhere(
          (z) => z.triggerClasses.contains(det.label),
          orElse: () => DetectionZone(
            id: 'unknown',
            name: 'Hors zone',
            polygon: [],
            type: ZoneType.surveillance,
            level: SecurityLevel.low,
            triggerClasses: [],
          ),
        );

        // Vérifier le temps de présence si requis
        if (rule.requireDwellTime && !dwellTimeZones.contains(zone)) {
          continue;
        }

        // Vérifier le niveau de sécurité
        if (zone.level.index < rule.minLevel.index) continue;

        // Créer l'alerte
        final alert = SecurityAlert(
          id: '${rule.id}_${now.millisecondsSinceEpoch}',
          type: rule.type,
          level: zone.level,
          zoneId: zone.id,
          zoneName: zone.name,
          detectionLabel: det.label,
          confidence: det.confidence,
          timestamp: now,
        );

        newAlerts.add(alert);
        _alerts.add(alert);
        _lastAlertTimes[rule.id] = now;

        break; // Une alerte par règle par cycle
      }
    }

    // Limiter l'historique des alertes
    if (_alerts.length > 1000) {
      _alerts.removeRange(0, _alerts.length - 1000);
    }

    return newAlerts;
  }

  void clearAlerts() {
    _alerts.clear();
    _lastAlertTimes.clear();
  }
}
```

### 3.2 Règles prédéfinies

```dart
class SecurityRules {
  static AlertRule intrusionRule() => AlertRule(
        id: 'intrusion_person',
        name: 'Intrusion personne',
        type: AlertType.intrusion,
        minLevel: SecurityLevel.medium,
        triggerClasses: ['person'],
        minConfidence: 0.6,
        cooldown: const Duration(seconds: 15),
        requireDwellTime: true,
      );

  static AlertRule vehiculeRule() => AlertRule(
        id: 'vehicule_non_autorise',
        name: 'Véhicule non autorisé',
        type: AlertType.vehicule,
        minLevel: SecurityLevel.low,
        triggerClasses: ['car', 'truck', 'motorcycle', 'bus'],
        minConfidence: 0.5,
        cooldown: const Duration(seconds: 30),
        requireDwellTime: true,
      );

  static AlertRule presenceProlongeeRule() => AlertRule(
        id: 'presence_prolongee',
        name: 'Présence prolongée',
        type: AlertType.presence,
        minLevel: SecurityLevel.low,
        triggerClasses: ['person'],
        minConfidence: 0.4,
        cooldown: const Duration(minutes: 1),
        requireDwellTime: true,
      );
}
```

---

## 4. Système de Notification

### 4.1 Service de notification local

```dart
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class SecurityNotificationService {
  final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();
  bool _isInitialized = false;

  static const String _channelId = 'security_alerts';
  static const String _channelName = 'Alertes de sécurité';
  static const String _channelDescription =
      'Notifications pour les alertes de sécurité';

  Future<void> initialize() async {
    if (_isInitialized) return;

    const androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    const settings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );

    await _notifications.initialize(settings);

    // Créer le canal Android
    const androidChannel = AndroidNotificationChannel(
      _channelId,
      _channelName,
      description: _channelDescription,
      importance: Importance.high,
      enableVibration: true,
      playSound: true,
    );

    await _notifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(androidChannel);

    _isInitialized = true;
  }

  /// Affiche une notification d'alerte de sécurité
  Future<void> showSecurityAlert(SecurityAlert alert) async {
    if (!_isInitialized) await initialize();

    final androidDetails = AndroidNotificationDetails(
      _channelId,
      _channelName,
      channelDescription: _channelDescription,
      importance: Importance.high,
      priority: Priority.high,
      enableVibration: true,
      playSound: true,
      styleInformation: BigTextStyleInformation(
        _formatAlertBody(alert),
        contentTitle: _formatAlertTitle(alert),
      ),
      category: AndroidNotificationCategory.alarm,
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
      interruptionLevel: InterruptionLevel.timeSensitive,
    );

    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      alert.id.hashCode,
      _formatAlertTitle(alert),
      _formatAlertBody(alert),
      details,
    );
  }

  /// Notification de niveau critique (avec son d'alarme)
  Future<void> showCriticalAlert(SecurityAlert alert) async {
    if (!_isInitialized) await initialize();

    final androidDetails = AndroidNotificationDetails(
      'critical_alerts',
      'Alertes critiques',
      channelDescription: 'Alertes de sécurité critiques',
      importance: Importance.max,
      priority: Priority.max,
      enableVibration: true,
      playSound: true,
      sound: const RawResourceAndroidNotificationSound('alarm_sound'),
      category: AndroidNotificationCategory.alarm,
      fullScreenIntent: true, // Affiche même si l'appareil est verrouillé
    );

    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
      sound: 'alarm_sound.aiff',
      interruptionLevel: InterruptionLevel.critical,
    );

    final details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _notifications.show(
      alert.id.hashCode,
      '🚨 ${_formatAlertTitle(alert)}',
      _formatAlertBody(alert),
      details,
    );
  }

  String _formatAlertTitle(SecurityAlert alert) {
    switch (alert.type) {
      case AlertType.intrusion:
        return 'INTRUSION DÉTECTÉE';
      case AlertType.vehicule:
        return 'Véhicule non autorisé';
      case AlertType.presence:
        return 'Présence prolongée';
      case AlertType.mouvement:
        return 'Mouvement détecté';
      case AlertType.inconnu:
        return 'Alerte de sécurité';
    }
  }

  String _formatAlertBody(SecurityAlert alert) {
    final timeStr =
        '${alert.timestamp.hour.toString().padLeft(2, '0')}:'
        '${alert.timestamp.minute.toString().padLeft(2, '0')}:'
        '${alert.timestamp.second.toString().padLeft(2, '0')}';
    return '${alert.detectionLabel} détecté dans "${alert.zoneName}" '
        'à $timeStr (confiance: ${(alert.confidence * 100).toStringAsFixed(0)}%)';
  }

  Future<void> cancelAll() async {
    await _notifications.cancelAll();
  }
}
```

### 4.2 Service de notification push (Firebase)

```dart
import 'package:firebase_messaging/firebase_messaging.dart';

class PushNotificationService {
  final FirebaseMessaging _messaging = FirebaseMessaging.instance;

  Future<void> initialize() async {
    // Demander les permissions
    await _messaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
      criticalAlert: true,
    );

    // Configurer le handler en arrière-plan
    FirebaseMessaging.onBackgroundMessage(_backgroundHandler);

    // Configurer le handler au premier plan
    FirebaseMessaging.onMessage.listen(_foregroundHandler);

    // Récupérer le token FCM
    final token = await _messaging.getToken();
    debugPrint('FCM Token: $token');
  }

  void _foregroundHandler(RemoteMessage message) {
    final notification = message.notification;
    if (notification != null) {
      debugPrint('Notification reçue: ${notification.title}');
    }
  }

  static Future<void> _backgroundHandler(RemoteMessage message) async {
    debugPrint('Notification en arrière-plan: ${message.messageId}');
  }

  /// Envoyer une alerte push à un appareil spécifique
  Future<void> sendAlertToDevice(
    String deviceToken,
    SecurityAlert alert,
  ) async {
    // Ceci doit être fait côté serveur (Firebase Cloud Functions)
    // Exemple de structure de message :
    final message = {
      'to': deviceToken,
      'notification': {
        'title': '🚨 ${alert.type.name.toUpperCase()}',
        'body': '${alert.detectionLabel} dans ${alert.zoneName}',
      },
      'data': {
        'alert_id': alert.id,
        'zone_id': alert.zoneId,
        'type': alert.type.name,
        'level': alert.level.name,
        'timestamp': alert.timestamp.toIso8601String(),
      },
      'android': {
        'priority': 'high',
        'notification': {
          'channel_id': 'security_alerts',
          'sound': 'alarm_sound',
        },
      },
      'apns': {
        'headers': {
          'apns-priority': '10',
        },
        'payload': {
          'aps': {
            'alert': {
              'title': '🚨 ${alert.type.name.toUpperCase()}',
              'body': '${alert.detectionLabel} dans ${alert.zoneName}',
            },
            'sound': 'alarm_sound.aiff',
            'badge': 1,
          },
        },
      },
    };

    // Envoyer via l'API FCM (côté serveur)
    debugPrint('Message FCM préparé: $message');
  }
}
```

---

## 5. Journalisation des Événements

### 5.1 Modèle d'événement

```dart
/// Événement de sécurité journalisé
class SecurityEvent {
  final String id;
  final DateTime timestamp;
  final String type; // 'alert', 'detection', 'zone_entry', 'zone_exit', 'system'
  final String? zoneId;
  final String? zoneName;
  final String? detectionLabel;
  final double? confidence;
  final String? snapshotPath;
  final Map<String, dynamic>? metadata;

  const SecurityEvent({
    required this.id,
    required this.timestamp,
    required this.type,
    this.zoneId,
    this.zoneName,
    this.detectionLabel,
    this.confidence,
    this.snapshotPath,
    this.metadata,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'timestamp': timestamp.toIso8601String(),
        'type': type,
        'zoneId': zoneId,
        'zoneName': zoneName,
        'detectionLabel': detectionLabel,
        'confidence': confidence,
        'snapshotPath': snapshotPath,
        'metadata': metadata,
      };

  factory SecurityEvent.fromJson(Map<String, dynamic> json) => SecurityEvent(
        id: json['id'],
        timestamp: DateTime.parse(json['timestamp']),
        type: json['type'],
        zoneId: json['zoneId'],
        zoneName: json['zoneName'],
        detectionLabel: json['detectionLabel'],
        confidence: json['confidence']?.toDouble(),
        snapshotPath: json['snapshotPath'],
        metadata: json['metadata'],
      );
}
```

### 5.2 Service de journalisation

```dart
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class EventLogger {
  final List<SecurityEvent> _events = [];
  static const int _maxEvents = 5000;
  static const int _eventsPerFile = 500;

  List<SecurityEvent> get events => List.unmodifiable(_events);

  /// Journalise un événement
  void log(SecurityEvent event) {
    _events.add(event);

    // Limiter la mémoire
    if (_events.length > _maxEvents) {
      _events.removeRange(0, _events.length - _maxEvents);
    }

    // Écrire sur disque de manière asynchrone
    _persistEvent(event);
  }

  /// Journalise une alerte
  void logAlert(SecurityAlert alert, {String? snapshotPath}) {
    log(SecurityEvent(
      id: 'alert_${alert.id}',
      timestamp: alert.timestamp,
      type: 'alert',
      zoneId: alert.zoneId,
      zoneName: alert.zoneName,
      detectionLabel: alert.detectionLabel,
      confidence: alert.confidence,
      snapshotPath: snapshotPath,
      metadata: {
        'alert_type': alert.type.name,
        'alert_level': alert.level.name,
      },
    ));
  }

  /// Journalise une détection simple
  void logDetection(Detection det, {String? zoneId, String? zoneName}) {
    log(SecurityEvent(
      id: 'det_${DateTime.now().millisecondsSinceEpoch}',
      timestamp: DateTime.now(),
      type: 'detection',
      zoneId: zoneId,
      zoneName: zoneName,
      detectionLabel: det.label,
      confidence: det.confidence,
    ));
  }

  /// Récupère les événements d'une plage horaire
  List<SecurityEvent> getEvents({
    DateTime? from,
    DateTime? to,
    String? type,
    String? zoneId,
  }) {
    return _events.where((e) {
      if (from != null && e.timestamp.isBefore(from)) return false;
      if (to != null && e.timestamp.isAfter(to)) return false;
      if (type != null && e.type != type) return false;
      if (zoneId != null && e.zoneId != zoneId) return false;
      return true;
    }).toList();
  }

  /// Exporte les événements en JSON
  Future<String> exportToJson() async {
    final data = _events.map((e) => e.toJson()).toList();
    return jsonEncode(data);
  }

  /// Persiste un événement sur disque
  Future<void> _persistEvent(SecurityEvent event) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final logDir = Directory('${dir.path}/security_logs');
      if (!await logDir.exists()) {
        await logDir.create(recursive: true);
      }

      // Nom de fichier basé sur la date
      final dateStr = '${event.timestamp.year}-'
          '${event.timestamp.month.toString().padLeft(2, '0')}-'
          '${event.timestamp.day.toString().padLeft(2, '0')}';
      final file = File('${logDir.path}/events_$dateStr.jsonl');

      // Ajouter en mode append (JSON Lines)
      await file.writeAsString(
        '${jsonEncode(event.toJson())}\n',
        mode: FileMode.append,
      );
    } catch (e) {
      debugPrint('Erreur de journalisation: $e');
    }
  }

  /// Charge les événements d'une date spécifique
  Future<List<SecurityEvent>> loadEvents(DateTime date) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final dateStr = '${date.year}-'
          '${date.month.toString().padLeft(2, '0')}-'
          '${date.day.toString().padLeft(2, '0')}';
      final file = File('${dir.path}/security_logs/events_$dateStr.jsonl');

      if (!await file.exists()) return [];

      final lines = await file.readAsLines();
      return lines
          .where((l) => l.trim().isNotEmpty)
          .map((l) => SecurityEvent.fromJson(jsonDecode(l)))
          .toList();
    } catch (e) {
      debugPrint('Erreur de chargement: $e');
      return [];
    }
  }

  void clear() => _events.clear();
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
│   ├── detection_zone.dart
│   └── security_alert.dart
├── services/
│   ├── yolo_detection_service.dart
│   ├── zone_manager.dart
│   ├── alert_engine.dart
│   ├── notification_service.dart
│   └── event_logger.dart
├── screens/
│   └── security_screen.dart
└── widgets/
    ├── zone_overlay_painter.dart
    ├── alert_banner.dart
    └── event_list_widget.dart
```

### `models/detection.dart`

```dart
/// Détection YOLO26
class Detection {
  final String label;
  final double confidence;
  final double x;      // normalisé 0-1
  final double y;      // normalisé 0-1
  final double width;  // normalisé 0-1
  final double height; // normalisé 0-1

  const Detection({
    required this.label,
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  double get centerX => x + width / 2;
  double get centerY => y + height / 2;

  factory Detection.fromNative(Map<dynamic, dynamic> map) => Detection(
        label: map['label'] as String,
        confidence: (map['confidence'] as num).toDouble(),
        x: (map['x'] as num).toDouble(),
        y: (map['y'] as num).toDouble(),
        width: (map['width'] as num).toDouble(),
        height: (map['height'] as num).toDouble(),
      );
}
```

### `services/yolo_detection_service.dart`

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart';
import '../models/detection.dart';

class YoloDetectionService {
  static const MethodChannel _channel = MethodChannel('com.app/yolo_detect');
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    await _channel.invokeMethod('initializeModel', {
      'model': 'yolo26n',
      'confidence': 0.4,
    });
    _isInitialized = true;
  }

  Future<List<Detection>> detect(Uint8List imageBytes) async {
    final result = await _channel.invokeMethod('detect', {
      'image': imageBytes,
    });

    final detections = <Detection>[];
    for (final item in result['detections']) {
      detections.add(Detection.fromNative(item));
    }
    return detections;
  }

  void dispose() {
    _channel.invokeMethod('disposeModel');
    _isInitialized = false;
  }
}
```

### `screens/security_screen.dart`

```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../models/detection.dart';
import '../models/detection_zone.dart';
import '../models/security_alert.dart';
import '../services/yolo_detection_service.dart';
import '../services/zone_manager.dart';
import '../services/alert_engine.dart';
import '../services/notification_service.dart';
import '../services/event_logger.dart';
import '../widgets/zone_overlay_painter.dart';
import '../widgets/alert_banner.dart';

class SecurityScreen extends StatefulWidget {
  const SecurityScreen({super.key});

  @override
  State<SecurityScreen> createState() => _SecurityScreenState();
}

class _SecurityScreenState extends State<SecurityScreen> {
  CameraController? _cameraController;
  final YoloDetectionService _yoloService = YoloDetectionService();
  final ZoneManager _zoneManager = ZoneManager();
  final AlertEngine _alertEngine = AlertEngine();
  final SecurityNotificationService _notificationService =
      SecurityNotificationService();
  final EventLogger _eventLogger = EventLogger();

  List<Detection> _detections = [];
  List<SecurityAlert> _recentAlerts = [];
  bool _isProcessing = false;
  bool _isInitialized = false;
  bool _isArmed = true; // Système armé/désarmé

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    // Initialiser les services
    await _yoloService.initialize();
    await _notificationService.initialize();

    // Configurer les zones
    _zoneManager.addZone(SecurityZones.entranceZone());
    _zoneManager.addZone(SecurityZones.corridorZone());
    _zoneManager.addZone(SecurityZones.parkingZone());
    _zoneManager.addZone(SecurityZones.restrictedZone());

    // Configurer les règles d'alerte
    _alertEngine.addRule(SecurityRules.intrusionRule());
    _alertEngine.addRule(SecurityRules.vehiculeRule());
    _alertEngine.addRule(SecurityRules.presenceProlongeeRule());

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
    if (_isProcessing || !_isArmed) return;
    _isProcessing = true;

    try {
      final bytes = _imageToBytes(image);
      final detections = await _yoloService.detect(bytes);

      // Filtrer par classes de sécurité
      final securityDetections = detections
          .where((d) => ['person', 'car', 'truck', 'motorcycle', 'bus', 'bicycle']
              .contains(d.label))
          .toList();

      // Vérifier les zones
      final triggeredZones = _zoneManager.checkDetections(securityDetections);
      final dwellTimeZones = _zoneManager.updateDwellTimes(
        triggeredZones,
        DateTime.now(),
      );

      // Traiter les alertes
      final newAlerts = _alertEngine.processDetections(
        securityDetections,
        triggeredZones,
        dwellTimeZones,
      );

      // Envoyer les notifications
      for (final alert in newAlerts) {
        _eventLogger.logAlert(alert);

        if (alert.level == SecurityLevel.critical) {
          await _notificationService.showCriticalAlert(alert);
        } else {
          await _notificationService.showSecurityAlert(alert);
        }
      }

      setState(() {
        _detections = securityDetections;
        _recentAlerts = _alertEngine.alerts.reversed.take(10).toList();
      });
    } catch (e) {
      debugPrint('Erreur de traitement: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Uint8List _imageToBytes(CameraImage image) {
    final buffer = image.planes[0].bytes;
    return buffer;
  }

  void _toggleArmed() {
    setState(() => _isArmed = !_isArmed);
    if (!_isArmed) {
      _eventLogger.log(SecurityEvent(
        id: 'system_${DateTime.now().millisecondsSinceEpoch}',
        timestamp: DateTime.now(),
        type: 'system',
        metadata: {'action': 'disarmed'},
      ));
    } else {
      _eventLogger.log(SecurityEvent(
        id: 'system_${DateTime.now().millisecondsSinceEpoch}',
        timestamp: DateTime.now(),
        type: 'system',
        metadata: {'action': 'armed'},
      ));
    }
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
        title: const Text('Sécurité'),
        actions: [
          // Bouton armer/désarmer
          IconButton(
            icon: Icon(
              _isArmed ? Icons.shield : Icons.shield_outlined,
              color: _isArmed ? Colors.green : Colors.red,
            ),
            onPressed: _toggleArmed,
          ),
          // Indicateur d'état
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Chip(
              label: Text(_isArmed ? 'ARMÉ' : 'DÉSARMÉ'),
              backgroundColor: _isArmed ? Colors.green[100] : Colors.red[100],
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Bannière d'alerte
          if (_recentAlerts.isNotEmpty)
            AlertBanner(alert: _recentAlerts.first),

          // Zone caméra avec overlay
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                CustomPaint(
                  painter: ZoneOverlayPainter(
                    zones: _zoneManager.zones,
                    detections: _detections,
                    imageSize: Size(
                      _cameraController!.value.previewSize!.height,
                      _cameraController!.value.previewSize!.width,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Liste des événements récents
          Expanded(
            flex: 1,
            child: EventListWidget(events: _eventLogger.events.reversed.take(20).toList()),
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

### `widgets/zone_overlay_painter.dart`

```dart
import 'package:flutter/material.dart';
import '../models/detection.dart';
import '../models/detection_zone.dart';

class ZoneOverlayPainter extends CustomPainter {
  final List<DetectionZone> zones;
  final List<Detection> detections;
  final Size imageSize;

  ZoneOverlayPainter({
    required this.zones,
    required this.detections,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Dessiner les zones
    for (final zone in zones) {
      _drawZone(canvas, size, zone);
    }

    // Dessiner les détections
    for (final det in detections) {
      _drawDetection(canvas, size, det);
    }
  }

  void _drawZone(Canvas canvas, Size size, DetectionZone zone) {
    if (zone.polygon.length < 3) return;

    final path = Path();
    final first = zone.polygon.first.toOffset(size);
    path.moveTo(first.dx, first.dy);

    for (int i = 1; i < zone.polygon.length; i++) {
      final pt = zone.polygon[i].toOffset(size);
      path.lineTo(pt.dx, pt.dy);
    }
    path.close();

    // Remplissage semi-transparent
    final fillColor = _zoneColor(zone.level).withOpacity(0.15);
    canvas.drawPath(path, Paint()..color = fillColor);

    // Bordure
    final borderColor = _zoneColor(zone.level).withOpacity(0.6);
    canvas.drawPath(
      path,
      Paint()
        ..color = borderColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Nom de la zone
    final textPainter = TextPainter(
      text: TextSpan(
        text: zone.name,
        style: TextStyle(
          color: _zoneColor(zone.level),
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black.withOpacity(0.5),
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    final labelPos = zone.polygon.first.toOffset(size);
    textPainter.paint(canvas, Offset(labelPos.dx + 4, labelPos.dy + 4));
  }

  void _drawDetection(Canvas canvas, Size size, Detection det) {
    final rect = Rect.fromLTWH(
      det.x * size.width,
      det.y * size.height,
      det.width * size.width,
      det.height * size.height,
    );

    // Boîte de détection
    final color = _detectionColor(det.label);
    canvas.drawRect(
      rect,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Label
    final labelText = '${det.label} ${(det.confidence * 100).toStringAsFixed(0)}%';
    final textPainter = TextPainter(
      text: TextSpan(
        text: labelText,
        style: TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
          backgroundColor: color,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(rect.left, rect.top - 16));
  }

  Color _zoneColor(SecurityLevel level) {
    switch (level) {
      case SecurityLevel.low:
        return Colors.blue;
      case SecurityLevel.medium:
        return Colors.orange;
      case SecurityLevel.high:
        return Colors.red;
      case SecurityLevel.critical:
        return Colors.purple;
    }
  }

  Color _detectionColor(String label) {
    switch (label) {
      case 'person':
        return Colors.red;
      case 'car':
        return Colors.blue;
      case 'truck':
        return Colors.indigo;
      case 'motorcycle':
        return Colors.orange;
      case 'bus':
        return Colors.purple;
      case 'bicycle':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  @override
  bool shouldRepaint(covariant ZoneOverlayPainter oldDelegate) => true;
}
```

### `widgets/alert_banner.dart`

```dart
import 'package:flutter/material.dart';
import '../models/security_alert.dart';

class AlertBanner extends StatelessWidget {
  final SecurityAlert alert;

  const AlertBanner({super.key, required this.alert});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      color: _backgroundColor(alert.level),
      child: Row(
        children: [
          Icon(_icon(alert.type), color: Colors.white, size: 28),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _title(alert),
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                Text(
                  '${alert.zoneName} • ${_formatTime(alert.timestamp)}',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _backgroundColor(SecurityLevel level) {
    switch (level) {
      case SecurityLevel.low:
        return Colors.blue;
      case SecurityLevel.medium:
        return Colors.orange;
      case SecurityLevel.high:
        return Colors.red;
      case SecurityLevel.critical:
        return Colors.purple;
    }
  }

  IconData _icon(AlertType type) {
    switch (type) {
      case AlertType.intrusion:
        return Icons.warning;
      case AlertType.vehicule:
        return Icons.directions_car;
      case AlertType.presence:
        return Icons.person;
      case AlertType.mouvement:
        return Icons.motion_photos_on;
      case AlertType.inconnu:
        return Icons.help;
    }
  }

  String _title(SecurityAlert alert) {
    switch (alert.type) {
      case AlertType.intrusion:
        return 'INTRUSION: ${alert.detectionLabel}';
      case AlertType.vehicule:
        return 'Véhicule: ${alert.detectionLabel}';
      case AlertType.presence:
        return 'Présence: ${alert.detectionLabel}';
      case AlertType.mouvement:
        return 'Mouvement détecté';
      case AlertType.inconnu:
        return 'Alerte: ${alert.detectionLabel}';
    }
  }

  String _formatTime(DateTime dt) {
    return '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}:'
        '${dt.second.toString().padLeft(2, '0')}';
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
| FPS cible | ≥ 10 | Suffisant pour la détection d'intrusion |
| Seuil confiance | 0.4-0.6 | Éviter les faux positifs |
| Cooldown alertes | 10-30s | Éviter le spam de notifications |
| Taille modèle | Nano (n) | CPU mobile, temps réel |
| Quantification | INT8/W8A32 | Réduction taille + vitesse |

### Gestion de la mémoire

```dart
class SecurityMemoryManager {
  static const int maxDetectionHistory = 50;
  static const int maxAlertHistory = 100;
  static const int maxEventHistory = 5000;

  /// Limite l'historique des détections
  static List<T> limitDetections<T>(List<T> history) {
    if (history.length > maxDetectionHistory) {
      return history.sublist(history.length - maxDetectionHistory);
    }
    return history;
  }

  /// Nettoie les anciens fichiers de log
  static Future<void> cleanupOldLogs(Duration maxAge) async {
    // Implémentation : supprimer les fichiers de log plus anciens que maxAge
  }
}
```

---

## 8. Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| Faux positifs fréquents | Seuil de confiance trop bas | Augmenter `minConfidence` à 0.6+ |
| Alertes non déclenchées | Cooldown trop long | Réduire le `cooldown` des règles |
| Zones non précises | Coordonnées normalisées incorrectes | Vérifier le mapping des coordonnées |
| Notifications non reçues | Permissions non accordées | Vérifier les permissions de notification |
| Détection lente | Modèle trop gros | Utiliser `yolo26n.pt` + INT8 |
| Fuite mémoire | Historique non limité | Utiliser `SecurityMemoryManager` |
| Caméra ne s'initialise pas | Permissions manquantes | Vérifier Info.plist / AndroidManifest |
| Alertes critiques non affichées | Canal non configuré | Créer le canal `critical_alerts` |

### Commandes de debug

```dart
if (kDebugMode) {
  debugPrint('Zones actives: ${_zoneManager.zones.length}');
  debugPrint('Détections: ${_detections.length}');
  debugPrint('Alertes: ${_alertEngine.alerts.length}');
  debugPrint('Événements: ${_eventLogger.events.length}');
}
```

---

## Références

- [Documentation YOLO26 Ultralytics](https://docs.ultralytics.com/fr/tasks/detect/)
- [Référence du modèle YOLO26](../yolo26-model.md)
- [Pipeline d'export](../export-pipeline.md)
- [Intégration Flutter](../frameworks/flutter-integration.md)
- [Coach Fitness (exemple similaire)](./fitness-coach.md)
