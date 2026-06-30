#!/usr/bin/env python3
"""
YOLO26 Mobile Project Creator
Scaffolds a new mobile project with YOLO integration.

Usage:
    python create_project.py --framework flutter --task detect --platform both
    python create_project.py --framework react-native --task pose --platform ios
    python create_project.py --framework native-swift --task segment --platform ios
"""

import argparse
import os
import sys
from pathlib import Path


# Template directories
TEMPLATES_DIR = Path(__file__).parent.parent / "assets" / "templates"


def create_flutter_project(
    project_name: str,
    task: str,
    platform: str,
    output_dir: str,
):
    """Scaffold a Flutter project with YOLO integration."""
    print(f"Creation du projet Flutter: {project_name}")
    print(f"Tache: {task} | Plateforme: {platform}")

    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    dirs = [
        "lib/screens",
        "lib/services",
        "lib/models",
        "lib/widgets",
        "lib/providers",
        "assets/models",
        "ios/Runner",
        "android/app/src/main/kotlin",
    ]
    for d in dirs:
        (project_dir / d).mkdir(parents=True, exist_ok=True)

    # Generate pubspec.yaml
    pubspec = f"""name: {project_name}
description: Application mobile YOLO26 - Detection en temps reel
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  camera: ^0.10.0
  image: ^4.0.0
  provider: ^6.0.0
  path_provider: ^2.0.0
  permission_handler: ^10.0.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0

flutter:
  uses-material-design: true
  assets:
    - assets/models/
"""
    (project_dir / "pubspec.yaml").write_text(pubspec)

    # Generate main service file
    task_imports = {
        "detect": "from ultralytics import YOLO",
        "segment": "from ultralytics import YOLO",
        "classify": "from ultralytics import YOLO",
        "pose": "from ultralytics import YOLO",
        "obb": "from ultralytics import YOLO",
    }

    service_code = f'''import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Service d'inférence YOLO26 pour {task}
class YoloService {{
  static const MethodChannel _channel = MethodChannel('yolo_inference');
  bool _isInitialized = false;

  /// Initialise le modèle YOLO26
  Future<void> initialize({{String? modelPath}}) async {{
    if (_isInitialized) return;

    try {{
      await _channel.invokeMethod('initialize', {{
        'task': '{task}',
        'modelPath': modelPath ?? 'assets/models/yolo26n.mlpackage',
      }});
      _isInitialized = true;
    }} on PlatformException catch (e) {{
      throw Exception('Erreur d\\'initialisation YOLO: ${{e.message}}');
    }}
  }}

  /// Lance l'inférence sur une image
  Future<List<Map<String, dynamic>>> detect(Uint8List imageBytes) async {{
    if (!_isInitialized) {{
      throw Exception('Service non initialisé. Appelez initialize() d\\'abord.');
    }}

    try {{
      final results = await _channel.invokeMethod('detect', {{
        'imageBytes': imageBytes,
      }});
      return List<Map<String, dynamic>>.from(results);
    }} on PlatformException catch (e) {{
      throw Exception('Erreur d\\'inférence: ${{e.message}}');
    }}
  }}

  /// Libère les ressources
  Future<void> dispose() async {{
    if (_isInitialized) {{
      await _channel.invokeMethod('dispose');
      _isInitialized = false;
    }}
  }}
}}
'''
    (project_dir / "lib/services/yolo_service.dart").write_text(service_code)

    # Generate detection screen
    screen_code = f'''import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../services/yolo_service.dart';

/// Écran de détection YOLO26 en temps réel
class DetectionScreen extends StatefulWidget {{
  const DetectionScreen({{Key? key}}) : super(key: key);

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}}

class _DetectionScreenState extends State<DetectionScreen> {{
  final YoloService _yoloService = YoloService();
  CameraController? _cameraController;
  List<Map<String, dynamic>> _detections = [];
  bool _isProcessing = false;
  bool _isInitialized = false;

  @override
  void initState() {{
    super.initState();
    _initializeCamera();
  }}

  Future<void> _initializeCamera() async {{
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController!.initialize();
    await _yoloService.initialize();

    setState(() => _isInitialized = true);

    // Start detection loop
    _startDetectionLoop();
  }}

  void _startDetectionLoop() async {{
    while (mounted) {{
      if (_cameraController != null && !_isProcessing) {{
        _isProcessing = true;
        try {{
          final image = await _cameraController!.takePicture();
          final bytes = await image.readAsBytes();
          final results = await _yoloService.detect(bytes);
          if (mounted) {{
            setState(() => _detections = results);
          }}
        }} catch (e) {{
          debugPrint('Erreur de détection: $e');
        }}
        _isProcessing = false;
      }}
      await Future.delayed(const Duration(milliseconds: 33)); // ~30 FPS
    }}
  }}

  @override
  Widget build(BuildContext context) {{
    if (!_isInitialized) {{
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }}

    return Scaffold(
      appBar: AppBar(title: const Text('Détection YOLO26')),
      body: Stack(
        children: [
          CameraPreview(_cameraController!),
          CustomPaint(
            painter: DetectionPainter(_detections),
            size: Size.infinite,
          ),
        ],
      ),
    );
  }}

  @override
  void dispose() {{
    _yoloService.dispose();
    _cameraController?.dispose();
    super.dispose();
  }}
}}

/// Peintre pour dessiner les bounding boxes
class DetectionPainter extends CustomPainter {{
  final List<Map<String, dynamic>> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {{
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (final det in detections) {{
      final box = det['box'];
      if (box == null) continue;

      final rect = Rect.fromLTRB(
        box['x1'] * size.width,
        box['y1'] * size.height,
        box['x2'] * size.width,
        box['y2'] * size.height,
      );

      // Couleur basée sur la classe
      final classId = det['class'] ?? 0;
      final hue = (classId * 30.0) % 360;
      paint.color = HSVColor.fromAHSV(1.0, hue, 0.8, 0.9).toColor();

      canvas.drawRect(rect, paint);

      // Label
      final label = '${{det['name'] ?? "object"}} ${{((det['confidence'] ?? 0) * 100).toStringAsFixed(0)}}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(color: Colors.white, fontSize: 12),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - 16));
    }}
  }}

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}}
'''
    (project_dir / "lib/screens/detection_screen.dart").write_text(screen_code)

    # Generate main.dart
    main_code = f'''import 'package:flutter/material.dart';
import 'screens/detection_screen.dart';

void main() {{
  runApp(const MyApp());
}}

class MyApp extends StatelessWidget {{
  const MyApp({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return MaterialApp(
      title: '{project_name}',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const DetectionScreen(),
    );
  }}
}}
'''
    (project_dir / "lib/main.dart").write_text(main_code)

    # Generate README
    readme = f"""# {project_name}

Application mobile Flutter avec integration YOLO26.

## Tache: {task}
## Plateforme: {platform}

## Deploiement

### iOS
```bash
flutter build ios
```

### Android
```bash
flutter build apk
```

## Modele YOLO26

Le modele est place dans `assets/models/`. Pour changer le modele:

1. Exportez un modele YOLO26:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo26n.pt")
   model.export(format="coreml", quantize=8)  # iOS
   model.export(format="litert", quantize=8)  # Android
   ```

2. Copiez le fichier exporte dans `assets/models/`

3. Mettez a jour le chemin du modele dans `lib/services/yolo_service.dart`
"""
    (project_dir / "README.md").write_text(readme)

    print(f"\nProjet cree avec succes: {project_dir}")
    print(f"\nPour commencer:")
    print(f"  cd {project_dir}")
    print(f"  flutter pub get")
    print(f"  flutter run")


def create_react_native_project(
    project_name: str,
    task: str,
    platform: str,
    output_dir: str,
):
    """Scaffold a React Native project with YOLO integration."""
    print(f"Creation du projet React Native: {project_name}")
    print(f"Tache: {task} | Plateforme: {platform}")

    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    dirs = [
        "src/services",
        "src/screens",
        "src/components",
        "src/models",
        "android/app/src/main/java",
        "ios/YoloApp",
    ]
    for d in dirs:
        (project_dir / d).mkdir(parents=True, exist_ok=True)

    # Generate package.json
    package_json = f"""{{
  "name": "{project_name}",
  "version": "1.0.0",
  "description": "Application React Native avec YOLO26",
  "main": "index.js",
  "scripts": {{
    "android": "react-native run-android",
    "ios": "react-native run-ios",
    "start": "react-native start"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-native": "^0.73.0",
    "react-native-vision-camera": "^3.0.0"
  }},
  "devDependencies": {{
    "@types/react": "^18.2.0",
    "typescript": "^5.0.0"
  }}
}}
"""
    (project_dir / "package.json").write_text(package_json)

    # Generate TypeScript service
    service_code = f'''import {{ NativeModules, Platform }} from 'react-native';

const {{ YoloModule }} = NativeModules;

export interface DetectionResult {{
  box: {{ x1: number; y1: number; x2: number; y2: number }};
  confidence: number;
  class: number;
  name: string;
}}

export class YoloService {{
  private initialized = false;

  async initialize(modelPath?: string): Promise<void> {{
    if (this.initialized) return;

    const path = modelPath || (
      Platform.OS === 'ios'
        ? 'yolo26n.mlpackage'
        : 'yolo26n.tflite'
    );

    await YoloModule.initialize('{task}', path);
    this.initialized = true;
  }}

  async detect(imagePath: string): Promise<DetectionResult[]> {{
    if (!this.initialized) {{
      throw new Error('Service non initialisé. Appelez initialize() d\\'abord.');
    }}

    const results = await YoloModule.detect(imagePath);
    return results as DetectionResult[];
  }}

  async dispose(): Promise<void> {{
    if (this.initialized) {{
      await YoloModule.dispose();
      this.initialized = false;
    }}
  }}
}}
'''
    (project_dir / "src/services/yolo_service.ts").write_text(service_code)

    # Generate detection screen
    screen_code = f'''import React, {{ useEffect, useState, useRef }} from 'react';
import {{ View, StyleSheet, Text, Alert }} from 'react-native';
import {{ Camera, useCameraDevices, useFrameProcessor }} from 'react-native-vision-camera';
import {{ YoloService, DetectionResult }} from '../services/yolo_service';

const yoloService = new YoloService();

export function DetectionScreen() {{
  const devices = useCameraDevices();
  const device = devices.back;
  const [detections, setDetections] = useState<DetectionResult[]>([]);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {{
    (async () => {{
      try {{
        await yoloService.initialize();
        setIsReady(true);
      }} catch (error) {{
        Alert.alert('Erreur', 'Impossible d\\'initialiser YOLO');
      }}
    }})();

    return () => {{
      yoloService.dispose();
    }};
  }}, []);

  const frameProcessor = useFrameProcessor((frame) => {{
    'worklet';
    if (!isReady) return;

    // Le frame est traite nativement
    // Les resultats sont retournes via le module natif
  }}, [isReady]);

  if (!device || !isReady) {{
    return (
      <View style={{styles.container}}>
        <Text>Chargement...</Text>
      </View>
    );
  }}

  return (
    <View style={{styles.container}}>
      <Camera
        style={{StyleSheet.absoluteFill}}
        device={{device}}
        isActive={{true}}
        frameProcessor={{frameProcessor}}
      />
      <View style={{styles.overlay}}>
        {{detections.map((det, i) => (
          <View key={{i}} style={{styles.detectionBox}}>
            <Text style={{styles.detectionText}}>
              {{det.name}} {{(det.confidence * 100).toFixed(0)}}%
            </Text>
          </View>
        ))}}
      </View>
    </View>
  );
}}

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    backgroundColor: 'black',
  }},
  overlay: {{
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
  }},
  detectionBox: {{
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 8,
    borderRadius: 4,
    marginBottom: 4,
  }},
  detectionText: {{
    color: 'white',
    fontSize: 14,
  }},
}});
'''
    (project_dir / "src/screens/DetectionScreen.tsx").write_text(screen_code)

    print(f"\nProjet cree avec succes: {project_dir}")
    print(f"\nPour commencer:")
    print(f"  cd {project_dir}")
    print(f"  npm install")
    print(f"  npx react-native run-ios")


def main():
    parser = argparse.ArgumentParser(
        description="Creer un projet mobile avec integration YOLO26"
    )
    parser.add_argument(
        "--name", "-n",
        default="yolo-mobile-app",
        help="Nom du projet (defaut: yolo-mobile-app)",
    )
    parser.add_argument(
        "--framework", "-f",
        required=True,
        choices=["flutter", "react-native", "native-swift", "native-kotlin"],
        help="Framework mobile",
    )
    parser.add_argument(
        "--task", "-t",
        default="detect",
        choices=["detect", "segment", "classify", "pose", "obb"],
        help="Tache YOLO (defaut: detect)",
    )
    parser.add_argument(
        "--platform", "-p",
        default="both",
        choices=["ios", "android", "both"],
        help="Plateforme cible (defaut: both)",
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Repertoire de sortie (defaut: .)",
    )

    args = parser.parse_args()

    if args.framework == "flutter":
        create_flutter_project(args.name, args.task, args.platform, args.output)
    elif args.framework == "react-native":
        create_react_native_project(args.name, args.task, args.platform, args.output)
    else:
        print(f"Framework '{args.framework}' pas encore implemente.")
        print("Utilisez flutter ou react-native.")
        sys.exit(1)


if __name__ == "__main__":
    main()
