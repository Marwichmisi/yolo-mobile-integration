# Flutter + YOLO26 — Guide d'intégration complet

## PRÉREQUISITES

> **AVANT d'utiliser cette intégration, vous DEVEZ charger le skill `flutter-apply-architecture-best-practices` pour les optimisations d'architecture :**
>
> ```
> /skill flutter-apply-architecture-best-practices
> ```
>
> Ce skill contient les directives critiques pour l'architecture MVVM, la gestion de la mémoire, le throttling des frames, et les bonnes pratiques Flutter qui s'appliquent directement à cette intégration YOLO.

---

> **IMPORTANT — Pour les applications de production, utilisez les Platform Channels natifs (pas les packages tiers) pour l'inférence YOLO.**
>
> Les packages tiers (`tflite_flutter`, `flutter_vision`, etc.) ajoutent une couche d'abstraction supplémentaire, réduisent le contrôle sur la mémoire et le threading, et ne permettent pas d'optimiser finement l'inférence sur le GPU/Neural Engine. Pour une détection en temps réel à 10+ FPS, les **Platform Channels classiques** (MethodChannel) sont l'approche recommandée.

---

## Vue d'ensemble

Ce guide détaille l'intégration d'Ultralytics YOLO26 dans une application Flutter pour la détection d'objets en temps réel. L'architecture repose sur les **Platform Channels** de Flutter pour communiquer avec le code natif (iOS/Android) qui exécute l'inférence YOLO.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Flutter (Dart)                 │
│  ┌─────────────┐    ┌──────────────────┐    │
│  │ CameraPage  │───▶│ YoloService      │    │
│  │  (camera)   │    │ (MethodChannel)  │    │
│  └─────────────┘    └────────┬─────────┘    │
│                              │               │
├──────────────────────────────┼───────────────┤
│         Platform Channel                   │
├──────────────────────────────┼───────────────┤
│  iOS (MethodChannel)        │ Android (MethodChannel)│
│  ┌──────────────────┐       │ ┌──────────────────┐
│  │ CoreML Inference │◀──────┘ │ LiteRT Inference │
│  │ (YOLO26.mlmodel)│         │ (YOLO26.tflite)  │
│  └──────────────────┘         └──────────────────┘
└─────────────────────────────────────────────┘
```

**Flux de données :**
1. Flutter capture une image via la caméra
2. L'image est envoyée au natif via MethodChannel
3. Le natif charge le modèle (CoreML sur iOS, LiteRT sur Android)
4. L'inférence est exécutée sur l'image
5. Les résultats sont renvoyés à Dart
6. Flutter affiche les annotations (bounding boxes)

---

## 1. Configuration du projet

### `pubspec.yaml`

```yaml
name: yolo_app
description: YOLO26 Real-time Object Detection

publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter

  # Caméra
  camera: ^0.11.0

  # Gestion d'état
  provider: ^6.1.2

  # Utilitaires
  path_provider: ^2.1.4
  permission_handler: ^11.3.1

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^5.0.0

flutter:
  uses-material-design: true

  assets:
    - assets/models/
```

### iOS — `ios/Runner/Info.plist`

Ajouter les permissions caméra et photo :

```xml
<key>NSCameraUsageDescription</key>
<string>Cette application a besoin d'accéder à la caméra pour la détection d'objets en temps réel.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Cette application a besoin d'accéder à la bibliothèque photo pour sauvegarder les détections.</string>
```

### Android — `android/app/src/main/AndroidManifest.xml`

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Permissions caméra -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <!-- Pour Android 13+ -->
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />

    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />

    <application
        android:label="YOLO26 Detection"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:taskAffinity=""
            android:theme="@style/LaunchTheme"
            android:configChanges="orientation|keyboardHidden|keyboard|screenSize|smallestScreenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
            android:hardwareAccelerated="true"
            android:windowSoftInputMode="adjustResize">
            <meta-data
              android:name="io.flutter.embedding.android.NormalTheme"
              android:resource="@style/NormalTheme"
              />
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        <meta-data
            android:name="flutterEmbedding"
            android:value="2" />
    </application>
</manifestinder>
```

### Android — `android/app/build.gradle`

```gradle
android {
    compileSdk 34

    defaultConfig {
        minSdk 21
        targetSdk 34

        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }

    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### Android — `android/app/proguard-rules.pro`

```proguard
-keep class com.yoloapp.** { *; }
-keep class org.tensorflow.lite.** { *; }
```

---

## 2. Architecture des Platform Channels

### Côté Dart — `lib/services/yolo_service.dart`

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Représente une détection individuelle
class YoloDetection {
  final String label;
  final double confidence;
  final double x;
  final double y;
  final double width;
  final double height;

  const YoloDetection({
    required this.label,
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory YoloDetection.fromMap(Map<dynamic, dynamic> map) {
    return YoloDetection(
      label: map['label'] as String,
      confidence: (map['confidence'] as num).toDouble(),
      x: (map['x'] as num).toDouble(),
      y: (map['y'] as num).toDouble(),
      width: (map['width'] as num).toDouble(),
      height: (map['height'] as num).toDouble(),
    );
  }

  @override
  String toString() =>
      'YoloDetection(label: $label, confidence: ${(confidence * 100).toStringAsFixed(1)}%, '
      'bbox: [$x, $y, $width, $height])';
}

/// Service principal pour communiquer avec le natif via MethodChannel
class YoloService {
  static const MethodChannel _channel = MethodChannel('com.yoloapp/yolo');

  static bool _isModelLoaded = false;
  static bool get isModelLoaded => _isModelLoaded;

  /// Charger le modèle YOLO depuis les assets
  static Future<bool> loadModel(String modelName) async {
    try {
      final result = await _channel.invokeMethod<bool>('loadModel', {
        'modelName': modelName,
      });
      _isModelLoaded = result ?? false;
      return _isModelLoaded;
    } on PlatformException catch (e) {
      _isModelLoaded = false;
      throw YoloException('Erreur chargement modèle: ${e.message}', e.code);
    }
  }

  /// Exécuter la détection sur une image
  static Future<List<YoloDetection>> detect(
    Uint8List imageBytes, {
    required int width,
    required int height,
    double confidenceThreshold = 0.25,
  }) async {
    if (!_isModelLoaded) {
      throw YoloException('Modèle non chargé. Appelez loadModel() d\'abord.', 'MODEL_NOT_LOADED');
    }

    try {
      final result = await _channel.invokeMethod<List<dynamic>>('detect', {
        'imageBytes': imageBytes,
        'width': width,
        'height': height,
        'confidenceThreshold': confidenceThreshold,
      });

      if (result == null) return [];

      return result
          .map((e) => YoloDetection.fromMap(e as Map<dynamic, dynamic>))
          .toList();
    } on PlatformException catch (e) {
      throw YoloException('Erreur détection: ${e.message}', e.code);
    }
  }

  /// Libérer le modèle de la mémoire
  static Future<void> unloadModel() async {
    try {
      await _channel.invokeMethod('unloadModel');
      _isModelLoaded = false;
    } on PlatformException catch (e) {
      throw YoloException('Erreur déchargement: ${e.message}', e.code);
    }
  }

  /// Vérifier si le modèle est chargé
  static Future<bool> isLoaded() async {
    try {
      final result = await _channel.invokeMethod<bool>('isLoaded');
      _isModelLoaded = result ?? false;
      return _isModelLoaded;
    } on PlatformException {
      return false;
    }
  }
}

/// Exception personnalisée pour les erreurs YOLO
class YoloException implements Exception {
  final String message;
  final String code;

  const YoloException(this.message, this.code);

  @override
  String toString() => 'YoloException($code): $message';
}
```

---

## 3. Côté iOS — Swift (MethodChannel + CoreML)

### `ios/Runner/YoloHandler.swift`

```swift
import Foundation
import Flutter
import UIKit
import CoreML
import Vision

/// Gestionnaire principal YOLO pour iOS
public class YoloHandler: NSObject {
    private var model: VNCoreMLModel?
    private var isModelLoaded = false
    private let processingQueue = DispatchQueue(
        label: "com.yoloapp.inference",
        qos: .userInitiated,
        attributes: .concurrent
    )

    /// Charger le modèle YOLO depuis le bundle
    func loadModel(modelName: String, result: @escaping FlutterResult) {
        processingQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                let modelConfig = MLModelConfiguration()
                modelConfig.computeUnits = .all

                // Chercher le modèle dans le bundle
                guard let modelURL = Bundle.main.url(
                    forResource: modelName,
                    withExtension: "mlmodelc"
                ) else {
                    // Essayer sans extension (déjà compilé)
                    if let compiledURL = Bundle.main.url(
                        forResource: modelName,
                        withExtension: nil
                    ) {
                        let mlModel = try MLModel(contentsOf: compiledURL, configuration: modelConfig)
                        self.model = try VNCoreMLModel(for: mlModel)
                        self.isModelLoaded = true
                        DispatchQueue.main.async { result(true) }
                        return
                    }

                    DispatchQueue.main.async {
                        result(FlutterError(
                            code: "MODEL_ERROR",
                            message: "Modèle '\(modelName)' introuvable dans le bundle",
                            details: nil
                        ))
                    }
                    return
                }

                let mlModel = try MLModel(contentsOf: modelURL, configuration: modelConfig)
                self.model = try VNCoreMLModel(for: mlModel)
                self.isModelLoaded = true

                DispatchQueue.main.async { result(true) }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "MODEL_ERROR",
                        message: "Impossible de charger le modèle: \(error.localizedDescription)",
                        details: nil
                    ))
                }
            }
        }
    }

    /// Exécuter la détection sur une image
    func detect(
        imageBytes: FlutterStandardTypedData,
        width: Int,
        height: Int,
        confidenceThreshold: Double,
        result: @escaping FlutterResult
    ) {
        guard let model = model, isModelLoaded else {
            result(FlutterError(
                code: "MODEL_NOT_LOADED",
                message: "Modèle non chargé. Appelez loadModel() d'abord.",
                details: nil
            ))
            return
        }

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            let data = imageBytes.data

            guard let image = UIImage(data: data),
                  let cgImage = image.cgImage else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "IMAGE_ERROR",
                        message: "Impossible de décoder l'image. Vérifiez le format.",
                        details: nil
                    ))
                }
                return
            }

            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    DispatchQueue.main.async {
                        result(FlutterError(
                            code: "INFERENCE_ERROR",
                            message: error.localizedDescription,
                            details: nil
                        ))
                    }
                    return
                }

                guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                    DispatchQueue.main.async { result([]) }
                    return
                }

                var detections: [[String: Any]] = []

                for observation in observations {
                    guard let topLabel = observation.labels.first else { continue }

                    // Filtrer par seuil de confiance
                    if Double(topLabel.confidence) < confidenceThreshold { continue }

                    let bbox = observation.boundingBox

                    // Convertir les coordonnées Vision (origine bas-gauche) vers Flutter (origine haut-gauche)
                    let x = Double(bbox.origin.x)
                    let y = Double(1.0 - bbox.origin.y - bbox.height)
                    let w = Double(bbox.width)
                    let h = Double(bbox.height)

                    detections.append([
                        "label": topLabel.identifier,
                        "confidence": Double(topLabel.confidence),
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    ])
                }

                DispatchQueue.main.async { result(detections) }
            }

            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "INFERENCE_ERROR",
                        message: error.localizedDescription,
                        details: nil
                    ))
                }
            }
        }
    }

    /// Libérer le modèle de la mémoire
    func unloadModel(result: @escaping FlutterResult) {
        model = nil
        isModelLoaded = false
        result(true)
    }

    /// Retourner l'état du modèle
    func isLoaded(result: @escaping FlutterResult) {
        result(isModelLoaded)
    }
}
```

### `ios/Runner/AppDelegate.swift`

```swift
import UIKit
import Flutter

@main
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let controller = window?.rootViewController as! FlutterViewController

        // Enregistrer le MethodChannel YOLO
        let yoloChannel = FlutterMethodChannel(
            name: "com.yoloapp/yolo",
            binaryMessenger: controller.binaryMessenger
        )

        let yoloHandler = YoloHandler()

        yoloChannel.setMethodCallHandler { [weak yoloHandler] call, result in
            switch call.method {
            case "loadModel":
                guard let args = call.arguments as? [String: Any],
                      let modelName = args["modelName"] as? String else {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Arguments invalides pour loadModel",
                        details: nil
                    ))
                    return
                }
                yoloHandler?.loadModel(modelName: modelName, result: result)

            case "detect":
                guard let args = call.arguments as? [String: Any],
                      let imageBytes = args["imageBytes"] as? FlutterStandardTypedData,
                      let width = args["width"] as? Int,
                      let height = args["height"] as? Int else {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Arguments invalides pour detect",
                        details: nil
                    ))
                    return
                }
                let confidenceThreshold = (args["confidenceThreshold"] as? Double) ?? 0.25
                yoloHandler?.detect(
                    imageBytes: imageBytes,
                    width: width,
                    height: height,
                    confidenceThreshold: confidenceThreshold,
                    result: result
                )

            case "unloadModel":
                yoloHandler?.unloadModel(result: result)

            case "isLoaded":
                yoloHandler?.isLoaded(result: result)

            default:
                result(FlutterMethodNotImplemented)
            }
        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
}
```

### Ajouter le modèle CoreML au projet Xcode

1. Exporter le modèle depuis Python :
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="coreml", quantize=8)
```

2. Glisser-déposer le fichier `.mlmodel` dans le projet Xcode
3. Cocher "Copy items if needed" et la cible "Runner"
4. Xcode compile automatiquement le modèle en `.mlmodelc`

---

## 4. Côté Android — Kotlin (MethodChannel + LiteRT)

### `android/app/src/main/kotlin/com/yoloapp/YoloHandler.kt`

```kotlin
package com.yoloapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

class YoloHandler(private val context: Context) : MethodChannel.MethodCallHandler {

    companion object {
        private const val CHANNEL_NAME = "com.yoloapp/yolo"
        private const val INPUT_SIZE = 640
    }

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private val executor = Executors.newSingleThreadExecutor()

    fun registerChannel(flutterEngine: FlutterEngine) {
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL_NAME)
            .setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "loadModel" -> {
                val modelName = call.argument<String>("modelName")
                if (modelName == null) {
                    result.error("INVALID_ARGS", "modelName manquant", null)
                    return
                }
                loadModel(modelName, result)
            }
            "detect" -> {
                val imageBytes = call.argument<ByteArray>("imageBytes")
                val width = call.argument<Int>("width")
                val height = call.argument<Int>("height")
                val confidenceThreshold = call.argument<Double>("confidenceThreshold") ?: 0.25

                if (imageBytes == null || width == null || height == null) {
                    result.error("INVALID_ARGS", "Arguments invalides pour detect", null)
                    return
                }
                detect(imageBytes, width, height, confidenceThreshold, result)
            }
            "unloadModel" -> {
                unloadModel(result)
            }
            "isLoaded" -> {
                result.success(isModelLoaded)
            }
            else -> result.notImplemented()
        }
    }

    private fun loadModel(modelName: String, result: MethodChannel.Result) {
        executor.execute {
            try {
                if (isModelLoaded) {
                    interpreter?.close()
                }

                val options = Interpreter.Options().apply {
                    setNumThreads(Runtime.getRuntime().availableProcessors())
                }

                val modelBuffer = loadModelFile(modelName)
                interpreter = Interpreter(modelBuffer, options)
                isModelLoaded = true
                result.success(true)
            } catch (e: Exception) {
                result.error("MODEL_ERROR", "Impossible de charger le modèle: ${e.message}", null)
            }
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        return try {
            // Essayer d'abord les assets
            val fileDescriptor = context.assets.openFd(modelName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            ).also {
                fileDescriptor.close()
            }
        } catch (e: Exception) {
            // Sinon, essayer le chemin direct
            val file = java.io.File(modelName)
            if (!file.exists()) {
                throw IllegalArgumentException("Fichier modèle introuvable: $modelName")
            }
            val inputStream = FileInputStream(file)
            val fileChannel = inputStream.channel
            fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).also {
                fileChannel.close()
            }
        }
    }

    private fun detect(
        imageBytes: ByteArray,
        width: Int,
        height: Int,
        confidenceThreshold: Double,
        result: MethodChannel.Result
    ) {
        val interp = interpreter
        if (interp == null || !isModelLoaded) {
            result.error("MODEL_NOT_LOADED", "Modèle non chargé. Appelez loadModel() d'abord.", null)
            return
        }

        executor.execute {
            try {
                // 1. Décoder le JPEG → Bitmap
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: throw IllegalArgumentException("Impossible de décoder l'image JPEG")

                // 2. Redimensionner en 640x640
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

                // 3. Prétraiter : RGB float [0.0, 1.0]
                val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
                inputBuffer.order(ByteOrder.nativeOrder())

                val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
                resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

                for (pixel in pixels) {
                    inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
                    inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
                    inputBuffer.putFloat((pixel and 0xFF) / 255.0f)           // B
                }

                // 4. Buffer de sortie : format YOLO26 end-to-end [1, 300, 6]
                val outputBuffer = Array(1) { Array(300) { FloatArray(6) } }

                // 5. Inférence
                interp.run(inputBuffer, outputBuffer)

                // 6. Post-traitement : extraire les détections (déjà NMS-free)
                val detections = postProcess(outputBuffer[0], confidenceThreshold.toFloat())

                // 7. Convertir en List<Map> pour Flutter
                val resultList = ArrayList<Map<String, Any>>()
                for (det in detections) {
                    val map = HashMap<String, Any>()
                    map["label"] = det["label"] as String
                    map["confidence"] = det["confidence"] as Double
                    map["x"] = det["x"] as Double
                    map["y"] = det["y"] as Double
                    map["width"] = det["width"] as Double
                    map["height"] = det["height"] as Double
                    resultList.add(map)
                }

                result.success(resultList)

                // Nettoyage
                resizedBitmap.recycle()
                bitmap.recycle()
            } catch (e: Exception) {
                result.error("INFERENCE_ERROR", e.message, null)
            }
        }
    }

    private fun postProcess(
        output: Array<FloatArray>,
        confidenceThreshold: Float
    ): List<Map<String, Any>> {
        val detections = mutableListOf<Map<String, Any>>()

        // Format YOLO26 end-to-end: [300, 6]
        // Chaque ligne: [cx, cy, w, h, class_id, confidence]
        for (i in 0 until 300) {
            val confidence = output[i][5]
            if (confidence < confidenceThreshold) continue

            val cx = output[i][0]
            val cy = output[i][1]
            val w = output[i][2]
            val h = output[i][3]
            val classId = output[i][4].toInt()

            // Convertir en coordonnées normalisées [0..1]
            val x = (cx - w / 2) / INPUT_SIZE
            val y = (cy - h / 2) / INPUT_SIZE
            val width = w / INPUT_SIZE
            val height = h / INPUT_SIZE

            val className = getClassName(classId)

            detections.add(mapOf(
                "label" to className,
                "confidence" to confidence.toDouble(),
                "x" to x.toDouble(),
                "y" to y.toDouble(),
                "width" to width.toDouble(),
                "height" to height.toDouble()
            ))
        }

        return detections
    }

    private fun getClassName(classIdx: Int): String {
        val classes = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        )
        return if (classIdx in classes.indices) classes[classIdx] else "class_$classIdx"
    }

    private fun unloadModel(result: MethodChannel.Result) {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        result.success(true)
    }
}
```

### `android/app/src/main/kotlin/com/yoloapp/MainActivity.kt`

```kotlin
package com.yoloapp

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Enregistrer le MethodChannel YOLO
        YoloHandler(applicationContext).registerChannel(flutterEngine)
    }
}
```

### Ajouter le modèle LiteRT au projet Android

1. Exporter le modèle depuis Python :
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="litert", quantize="w8a32")
```

2. Copier le fichier `.tflite` dans `android/app/src/main/assets/`

### Ajouter la dépendance LiteRT — `android/app/build.gradle`

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

---

## 5. Intégration caméra temps réel

### `lib/services/camera_service.dart`

```dart
import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

/// Service de gestion de la caméra avec inférence YOLO
class CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;

  CameraController? get controller => _controller;
  bool get isInitialized => _isInitialized;

  /// Initialiser la caméra
  Future<void> initialize() async {
    // Demander la permission caméra
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      throw CameraException('Permission caméra refusée');
    }

    // Lister les caméras disponibles
    _cameras = await availableCameras();
    if (_cameras == null || _cameras!.isEmpty) {
      throw CameraException('Aucune caméra disponible');
    }

    // Utiliser la caméra arrière
    final backCamera = _cameras!.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras!.first,
    );

    // Créer le contrôleur
    _controller = CameraController(
      backCamera,
      ResolutionPreset.medium, // 480p suffit pour YOLO
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await _controller!.initialize();
    _isInitialized = true;
  }

  /// Capturer une image et la convertir en Uint8List
  Future<Uint8List> captureImage() async {
    if (!_isInitialized || _controller == null) {
      throw CameraException('Caméra non initialisée');
    }

    final XFile photo = await _controller!.takePicture();
    return await photo.readAsBytes();
  }

  /// Obtenir les dimensions de la caméra
  Size getPreviewSize() {
    if (_controller == null || !_isInitialized) {
      return const Size(640, 480);
    }
    final size = _controller!.value.previewSize;
    return size ?? const Size(640, 480);
  }

  /// Libérer les ressources
  Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
  }
}

/// Exception caméra
class CameraException implements Exception {
  final String message;
  const CameraException(this.message);

  @override
  String toString() => 'CameraException: $message';
}

/// Classe utilitaire pour les dimensions
class Size {
  final double width;
  final double height;
  const Size(this.width, this.height);
}
```

---

## 6. Overlay des bounding boxes (CustomPainter)

### `lib/widgets/detection_overlay.dart`

```dart
import 'package:flutter/material.dart';
import '../services/yolo_service.dart';

/// CustomPainter pour dessiner les bounding boxes
class DetectionOverlay extends CustomPainter {
  final List<YoloDetection> detections;
  final Size imageSize;
  final Size widgetSize;

  DetectionOverlay({
    required this.detections,
    required this.imageSize,
    required this.widgetSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculer le ratio de redimensionnement
    final scaleX = widgetSize.width / imageSize.width;
    final scaleY = widgetSize.height / imageSize.height;

    // Utiliser le ratio le plus petit pour préserver l'aspect
    final scale = scaleX < scaleY ? scaleX : scaleY;

    // Calculer les offsets pour centrer l'image
    final offsetX = (widgetSize.width - imageSize.width * scale) / 2;
    final offsetY = (widgetSize.height - imageSize.height * scale) / 2;

    for (final detection in detections) {
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0
        ..color = _getColorForClass(detection.label);

      // Convertir les coordonnées normalisées en coordonnées écran
      final rect = Rect.fromLTWH(
        offsetX + detection.x * imageSize.width * scale,
        offsetY + detection.y * imageSize.height * scale,
        detection.width * imageSize.width * scale,
        detection.height * imageSize.height * scale,
      );

      // Dessiner le rectangle
      canvas.drawRect(rect, paint);

      // Dessiner le fond du label
      final labelText = '${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final labelRect = Rect.fromLTWH(
        rect.left,
        rect.top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final bgPaint = Paint()..color = _getColorForClass(detection.label);
      canvas.drawRect(labelRect, bgPaint);

      // Dessiner le texte
      textPainter.paint(
        canvas,
        Offset(labelRect.left + 4, labelRect.top + 2),
      );
    }
  }

  Color _getColorForClass(String className) {
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.amber,
    ];

    int hash = 0;
    for (int i = 0; i < className.length; i++) {
      hash = className.codeUnitAt(i) + ((hash << 5) - hash);
    }
    return colors[hash.abs() % colors.length];
  }

  @override
  bool shouldRepaint(covariant DetectionOverlay oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.widgetSize != widgetSize;
  }
}
```

---

## 7. Gestion d'état avec Provider

### `lib/providers/detection_provider.dart`

```dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../services/yolo_service.dart';
import '../services/camera_service.dart';

/// État de la détection
enum DetectionState {
  idle,
  loadingModel,
  ready,
  detecting,
  error,
}

/// Provider pour gérer l'état des détections
class DetectionProvider extends ChangeNotifier {
  DetectionState _state = DetectionState.idle;
  List<YoloDetection> _detections = [];
  String? _errorMessage;
  int _fps = 0;
  DateTime? _lastFrameTime;
  int _frameCount = 0;

  // Getters
  DetectionState get state => _state;
  List<YoloDetection> get detections => _detections;
  String? get errorMessage => _errorMessage;
  int get fps => _fps;
  bool get isReady => _state == DetectionState.ready || _state == DetectionState.detecting;

  final CameraService _cameraService = CameraService();

  CameraService get cameraService => _cameraService;

  /// Initialiser le service (caméra + modèle)
  Future<void> initialize() async {
    try {
      _setState(DetectionState.loadingModel);

      // Initialiser la caméra
      await _cameraService.initialize();

      // Charger le modèle
      await YoloService.loadModel('yolo26n');

      _setState(DetectionState.ready);
    } catch (e) {
      _errorMessage = e.toString();
      _setState(DetectionState.error);
    }
  }

  /// Exécuter une détection sur une frame
  Future<void> detectFrame() async {
    if (!isReady) return;

    try {
      _setState(DetectionState.detecting);

      // Capturer l'image
      final imageBytes = await _cameraService.captureImage();
      final previewSize = _cameraService.getPreviewSize();

      // Exécuter l'inférence
      final results = await YoloService.detect(
        imageBytes,
        width: previewSize.width.toInt(),
        height: previewSize.height.toInt(),
      );

      _detections = results;
      _updateFps();
      _setState(DetectionState.ready);
    } catch (e) {
      _errorMessage = e.toString();
      _setState(DetectionState.error);
    }
  }

  /// Détection en continu avec throttling
  Stream<List<YoloDetection>> detectContinuously({int maxFps = 10}) async* {
    final interval = Duration(milliseconds: (1000 / maxFps).round());

    while (isReady) {
      final stopwatch = Stopwatch()..start();

      try {
        final imageBytes = await _cameraService.captureImage();
        final previewSize = _cameraService.getPreviewSize();

        final results = await YoloService.detect(
          imageBytes,
          width: previewSize.width.toInt(),
          height: previewSize.height.toInt(),
        );

        _detections = results;
        _updateFps();
        notifyListeners();

        yield results;
      } catch (e) {
        // Ignorer les erreurs en mode continu
      }

      // Throttling
      final elapsed = stopwatch.elapsed;
      if (elapsed < interval) {
        await Future.delayed(interval - elapsed);
      }
    }
  }

  void _updateFps() {
    _frameCount++;
    final now = DateTime.now();

    if (_lastFrameTime != null) {
      final diff = now.difference(_lastFrameTime!).inMilliseconds;
      if (diff >= 1000) {
        _fps = _frameCount;
        _frameCount = 0;
        _lastFrameTime = now;
      }
    } else {
      _lastFrameTime = now;
    }
  }

  void _setState(DetectionState newState) {
    _state = newState;
    notifyListeners();
  }

  @override
  void dispose() {
    _cameraService.dispose();
    YoloService.unloadModel();
    super.dispose();
  }
}
```

---

## 8. Exemple complet — Écran de détection

### `lib/screens/detection_screen.dart`

```dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/detection_provider.dart';
import '../widgets/detection_overlay.dart';

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  StreamSubscription? _detectionSubscription;

  @override
  void initState() {
    super.initState();
    // Initialiser le provider après le premier build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<DetectionProvider>().initialize();
    });
  }

  @override
  void dispose() {
    _detectionSubscription?.cancel();
    super.dispose();
  }

  void _startDetection() {
    final provider = context.read<DetectionProvider>();
    _detectionSubscription?.cancel();
    _detectionSubscription = provider
        .detectContinuously(maxFps: 10)
        .listen((_) {
      // Les mises à jour sont gérées par le Provider
    });
  }

  void _stopDetection() {
    _detectionSubscription?.cancel();
    _detectionSubscription = null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Consumer<DetectionProvider>(
        builder: (context, provider, child) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // Caméra ou écran de chargement
              _buildCameraPreview(provider),

              // Overlay des détections
              if (provider.isReady && provider.detections.isNotEmpty)
                _buildDetectionOverlay(provider),

              // Contrôles
              _buildControls(provider),

              // Indicateur FPS
              if (provider.isReady) _buildFpsIndicator(provider),

              // Erreur
              if (provider.state == DetectionState.error)
                _buildErrorOverlay(provider),
            ],
          );
        },
      ),
    );
  }

  Widget _buildCameraPreview(DetectionProvider provider) {
    if (provider.state == DetectionState.idle ||
        provider.state == DetectionState.loadingModel) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 16),
            Text(
              'Chargement du modèle...',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      );
    }

    if (provider.state == DetectionState.error) {
      return const Center(
        child: Text(
          'Erreur d\'initialisation',
          style: TextStyle(color: Colors.red, fontSize: 16),
        ),
      );
    }

    final controller = provider.cameraService.controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    return CameraPreview(controller);
  }

  Widget _buildDetectionOverlay(DetectionProvider provider) {
    return Positioned.fill(
      child: CustomPaint(
        painter: DetectionOverlay(
          detections: provider.detections,
          imageSize: provider.cameraService.getPreviewSize(),
          widgetSize: MediaQuery.of(context).size,
        ),
      ),
    );
  }

  Widget _buildControls(DetectionProvider provider) {
    if (!provider.isReady) return const SizedBox.shrink();

    return Positioned(
      bottom: 40,
      left: 0,
      right: 0,
      child: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FloatingActionButton(
              heroTag: "start",
              backgroundColor: Colors.green,
              onPressed: _startDetection,
              child: const Icon(Icons.play_arrow),
            ),
            const SizedBox(width: 16),
            FloatingActionButton(
              heroTag: "stop",
              backgroundColor: Colors.red,
              onPressed: _stopDetection,
              child: const Icon(Icons.stop),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFpsIndicator(DetectionProvider provider) {
    return Positioned(
      top: 40,
      right: 16,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          '${provider.fps} FPS',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }

  Widget _buildErrorOverlay(DetectionProvider provider) {
    return Positioned.fill(
      child: Container(
        color: Colors.black87,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(
                provider.errorMessage ?? 'Erreur inconnue',
                style: const TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => provider.initialize(),
                child: const Text('Réessayer'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
```

### `lib/main.dart`

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/detection_provider.dart';
import 'screens/detection_screen.dart';

void main() {
  runApp(const YoloApp());
}

class YoloApp extends StatelessWidget {
  const YoloApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => DetectionProvider(),
      child: MaterialApp(
        title: 'YOLO26 Detection',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
          useMaterial3: true,
        ),
        home: const DetectionScreen(),
      ),
    );
  }
}
```

---

## 9. Optimisations de performance

### Throttling des frames

```dart
/// Utilitaire pour limiter le FPS
class FrameThrottler {
  final int maxFps;
  DateTime? _lastFrameTime;

  FrameThrottler({this.maxFps = 10});

  bool shouldProcess() {
    final now = DateTime.now();
    if (_lastFrameTime == null) {
      _lastFrameTime = now;
      return true;
    }

    final elapsed = now.difference(_lastFrameTime!).inMilliseconds;
    final minInterval = (1000 / maxFps).round();

    if (elapsed >= minInterval) {
      _lastFrameTime = now;
      return true;
    }
    return false;
  }
}
```

### Gestion de la mémoire

```dart
/// Gestionnaire de mémoire pour l'inférence
class MemoryManager {
  static final MemoryManager _instance = MemoryManager._internal();
  factory MemoryManager() => _instance;
  MemoryManager._internal();

  bool _isCleaning = false;

  /// Nettoyer la mémoire périodiquement
  void startPeriodicCleanup() {
    Timer.periodic(const Duration(seconds: 30), (_) {
      _performCleanup();
    });
  }

  void _performCleanup() {
    if (_isCleaning) return;
    _isCleaning = true;

    // Forcer le garbage collector
    // Note: En Dart, on ne peut pas forcer le GC directement,
    // mais on peut aider en libérant les références

    _isCleaning = false;
  }

  /// Nettoyer avant une détection
  void prepareForInference() {
    // Réinitialiser les buffers temporaires
  }
}
```

### GPU Delegation (Android)

```kotlin
// Dans YoloHandler.kt — ajouter le GPU delegate
private fun createInterpreterOptions(): Interpreter.Options {
    val options = Interpreter.Options().apply {
        setNumThreads(Runtime.getRuntime().availableProcessors())
    }

    // Essayer d'utiliser le GPU delegate
    try {
        val gpuDelegate = org.tensorflow.lite.gpu.GpuDelegate()
        options.addDelegate(gpuDelegate)
    } catch (e: Exception) {
        // GPU delegate non disponible, utiliser le CPU
    }

    return options
}
```

### Optimisations iOS

```swift
// Dans YoloHandler.swift — optimiser les performances
private func optimizeModelConfig() -> MLModelConfiguration {
    let config = MLModelConfiguration()
    config.computeUnits = .all  // Utilise CPU + GPU + Neural Engine

    // Pour les appareils récents, utiliser le Neural Engine
    if #available(iOS 15.0, *) {
        config.allowLowPrecisionAccumulationOnGPU = true
    }

    return config
}
```

---

## 10. Problèmes courants (Troubleshooting)

### `MissingPluginException` pour `com.yoloapp/yolo`

**Cause :** Le MethodChannel n'est pas enregistré côté natif.

**Solutions :**
- **iOS** : Vérifiez que `AppDelegate.swift` enregistre bien le `FlutterMethodChannel` dans `application(_:didFinishLaunchingWithOptions:)`
- **Android** : Vérifiez que `MainActivity.kt` appelle `YoloHandler.registerChannel(flutterEngine)` dans `configureFlutterEngine`
- Nettoyez et reconstruisez : `flutter clean && flutter run`

### Erreur de chargement du modèle (`MODEL_ERROR`)

**Cause :** Le fichier modèle n'est pas accessible ou est corrompu.

**Solutions :**
- **iOS** : Le `.mlmodel` doit être dans le bundle Xcode. Vérifiez dans Xcode que le fichier est bien inclus dans la cible "Runner".
- **Android** : Le `.tflite` doit être dans `android/app/src/main/assets/`. Ne pas le mettre dans `res/`.
- Vérifiez la taille du fichier (un modèle YOLO26n fait ~6 Mo en FP32, ~2 Mo en INT8).

### Erreur `INFERENCE_ERROR` ou crash

**Cause :** Format d'image incorrect ou dimensions incompatibles.

**Solutions :**
- L'image doit être en **JPEG** ou **PNG** en format `Uint8List`.
- Le buffer natif attend des images en RGB normalisé [0, 1].
- Vérifiez les dimensions du modèle (640x640 pour YOLO26n).

### Performance faible (< 5 FPS)

**Causes et solutions :**
- Réduisez la résolution de la caméra (480p suffit pour YOLO)
- Limitez le FPS à 10 dans le throttling
- Sur Android, activez le GPU delegate
- Sur iOS, utilisez `computeUnits = .all` pour utiliser le Neural Engine
- Vérifiez que le modèle est bien quantifié (INT8 ou W8A32)

### L'application crash au démarrage après ajout du module

**Solutions :**
- Nettoyez complètement : `flutter clean && cd android && ./gradlew clean && cd ../ios && pod install`
- Vérifiez les versions de Flutter (`flutter --version`)
- Assurez-vous que la version de LiteRT/TFLite est compatible avec votre SDK

### Le module fonctionne en développement mais pas en release

**Cause :** ProGuard/R8 supprime le module en release.

**Solution** — Ajouter dans `android/app/proguard-rules.pro` :

```proguard
-keep class com.yoloapp.** { *; }
-keep class org.tensorflow.lite.** { *; }
```

### Caméra noire ou ne s'affiche pas

**Solutions :**
- Vérifiez les permissions dans `Info.plist` (iOS) et `AndroidManifest.xml` (Android)
- Sur iOS, vérifiez que `NSCameraUsageDescription` est bien défini
- Sur Android, demandez la permission à l'exécution avec `permission_handler`
- Vérifiez que `CameraController.initialize()` est bien appelé avant d'afficher la preview

---

## 11. Structure du projet

```
lib/
├── main.dart
├── providers/
│   └── detection_provider.dart   # Gestion d'état Provider
├── screens/
│   └── detection_screen.dart     # Écran principal
├── services/
│   ├── yolo_service.dart         # MethodChannel Dart
│   └── camera_service.dart       # Gestion caméra
└── widgets/
    └── detection_overlay.dart    # CustomPainter bounding boxes

ios/Runner/
├── AppDelegate.swift             # Enregistrement MethodChannel
├── YoloHandler.swift             # Inférence CoreML
└── Info.plist                    # Permissions caméra

android/app/src/main/
├── AndroidManifest.xml           # Permissions
├── assets/
│   └── yolo26n.tflite            # Modèle LiteRT
└── kotlin/com/yoloapp/
    ├── MainActivity.kt           # Enregistrement MethodChannel
    └── YoloHandler.kt            # Inférence LiteRT

assets/
└── models/
    └── yolo26n.mlmodel           # Modèle CoreML (iOS)
```

---

## 12. Références

- `flutter-apply-architecture-best-practices` — Architecture MVVM, Provider, Repository
- [Documentation Ultralytics](https://docs.ultralytics.com/fr)
- [Guide d'exportation](https://docs.ultralytics.com/fr/modes/export)
- [Format LiteRT](https://docs.ultralytics.com/fr/integrations/litert)
- [Format CoreML](https://docs.ultralytics.com/fr/integrations/coreml)
- [Package camera Flutter](https://pub.dev/packages/camera)
- [Package provider Flutter](https://pub.dev/packages/provider)
