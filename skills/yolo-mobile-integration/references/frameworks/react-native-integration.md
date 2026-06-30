# React Native + YOLO26 — Guide d'intégration complet

## PRÉREQUISITES

> **AVANT d'utiliser cette intégration, vous DEVEZ charger le skill `react-native-best-practices` pour les optimisations de performance :**
>
> ```
> /skill react-native-best-practices
> ```
>
> Ce skill contient les directives critiques pour l'optimisation Hermes, la gestion de la mémoire, le throttling des frames, et les bonnes pratiques React Native qui s'appliquent directement à cette intégration YOLO.

---

> **IMPORTANT — Pour les applications de production, utilisez les Native Modules (pas les modules Expo) pour l'inférence YOLO.**
>
> Les modules Expo (`expo-modules-core`) ajoutent une couche d'abstraction supplémentaire, réduisent le contrôle sur la mémoire et le threading, et ne permettent pas d'optimiser finement l'inférence sur le GPU/Neural Engine. Pour une détection en temps réel à 10+ FPS, les **Native Modules classiques** (RCTBridgeModule / ReactContextBaseJavaModule) sont l'approche recommandée.

---

## Vue d'ensemble

Ce guide détaille l'intégration d'Ultralytics YOLO26 dans une application React Native pour la détection d'objets en temps réel. L'architecture repose sur les **Native Modules** de React Native pour communiquer avec le code natif (iOS/Android) qui exécute l'inférence YOLO.

## Architecture

```
┌─────────────────────────────────────────────┐
│           React Native (JavaScript)         │
│  ┌─────────────┐    ┌──────────────────┐    │
│  │ CameraFeed  │───▶│ YoloDetector     │    │
│  │  Component  │    │ (NativeModules)  │    │
│  └─────────────┘    └────────┬─────────┘    │
│                              │               │
├──────────────────────────────┼───────────────┤
│          Native Module Bridge               │
├──────────────────────────────┼───────────────┤
│  iOS (RCTBridgeModule)      │ Android (@ReactMethod)│
│  ┌──────────────────┐       │ ┌──────────────────┐
│  │ CoreML Inference │◀──────┘ │ LiteRT Inference │
│  │ (YOLO26.mlmodel)│         │ (YOLO26.tflite)  │
│  └──────────────────┘         └──────────────────┘
└─────────────────────────────────────────────┘
```

**Flux de données :**
1. React Native capture une image via la caméra
2. L'image est envoyée au module natif via le pont (bridge)
3. Le natif charge le modèle (CoreML sur iOS, LiteRT sur Android)
4. L'inférence est exécutée sur l'image
5. Les résultats sont renvoyés à JavaScript
6. React Native affiche les annotations

---

## 1. Côté iOS — Objective-C + Swift (RCTBridgeModule + CoreML)

> **Note :** Le pont Objective-C est **obligatoire** pour exposer un module Swift à React Native. Ne pas utiliser de code purement Swift sans le fichier `.m`.

### Fichier `ios/YoloModule.swift`

```swift
import Foundation
import Vision
import CoreML
import UIKit

@objc(YoloModule)
class YoloModule: NSObject {

  private var model: VNCoreMLModel?
  private var isModelLoaded = false
  private let processingQueue = DispatchQueue(label: "com.yolo.inference", qos: .userInitiated, attributes: .concurrent)

  @objc static func requiresMainQueueSetup() -> Bool {
    return false
  }

  /// Charger le modèle YOLO depuis un fichier bundle
  @objc func loadModel(
    _ modelPath: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    processingQueue.async { [weak self] in
      guard let self = self else { return }
      do {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .all

        // Si le chemin est un nom de fichier dans le bundle
        let url: URL
        if FileManager.default.fileExists(atPath: modelPath) {
          url = URL(fileURLWithPath: modelPath)
        } else {
          guard let bundleURL = Bundle.main.url(forResource: modelPath, withExtension: "mlmodelc") else {
            reject("MODEL_ERROR", "Modèle '\(modelPath)' introuvable dans le bundle", nil)
            return
          }
          url = bundleURL
        }

        let mlModel = try MLModel(contentsOf: url, configuration: modelConfig)
        self.model = try VNCoreMLModel(for: mlModel)
        self.isModelLoaded = true
        resolve(true)
      } catch {
        reject("MODEL_ERROR", "Impossible de charger le modèle: \(error.localizedDescription)", error)
      }
    }
  }

  /// Exécuter la détection sur une image (JPEG data)
  @objc func detect(
    _ imageData: NSArray,
    width: Int,
    height: Int,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    guard let model = model, isModelLoaded else {
      reject("MODEL_NOT_LOADED", "Modèle non chargé. Appelez loadModel() d'abord.", nil)
      return
    }

    processingQueue.async { [weak self] in
      guard let self = self else { return }

      // Convertir NSArray [UInt8] → Data
      var data = Data(count: imageData.count)
      data.withUnsafeMutableBytes { rawBuffer in
        guard let ptr = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return }
        for i in 0..<imageData.count {
          ptr[i] = (imageData[i] as! NSNumber).uint8Value
        }
      }

      guard let image = UIImage(data: data),
            let cgImage = image.cgImage else {
        reject("IMAGE_ERROR", "Impossible de décoder l'image. Vérifiez le format JPEG.", nil)
        return
      }

      let request = VNCoreMLRequest(model: model) { request, error in
        guard error == nil else {
          reject("INFERENCE_ERROR", error?.localizedDescription, error)
          return
        }

        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
          resolve([])
          return
        }

        var detections: [[String: Any]] = []
        for observation in observations {
          guard let topLabel = observation.labels.first else { continue }
          let bbox = observation.boundingBox
          detections.append([
            "label": topLabel.identifier,
            "confidence": Float(topLabel.confidence),
            "x": Double(bbox.origin.x),
            "y": Double(1.0 - bbox.origin.y - bbox.height),
            "width": Double(bbox.width),
            "height": Double(bbox.height)
          ])
        }
        resolve(detections)
      }

      request.imageCropAndScaleOption = .scaleFill

      let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
      do {
        try handler.perform([request])
      } catch {
        reject("INFERENCE_ERROR", error.localizedDescription, error)
      }
    }
  }

  /// Libérer le modèle de la mémoire
  @objc func unloadModel(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    self.model = nil
    self.isModelLoaded = false
    resolve(true)
  }

  /// Retourne l'état du modèle
  @objc func isLoaded(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    resolve(isModelLoaded)
  }
}
```

### Fichier de pont Objective-C `ios/YoloModule.m`

```objc
#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(YoloModule, NSObject)

RCT_EXTERN_METHOD(loadModel:(NSString *)modelPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(detect:(NSArray *)imageData
                  width:(NSInteger)width
                  height:(NSInteger)height
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(unloadModel:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(isLoaded:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
```

### Enregistrement (iOS)

Dans `ios/YoloApp/AppDelegate.mm`, ajouter :

```objc
#import "YoloModule.h"  // Généré automatiquement par React Native

// Pas d'enregistrement explicite nécessaire — React Native détecte
// automatiquement les modules avec RCT_EXTERN_MODULE via le linking.
```

Assurez-vous que le modèle `.mlmodelc` est dans votre bundle Xcode (drag & drop dans le projet, cocher "Copy items if needed").

---

## 2. Côté Android — Kotlin (@ReactMethod + LiteRT)

### Fichier `android/app/src/main/java/com/yoloapp/YoloModule.kt`

```kotlin
package com.yoloapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.facebook.react.bridge.*
import com.facebook.react.module.annotations.ReactModule
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

@ReactModule(name = YoloModule.NAME)
class YoloModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "YoloModule"
        private const val INPUT_SIZE = 640
    }

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private val context: Context = reactContext.applicationContext
    private val executor = Executors.newSingleThreadExecutor()

    override fun getName(): String = NAME

    @ReactMethod
    fun loadModel(modelPath: String, promise: Promise) {
        executor.execute {
            try {
                if (isModelLoaded) {
                    interpreter?.close()
                }

                val options = Interpreter.Options().apply {
                    setNumThreads(Runtime.getRuntime().availableProcessors())
                }

                val modelBuffer = loadModelFile(modelPath)
                interpreter = Interpreter(modelBuffer, options)
                isModelLoaded = true
                promise.resolve(true)
            } catch (e: Exception) {
                promise.reject("MODEL_ERROR", "Impossible de charger le modèle: ${e.message}", e)
            }
        }
    }

    private fun loadModelFile(path: String): MappedByteBuffer {
        // Essayer d'abord les assets, sinon le filesystem
        return try {
            val fileDescriptor = context.assets.openFd(path)
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
            // Sinon, lire depuis le chemin absolu
            val file = java.io.File(path)
            if (!file.exists()) throw IllegalArgumentException("Fichier modèle introuvable: $path")
            val inputStream = FileInputStream(file)
            val fileChannel = inputStream.channel
            fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).also {
                fileChannel.close()
            }
        }
    }

    @ReactMethod
    fun detect(imageData: ReadableArray, width: Int, height: Int, promise: Promise) {
        val interp = interpreter
        if (interp == null || !isModelLoaded) {
            promise.reject("MODEL_NOT_LOADED", "Modèle non chargé. Appelez loadModel() d'abord.")
            return
        }

        executor.execute {
            try {
                // 1. Convertir ReadableArray → ByteArray
                val bytes = ByteArray(imageData.size())
                for (i in 0 until imageData.size()) {
                    bytes[i] = imageData.getInt(i).toByte()
                }

                // 2. Décoder le JPEG → Bitmap
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?: throw IllegalArgumentException("Impossible de décoder l'image JPEG")

                // 3. Redimensionner en 640x640
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

                // 4. Prétraiter : RGB float [0.0, 1.0]
                val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
                inputBuffer.order(ByteOrder.nativeOrder())

                val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
                resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

                for (pixel in pixels) {
                    inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
                    inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
                    inputBuffer.putFloat((pixel and 0xFF) / 255.0f)           // B
                }

                // 5. Buffer de sortie : format YOLO [1, 84, 8400] (ou [1, 8400, 84])
                val outputBuffer = Array(1) { Array(84) { FloatArray(8400) } }

                // 6. Inférence
                interp.run(inputBuffer, outputBuffer)

                // 7. Post-traitement : NMS simplifié
                val detections = postProcess(outputBuffer[0], width.toFloat(), height.toFloat())

                // 8. Convertir en WritableArray pour React Native
                val result = Arguments.createArray()
                for (det in detections) {
                    val map = Arguments.createMap()
                    map.putString("label", det["label"] as String)
                    map.putDouble("confidence", det["confidence"] as Double)
                    map.putDouble("x", det["x"] as Double)
                    map.putDouble("y", det["y"] as Double)
                    map.putDouble("width", det["width"] as Double)
                    map.putDouble("height", det["height"] as Double)
                    result.pushMap(map)
                }

                promise.resolve(result)

                // Nettoyage
                resizedBitmap.recycle()
                bitmap.recycle()
            } catch (e: Exception) {
                promise.reject("INFERENCE_ERROR", e.message, e)
            }
        }
    }

    private fun postProcess(
        output: Array<FloatArray>,
        imgWidth: Float,
        imgHeight: Float
    ): List<Map<String, Any>> {
        val detections = mutableListOf<Map<String, Any>>()
        val confidenceThreshold = 0.5f

        // YOLO format: output[84][8400] → chaque colonne = [cx, cy, w, h, class_scores...]
        for (i in 0 until 8400) {
            // Trouver la classe avec le score max (indices 4..83)
            var maxScore = 0f
            var maxClassIdx = 0
            for (c in 4 until 84) {
                if (output[c][i] > maxScore) {
                    maxScore = output[c][i]
                    maxClassIdx = c
                }
            }

            if (maxScore < confidenceThreshold) continue

            val cx = output[0][i]
            val cy = output[1][i]
            val w = output[2][i]
            val h = output[3][i]

            // Convertir en coordonnées normalisées [0..1]
            val x = (cx - w / 2) / INPUT_SIZE
            val y = (cy - h / 2) / INPUT_SIZE
            val width = w / INPUT_SIZE
            val height = h / INPUT_SIZE

            val className = getClassName(maxClassIdx - 4)

            detections.add(mapOf(
                "label" to className,
                "confidence" to maxScore.toDouble(),
                "x" to x.toDouble(),
                "y" to y.toDouble(),
                "width" to width.toDouble(),
                "height" to height.toDouble()
            ))
        }

        // Simple NMS (Non-Maximum Suppression)
        return nms(detections, 0.45f)
    }

    private fun nms(detections: List<Map<String, Any>>, iouThreshold: Float): List<Map<String, Any>> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it["confidence"] as Double }
        val selected = mutableListOf<Map<String, Any>>()
        val active = BooleanArray(sorted.size) { true }

        for (i in sorted.indices) {
            if (!active[i]) continue
            selected.add(sorted[i])

            val boxA = sorted[i]
            for (j in i + 1 until sorted.size) {
                if (!active[j]) continue
                val boxB = sorted[j]
                if (computeIoU(boxA, boxB) > iouThreshold) {
                    active[j] = false
                }
            }
        }
        return selected
    }

    private fun computeIoU(a: Map<String, Any>, b: Map<String, Any>): Float {
        val ax = (a["x"] as Double).toFloat()
        val ay = (a["y"] as Double).toFloat()
        val aw = (a["width"] as Double).toFloat()
        val ah = (a["height"] as Double).toFloat()

        val bx = (b["x"] as Double).toFloat()
        val by = (b["y"] as Double).toFloat()
        val bw = (b["width"] as Double).toFloat()
        val bh = (b["height"] as Double).toFloat()

        val x1 = maxOf(ax, bx)
        val y1 = maxOf(ay, by)
        val x2 = minOf(ax + aw, bx + bw)
        val y2 = minOf(ay + ah, by + bh)

        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val union = aw * ah + bw * bh - intersection

        return if (union > 0f) intersection / union else 0f
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

    @ReactMethod
    fun unloadModel(promise: Promise) {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        promise.resolve(true)
    }

    @ReactMethod
    fun isLoaded(promise: Promise) {
        promise.resolve(isModelLoaded)
    }
}
```

### Fichier `android/app/src/main/java/com/yoloapp/YoloPackage.kt`

```kotlin
package com.yoloapp

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class YoloPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        return listOf(YoloModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}
```

### Enregistrement du module dans `MainApplication.kt`

```kotlin
override fun getPackages(): List<ReactPackage> {
    return PackageList(this).packages.apply {
        add(YoloPackage())
    }
}
```

---

## 3. Wrapper JavaScript/TypeScript — Appel aux NativeModules

### Fichier `src/services/yoloService.ts`

```typescript
import { NativeModules, Platform } from 'react-native';

interface NativeDetection {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface NativeYoloModule {
  loadModel(modelPath: string): Promise<boolean>;
  detect(imageData: number[], width: number, height: number): Promise<NativeDetection[]>;
  unloadModel(): Promise<boolean>;
  isLoaded(): Promise<boolean>;
}

const { YoloModule } = NativeModules as { YoloModule: NativeYoloModule };

if (!YoloModule) {
  throw new Error(
    'YoloModule non trouvé. Assurez-vous que le module natif est correctement lié.\n' +
    'iOS: vérifiez le fichier YoloModule.m et le bridging header.\n' +
    'Android: vérifiez YoloPackage dans MainApplication.'
  );
}

export class YoloDetection {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;

  constructor(label: string, confidence: number, x: number, y: number, width: number, height: number) {
    this.label = label;
    this.confidence = confidence;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }

  static fromNative(native: NativeDetection): YoloDetection {
    return new YoloDetection(
      native.label,
      native.confidence,
      native.x,
      native.y,
      native.width,
      native.height
    );
  }
}

export class YoloService {
  private static modelReady = false;

  static async loadModel(modelPath: string): Promise<boolean> {
    try {
      const result = await YoloModule.loadModel(modelPath);
      YoloService.modelReady = true;
      return result;
    } catch (error: any) {
      YoloService.modelReady = false;
      throw new Error(`Erreur chargement modèle: ${error.message}`);
    }
  }

  static async detect(imageData: number[], width: number, height: number): Promise<YoloDetection[]> {
    if (!YoloService.modelReady) {
      throw new Error('Modèle non chargé. Appelez YoloService.loadModel() d\'abord.');
    }
    try {
      const results = await YoloModule.detect(imageData, width, height);
      if (!results || !Array.isArray(results)) return [];
      return results.map((r: NativeDetection) => YoloDetection.fromNative(r));
    } catch (error: any) {
      throw new Error(`Erreur détection: ${error.message}`);
    }
  }

  static async unloadModel(): Promise<void> {
    await YoloModule.unloadModel();
    YoloService.modelReady = false;
  }

  static async isLoaded(): Promise<boolean> {
    return YoloModule.isLoaded();
  }
}

export default YoloService;
```

---

## 4. Composant CameraView complet avec inférence temps réel

### Fichier `src/components/CameraView.tsx`

```tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { View, StyleSheet, Text, Dimensions } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  Frame,
} from 'react-native-vision-camera';
import { YoloService, YoloDetection } from '../services/yoloService';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const MAX_FPS = 10;
const FRAME_INTERVAL_MS = 1000 / MAX_FPS;

interface CameraViewProps {
  modelPath: string;
  onDetections?: (detections: YoloDetection[]) => void;
  showOverlay?: boolean;
}

const CameraView: React.FC<CameraViewProps> = ({
  modelPath,
  onDetections,
  showOverlay = true,
}) => {
  const [detections, setDetections] = useState<YoloDetection[]>([]);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastFrameTime = useRef(0);
  const isProcessing = useRef(false);

  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  // Charger le modèle au montage
  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        await YoloService.loadModel(modelPath);
        if (mounted) setIsModelLoaded(true);
      } catch (err: any) {
        if (mounted) setError(err.message);
      }
    };
    init();
    return () => {
      mounted = false;
      YoloService.unloadModel();
    };
  }, [modelPath]);

  // Demander la permission caméra
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Frame processor throttled
  const frameProcessor = useCallback(
    (frame: Frame) => {
      const now = Date.now();
      if (now - lastFrameTime.current < FRAME_INTERVAL_MS) return;
      if (isProcessing.current) return;

      isProcessing.current = true;
      lastFrameTime.current = now;

      try {
        // Convertir la frame en données JPEG
        const buffer = frame.toArrayBuffer();
        const imageData = Array.from(new Uint8Array(buffer));

        YoloService.detect(imageData, frame.width, frame.height)
          .then((results) => {
            setDetections(results);
            onDetections?.(results);
          })
          .catch((err) => {
            console.warn('[CameraView] Erreur détection:', err.message);
          })
          .finally(() => {
            isProcessing.current = false;
          });
      } catch {
        isProcessing.current = false;
      }
    },
    [onDetections]
  );

  // --- États d'affichage ---

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Erreur: {error}</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permText}>Permission caméra requise</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permText}>Aucun appareil caméra disponible</Text>
      </View>
    );
  }

  if (!isModelLoaded) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permText}>Chargement du modèle...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />

      {showOverlay &&
        detections.map((det, index) => (
          <DetectionBox key={`${det.label}-${index}`} detection={det} />
        ))}
    </View>
  );
};

// --- Sous-composant boîte de détection ---

const DetectionBox: React.FC<{ detection: YoloDetection }> = React.memo(({ detection }) => {
  const color = getColorForClass(detection.label);

  return (
    <View
      style={[
        styles.boundingBox,
        {
          left: `${detection.x * 100}%`,
          top: `${detection.y * 100}%`,
          width: `${detection.width * 100}%`,
          height: `${detection.height * 100}%`,
          borderColor: color,
        },
      ]}
    >
      <View style={[styles.label, { backgroundColor: color }]}>
        <Text style={styles.labelText}>
          {detection.label} {(detection.confidence * 100).toFixed(0)}%
        </Text>
      </View>
    </View>
  );
});

function getColorForClass(className: string): string {
  const colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFA500',
    '#800080', '#008080', '#FFC0CB', '#FFD700',
  ];
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  errorText: {
    color: '#FF6B6B',
    textAlign: 'center',
    padding: 16,
  },
  permText: {
    color: '#FFF',
    textAlign: 'center',
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  label: {
    position: 'absolute',
    top: -24,
    left: 0,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  labelText: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: 'bold',
  },
});

export default CameraView;
```

---

## 5. Écran de détection complet

### Fichier `src/screens/DetectionScreen.tsx`

```tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { YoloService } from '../services/yoloService';
import CameraView from '../components/CameraView';

const MODEL_PATH_IOS = 'yolo26n';  // Nom du .mlmodelc dans le bundle
const MODEL_PATH_ANDROID = 'yolo26n.tflite';  // Dans android/app/src/main/assets/

const DetectionScreen: React.FC = () => {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadModel = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const path = Platform.OS === 'ios' ? MODEL_PATH_IOS : MODEL_PATH_ANDROID;
      await YoloService.loadModel(path);
      setIsModelLoaded(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadModel();
  }, []);

  return (
    <View style={styles.container}>
      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#FFF" />
          <Text style={styles.loadingText}>Chargement du modèle...</Text>
        </View>
      )}

      {error && (
        <View style={styles.errorOverlay}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={loadModel} style={styles.retryBtn}>
            <Text style={styles.retryText}>Réessayer</Text>
          </TouchableOpacity>
        </View>
      )}

      {isModelLoaded && (
        <CameraView
          modelPath={Platform.OS === 'ios' ? MODEL_PATH_IOS : MODEL_PATH_ANDROID}
          showOverlay={true}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.8)',
    zIndex: 10,
  },
  loadingText: { color: '#FFF', marginTop: 12 },
  errorOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.8)',
    zIndex: 10,
    padding: 24,
  },
  errorText: { color: '#FF6B6B', textAlign: 'center', marginBottom: 16 },
  retryBtn: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#4A90D9',
    borderRadius: 8,
  },
  retryText: { color: '#FFF', fontWeight: 'bold' },
});

export default DetectionScreen;
```

---

## 6. Optimisations Hermes

### Vérifier que Hermes est activé

**Android** — `android/app/build.gradle` :

```gradle
project.ext.react = [
    enableHermes: true,
]
```

**iOS** — `ios/Podfile` :

```ruby
:hermes_enabled => true
```

### Impact sur l'inférence YOLO

| Optimisation | Impact |
|---|---|
| **Bytecode compilé** | Démarrage ~50% plus rapide que JSC |
| **GC générational** | Moins de pauses GC pendant l'inférence |
| **Réduction bundle** | Bundle compressé plus petit en mémoire |
| **Typed Arrays natifs** | Les `ArrayBuffer` partagés avec le natif sont plus rapides |

### Bonnes pratiques Hermes + YOLO

```typescript
import { InteractionManager, Platform } from 'react-native';

// 1. Ne jamais bloquer le thread UI pour l'inférence
// L'inférence passe par NativeModules → thread natif dédié

// 2. Throttler les frames côté JS si nécessaire
const FRAME_INTERVAL = 100; // ms (10 FPS)

// 3. Utiliser InteractionManager pour charger le modèle après les animations
InteractionManager.runAfterInteractions(async () => {
  await YoloService.loadModel(modelPath);
});

// 4. Éviter les allocations dans la boucle de détection
const imageDataRef = useRef<number[]>([]);

// 5. Préférences Hermes pour les modules natifs
if (Platform.OS === 'android') {
  // Activer le JIT pour les computations intensives
  // (activé par défaut dans Hermes récent)
}
```

### Tuning Android (Gradle)

```gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }
    buildTypes {
        release {
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt')
        }
    }
}
```

---

## 7. Bonnes pratiques React Native

> **Référence :** Chargez le skill `react-native-best-practices` pour les optimisations détaillées.

### Performance

- **Éviter les re-rendus inutiles** : `React.memo`, `useCallback`, `useMemo`
- **Throttling des frames** : limiter à 10 FPS max pour éviter la surcharge CPU
- **FlashList** pour les listes de détections historiques (pas de FlatList)

```typescript
// Composant mémorisé
const DetectionBox = React.memo(({ detection }: { detection: YoloDetection }) => {
  // ...
});

// Callback mémorisé
const handleDetections = useCallback((dets: YoloDetection[]) => {
  setDetections(dets);
}, []);
```

### Mémoire

- **Décharger le modèle** au démontage : `useEffect(() => { return () => YoloService.unloadModel(); }, [])`
- **Limiter la fréquence** : max 10 FPS pour éviter la surcharge mémoire
- **Recycler les bitmaps** dans le code natif (iOS/Android)
- **Ne pas stocker les images** dans l'état React

```typescript
useEffect(() => {
  return () => {
    YoloService.unloadModel();
  };
}, []);
```

### Sécurité

- Ne jamais logger les données d'image brutes
- Valider les chemins de modèles (pas d'injection de chemins arbitraires)
- Le modèle doit être signé (iOS) ou vérifié (Android)

---

## 8. Problèmes courants (Common Issues)

### `NativeModules.YoloModule is undefined`

**Cause :** Le module natif n'est pas lié.

**Solutions :**
- **iOS** : Vérifiez que `YoloModule.m` est dans la cible Xcode. Nettoyez et reconstruisez (`Product → Clean Build Folder`).
- **Android** : Vérifiez que `YoloPackage` est ajouté dans `MainApplication.getPackages()`. Faites `./gradlew clean`.
- Vérifiez `react-native link` ou le linking automatique (RN 0.60+).

### Erreur de chargement du modèle (`MODEL_ERROR`)

**Cause :** Le fichier modèle n'est pas accessible ou est corrompu.

**Solutions :**
- **iOS** : Le `.mlmodelc` doit être dans le bundle. Vérifiez dans Xcode que le fichier est bien inclus dans la cible.
- **Android** : Le `.tflite` doit être dans `android/app/src/main/assets/`. Ne pas le mettre dans `res/`.
- Vérifiez la taille du fichier (un modèle YOLO26n fait ~6 Mo).

### Erreur `INFERENCE_ERROR` ou crash

**Cause :** Format d'image incorrect ou dimensions incompatibles.

**Solutions :**
- L'image doit être en **JPEG** ou **PNG** en format `UInt8[]`.
- Le buffer natif attend des images en RGB normalisé [0, 1].
- Vérifiez les dimensions du modèle (640x640 pour YOLO26n).

### `Module with name YoloModule has already been claimed`

**Cause :** Double enregistrement du module.

**Solution :** Vérifiez `MainApplication.kt` — le package ne doit être ajouté qu'une seule fois.

### Performance faible (< 5 FPS)

**Causes et solutions :**
- Vérifiez que Hermes est activé
- Réduisez la résolution de la caméra (480p suffit pour YOLO)
- Limitez le FPS à 10 dans le frame processor
- Utilisez `react-native-vision-camera` v4+ avec Worklets
- Sur Android, activez le GPU delegate : `Interpreter.Options().addDelegate(GpuDelegate())`

### L'application crash au démarrage après ajout du module

**Solutions :**
- Nettoyez complètement : `cd android && ./gradlew clean` puis `cd ios && pod deinstall && pod install`
- Vérifiez les versions de React Native (`react-native --version`)
- Assurez-vous que la version de LiteRT/TFLite est compatible avec votre SDK

### Le module fonctionne en développement mais pas en release

**Cause :** ProGuard/R8 supprime le module en release.

**Solution** — Ajouter dans `android/app/proguard-rules.pro` :

```proguard
-keep class com.yoloapp.YoloModule { *; }
-keep class com.yoloapp.YoloPackage { *; }
```

---

## 9. Fichiers de configuration

### `package.json` (dépendances)

```json
{
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.73.0",
    "react-native-vision-camera": "^4.0.0",
    "react-native-worklets-core": "^1.0.0"
  }
}
```

### Structure du projet

```
android/app/src/main/java/com/yoloapp/
├── YoloModule.kt        # Module natif Android
├── YoloPackage.kt       # Package React Native
└── MainApplication.kt   # Enregistrement du package

ios/
├── YoloModule.swift     # Module natif iOS
├── YoloModule.m         # Pont Objective-C (obligatoire)
└── AppDelegate.mm       # Enregistrement du module

src/
├── services/
│   └── yoloService.ts   # Wrapper TypeScript
├── components/
│   └── CameraView.tsx   # Composant caméra
└── screens/
    └── DetectionScreen.tsx  # Écran principal
```

---

## 10. Références

- `react-native-best-practices` — Optimisations de performance React Native
- [Documentation Ultralytics](https://docs.ultralytics.com/fr)
- [Guide d'exportation](https://docs.ultralytics.com/fr/modes/export)
- [Format LiteRT](https://docs.ultralytics.com/fr/integrations/litert)
- [Format CoreML](https://docs.ultralytics.com/fr/integrations/coreml)
- [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera)
