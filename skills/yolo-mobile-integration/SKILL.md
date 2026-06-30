---
name: yolo-mobile-integration
description: >
  Orchestrates the integration of YOLO26 (Ultralytics) into mobile applications for real-time computer vision.
  Use this skill whenever the user wants to: add object detection/segmentation/pose estimation to a mobile app,
  export YOLO models to CoreML/TFLite/NCNN/ExecuTorch, build a vision-powered mobile feature (fitness coach,
  security camera, object counter, parking manager, AR measurement, heatmap analytics), or deploy YOLO26 on
  edge devices (Raspberry Pi, Jetson, Coral). Also triggers when the user mentions YOLO + Flutter,
  YOLO + React Native, YOLO + iOS, YOLO + Android, mobile AI vision, on-device inference, or real-time
  object detection on mobile. This skill coordinates with existing Flutter and React Native skills.
metadata:
  author: opencode
  tags: [yolo, mobile, vision, ai, detection, segmentation, pose, coreml, tflite, ncnn, flutter, react-native]
  last_modified: "2026-06-30"
---

# YOLO26 Mobile Integration

End-to-end workflow for integrating YOLO26 vision models into mobile applications.

## CRITICAL: Before You Start

**You MUST read ALL relevant reference files before generating any code.** The references contain platform-specific code templates, export commands, and integration patterns that you need to follow. Do NOT rely on general knowledge — the reference files have the exact code for YOLO26 integration.

**You MUST also load companion skills** when the user's project uses Flutter or React Native. These skills provide app architecture, testing, and performance patterns that complement the YOLO integration.

---

## Step 0: Load All Relevant Resources

Before writing ANY code, execute this checklist:

### 1. Detect Framework → Load Reference + Companion Skill

| Framework | Reference to READ | Companion Skill to LOAD |
|-----------|------------------|------------------------|
| **Flutter** | READ `references/frameworks/flutter-integration.md` | **ALSO LOAD** the skill at `.agents/skills/flutter-apply-architecture-best-practices/SKILL.md` — it provides app architecture patterns (MVVM, Repository) |
| **React Native** | READ `references/frameworks/react-native-integration.md` | **ALSO LOAD** the skill at `.agents/skills/react-native-best-practices/SKILL.md` — it provides performance optimization (Hermes, native modules) |
| **Native Swift** | READ `references/platforms/ios-coreml.md` | No companion skill needed |
| **Native Kotlin** | READ `references/platforms/android-litert.md` | No companion skill needed |
| **Web/WebView** | READ `references/platforms/web-tfjs.md` | No companion skill needed |

### 2. Detect Task → Load Task Reference

| Task | Reference to READ |
|------|------------------|
| Detection (objets) | READ `references/tasks/detection.md` |
| Segmentation | READ `references/tasks/segmentation.md` |
| Classification | READ `references/tasks/classification.md` |
| Pose estimation (sport, exercice) | READ `references/tasks/pose-estimation.md` |
| Oriented boxes (documents, aérien) | READ `references/tasks/obb.md` |
| Semantic segmentation | READ `references/tasks/semantic.md` |

### 3. Detect Use Case → Load Use Case Reference

| Use Case | Reference to READ |
|----------|------------------|
| Fitness / sport / exercice | READ `references/use-cases/fitness-coach.md` |
| Sécurité / alarme | READ `references/use-cases/security-alarm.md` |
| Comptage d'objets | READ `references/use-cases/object-counting.md` |
| Parking | READ `references/use-cases/parking-management.md` |
| Mesure / AR | READ `references/use-cases/distance-measurement.md` |
| Heatmap / analytics | READ `references/use-cases/heatmap-analytics.md` |
| Floutage / vie privée | READ `references/use-cases/object-blurring.md` |

### 4. Detect Hardware → Load Hardware Reference

| Hardware | Reference to READ |
|----------|------------------|
| Raspberry Pi | READ `references/hardware/raspberry-pi.md` |
| NVIDIA Jetson | READ `references/hardware/nvidia-jetson.md` |
| Google Coral / Edge TPU | READ `references/hardware/edge-tpu.md` |

### 5. Always Load Model Reference

**ALWAYS READ** `references/yolo26-model.md` for model selection and `references/export-pipeline.md` for export instructions.

---

## Step 1: Choose the Model

Read `references/yolo26-model.md`. Quick decision table:

| Constraint | Model | Why |
|-----------|-------|-----|
| Mobile real-time, CPU only | **YOLO26n** | 3M params, fastest inference |
| Mobile with GPU/NPU | **YOLO26s** | Good balance speed/accuracy |
| High accuracy, cloud/hybrid | **YOLO26m** or **YOLO26l** | More parameters, better mAP |
| Maximum accuracy | **YOLO26x** | Server-grade |
| Tiny objects (small targets) | **YOLO26n-p2** | Extra feature pyramid level |
| Wide input (satellite, panorama) | **YOLO26-p6** | Large receptive field |

**Task variants** (replace `yolo26n.pt` with the right variant):
- Detection: `yolo26{n,s,m,l,x}.pt` (default, 80 COCO classes)
- Segmentation: `yolo26{n,s,m,l,x}-seg.pt` (pixel masks + boxes)
- Classification: `yolo26{n,s,m,l,x}-cls.pt` (single-label)
- Pose: `yolo26{n,s,m,l,x}-pose.pt` (17 keypoints, skeleton)
- OBB: `yolo26{n,s,m,l,x}-obb.pt` (rotated boxes)
- Semantic: `yolo26{n,s,m,l,x}-sem.pt` (pixel labels)

---

## Step 2: Export to Mobile Format

Read the export reference from `references/export-pipeline.md` and the platform-specific reference.

| Target | Reference to READ | Export Command |
|--------|------------------|----------------|
| iOS only | READ `references/platforms/ios-coreml.md` | `model.export(format="coreml", quantize=8)` |
| Android only | READ `references/platforms/android-litert.md` | `model.export(format="litert", quantize=8)` |
| Both iOS + Android | READ `references/platforms/cross-platform-executorch.md` | `model.export(format="executorch")` |
| Both (optimized ARM) | READ `references/platforms/android-ncnn.md` | `model.export(format="ncnn")` |
| Web browser | READ `references/platforms/web-tfjs.md` | `model.export(format="litert")` then use LiteRT.js |

**Quantization recommendation:**
- INT8 (`quantize=8`): Best compression, needs calibration data — use for production
- FP16 (`half=True`): Good compression, no calibration — use for quick tests
- W8A32 (`quantize="w8a32"`): Dynamic INT8, no calibration needed — good default for Android

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
# For iOS:
model.export(format="coreml", quantize=8)
# For Android:
model.export(format="litert", quantize=8)
# For cross-platform:
model.export(format="ncnn")
```

---

## Step 3: Integrate into Mobile App

### Flutter Integration

**You MUST READ `references/frameworks/flutter-integration.md` first.** It contains:
- Complete Platform Channel code (Dart + Swift + Kotlin)
- Camera integration with real-time inference
- Bounding box overlay rendering
- State management patterns

**You MUST ALSO LOAD the Flutter architecture skill** at `.agents/skills/flutter-apply-architecture-best-practices/SKILL.md` for:
- App structure (MVVM, layered architecture)
- Repository pattern for data access
- Provider/Bloc/Riverpod state management

The integration pattern:
1. Native side (iOS/Android) loads the YOLO model
2. Dart side calls native via MethodChannel
3. Camera frames are sent to native for inference
4. Results (boxes, classes, confidence) are returned to Dart
5. UI renders the overlay

### React Native Integration

**You MUST READ `references/frameworks/react-native-integration.md` first.** It contains:
- Complete Native Module code (Objective-C/Kotlin + JS)
- Camera integration for real-time detection
- Centroid tracking for object counting
- Line-crossing detection logic

**You MUST ALSO LOAD the React Native performance skill** at `.agents/skills/react-native-best-practices/SKILL.md` for:
- Hermes optimization
- Native module best practices
- Memory and re-render optimization

The integration pattern:
1. Native module wraps YOLO inference (CoreML on iOS, LiteRT on Android)
2. JS side calls native via NativeModules
3. Camera frames processed on native thread
4. Results passed back to JS for UI updates

### Web Integration

**You MUST READ `references/platforms/web-tfjs.md` first.** It contains:
- LiteRT.js (successor to TF.js) integration
- Camera access via getUserMedia
- WebGPU/WASM backend selection
- Service Worker for offline caching

---

## Step 4: Use Case Implementation

Each use case has specific logic beyond basic detection. **READ the relevant use case reference** to get the implementation patterns:

| Use Case | What the Reference Contains |
|----------|---------------------------|
| `fitness-coach.md` | Joint angle calculation, exercise phase detection (up/down), rep counting algorithm, form scoring (0-100%), voice feedback |
| `security-alarm.md` | Motion detection zones, alert triggers, notification system, event logging |
| `object-counting.md` | Centroid tracking, line-crossing logic, bidirectional counting (IN/OUT), polygon zones |
| `parking-management.md` | Spot annotation, occupancy detection, guidance system, status display |
| `distance-measurement.md` | Pixel-to-real-world conversion, depth estimation, AR overlay |
| `heatmap-analytics.md` | Grid-based accumulation, temporal decay, visualization overlay |
| `object-blurring.md` | Mask generation, Gaussian blur application, real-time redaction |

---

## Step 5: Optimize and Validate

1. **Benchmark** on target device using `scripts/benchmark_mobile.py`
2. **If too slow**: reduce input size (640→320), use INT8 quantization, switch to Nano variant
3. **If too large**: prune, quantize, or use knowledge distillation (see `references/yolo26-model.md`)
4. **FPS targets**: ≥15 FPS for real-time, ≥30 FPS for smooth video
5. **Test on real devices** — not just emulators
6. **Validate accuracy** on representative test images

---

## Step 6: Hardware Deployment (Edge Devices)

If deploying to edge hardware instead of phones:

| Hardware | Reference to READ | Format | Expected Performance |
|----------|------------------|--------|---------------------|
| Raspberry Pi 5 | READ `references/hardware/raspberry-pi.md` | NCNN | ~8 FPS (YOLO26n) |
| NVIDIA Jetson | READ `references/hardware/nvidia-jetson.md` | TensorRT | 50+ FPS (YOLO26n) |
| Google Coral | READ `references/hardware/edge-tpu.md` | Edge TPU | ~30 FPS (YOLO26n) |

---

## Quick Reference: Export Formats

| Format | iOS | Android | Web | Quantization | Notes |
|--------|-----|---------|-----|-------------|-------|
| CoreML | ✅ Native | ❌ | ❌ | FP16, INT8 | Uses Apple Neural Engine |
| LiteRT | ✅ | ✅ Native | ✅ LiteRT.js | INT8, w8a32 | Universal mobile format |
| NCNN | ✅ | ✅ Native | ❌ | FP16 | Best for ARM, very small binary |
| ExecuTorch | ✅ | ✅ | ❌ | FP32 only | PyTorch ecosystem |
| ONNX | Via inter. | Via inter. | ✅ | FP16, INT8 | Universal interchange |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Export fails "NMS not supported" | YOLO26 is end-to-end, NMS disabled by default. Use default settings. |
| Model too large for mobile | Use Nano variant + INT8 quantization + reduce input to 320 |
| Low FPS on device | Switch to smaller model, reduce resolution, use GPU/NPU delegate |
| Poor accuracy on custom data | Finetune on your dataset, increase image size, use larger variant |
| CoreML export requires macOS | Run export on macOS, or use ONNX as intermediate |
| LiteRT quantization needs calibration | Use `quantize="w8a32"` for dynamic quantification (no calibration) |
| Camera not initializing | Check permissions in AndroidManifest.xml / Info.plist |
| Flutter Platform Channel not working | Ensure native side registers the method call handler in AppDelegate/MainActivity |
