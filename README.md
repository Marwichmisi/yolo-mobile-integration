# YOLO26 Mobile Integration

[![skills.sh](https://skills.sh/b/Marwichmisi/yolo-mobile-integration)](https://skills.sh/Marwichmisi/yolo-mobile-integration)

Integrate YOLO26 computer vision models into mobile applications (Flutter, React Native, native iOS/Android) and edge devices (Raspberry Pi, Jetson, Coral).

## What It Does

This skill orchestrates the full pipeline from model selection to on-device deployment:

- **Model Selection** — Choose the right YOLO26 variant (nano/small/medium/large/xlarge) and task (detection, segmentation, pose, classification, OBB, semantic)
- **Export** — Convert to mobile formats (CoreML, LiteRT, NCNN, ExecuTorch, ONNX, TF.js)
- **Integration** — Generate platform-specific code (Flutter Platform Channels, React Native Native Modules, Swift Vision, Kotlin LiteRT)
- **Use Cases** — Implement real-world features (fitness coaching, security alarms, object counting, parking management, AR measurement, heatmaps, privacy blurring)
- **Edge Deployment** — Deploy on Raspberry Pi, NVIDIA Jetson, or Google Coral

## When to Use

Use this skill whenever you need to:

- Add object detection, segmentation, or pose estimation to a mobile app
- Export YOLO models to CoreML/TFLite/NCNN/ExecuTorch
- Build vision-powered mobile features (camera apps, fitness trackers, security systems)
- Deploy YOLO26 on edge devices (Raspberry Pi, Jetson, Coral)
- Create real-time computer vision applications on iOS, Android, or web

## Installation

```bash
npx skills add Marwichmisi/yolo-mobile-integration
```

Or install directly with OpenCode:

```bash
opencode skill install Marwichmisi/yolo-mobile-integration
```

## Available Skills

### yolo-mobile-integration

Complete workflow for integrating YOLO26 into mobile apps. Includes:

- **SKILL.md** — Main orchestration logic with context detection and routing
- **references/** — 14 detailed reference files:
  - `yolo26-model.md` — Model specs, training, architecture
  - `export-pipeline.md` — Full export pipeline with all formats
  - `platforms/ios-coreml.md` — iOS deployment with CoreML
  - `platforms/android-litert.md` — Android deployment with LiteRT
  - `platforms/android-ncnn.md` — Android deployment with NCNN
  - `platforms/cross-platform-executorch.md` — Cross-platform with ExecuTorch
  - `platforms/web-tfjs.md` — Browser-based detection with LiteRT.js
  - `frameworks/flutter-integration.md` — Flutter Platform Channel integration
  - `frameworks/react-native-integration.md` — React Native Native Module integration
  - `tasks/detection.md`, `segmentation.md`, `classification.md`, `pose-estimation.md`, `obb.md`, `semantic.md`
  - `use-cases/fitness-coach.md`, `security-alarm.md`, `object-counting.md`, `parking-management.md`, `distance-measurement.md`, `heatmap-analytics.md`, `object-blurring.md`
  - `hardware/raspberry-pi.md`, `nvidia-jetson.md`, `edge-tpu.md`
- **scripts/** — Automation tools:
  - `export_model.py` — Export YOLO26 to any mobile format
  - `benchmark_mobile.py` — Benchmark models on target devices
  - `create_project.py` — Scaffold Flutter/RN projects with YOLO

## Quick Start

### Export a Model

```bash
python scripts/export_model.py --model yolo26n --format coreml --quantize int8
```

### Scaffold a Flutter Project

```bash
python scripts/create_project.py --framework flutter --task detect --platform both
```

### Benchmark on Device

```bash
python scripts/benchmark_mobile.py --model yolo26n.mlpackage --device iphone
```

## Supported Platforms

| Platform | Format | Command |
|----------|--------|---------|
| iOS | CoreML | `export --format coreml --quantize int8` |
| Android | LiteRT | `export --format litert --quantize int8` |
| Android (ARM) | NCNN | `export --format ncnn` |
| Cross-platform | ExecuTorch | `export --format executorch` |
| Web | LiteRT.js | `export --format litert` + JS integration |
| Raspberry Pi | NCNN | `export --format ncnn` |
| Jetson | TensorRT | `export --format engine` |
| Coral | Edge TPU | `export --format edgetpu` |

## Supported Tasks

| Task | Model Variant | Use Case |
|------|--------------|----------|
| Detection | `yolo26{n,s,m,l,x}.pt` | Identify objects in camera feed |
| Segmentation | `yolo26{n,s,m,l,x}-seg.pt` | Pixel-level object masks |
| Classification | `yolo26{n,s,m,l,x}-cls.pt` | Categorize images |
| Pose | `yolo26{n,s,m,l,x}-pose.pt` | Human pose estimation |
| OBB | `yolo26{n,s,m,l,x}-obb.pt` | Oriented bounding boxes |
| Semantic | `yolo26{n,s,m,l,x}-sem.pt` | Pixel-level scene labels |

## Use Cases

| Use Case | Description | Reference |
|----------|-------------|-----------|
| Fitness Coach | Real-time exercise detection, rep counting, form analysis | `use-cases/fitness-coach.md` |
| Security Alarm | Motion detection, intrusion alerts, event logging | `use-cases/security-alarm.md` |
| Object Counting | People/vehicle counting with line-crossing logic | `use-cases/object-counting.md` |
| Parking Management | Spot detection, occupancy tracking | `use-cases/parking-management.md` |
| Distance Measurement | AR-based real-world measurement | `use-cases/distance-measurement.md` |
| Heatmap Analytics | People flow, dwell time visualization | `use-cases/heatmap-analytics.md` |
| Object Blurring | Privacy-preserving face/plate blurring | `use-cases/object-blurring.md` |

## Performance

YOLO26n on mobile devices:

| Device | Format | FPS | Latency |
|--------|--------|-----|---------|
| iPhone 17 Pro (Neural Engine) | CoreML | ~260 | 3.8ms |
| iPhone 17 Pro (CPU) | CoreML | ~110 | 9.1ms |
| Raspberry Pi 5 | NCNN | ~8 | 128ms |
| NVIDIA T4 | TensorRT FP16 | ~588 | 1.7ms |

## Companion Skills

This skill works best alongside:

- `flutter-apply-architecture-best-practices` — App architecture (MVVM, layered)
- `react-native-best-practices` — Performance optimization (Hermes, native modules)

## Repository Structure

```
yolo-mobile-integration/
├── README.md
├── skills/
│   └── yolo-mobile-integration/
│       ├── SKILL.md                    # Main orchestration
│       ├── references/                 # 14 reference files
│       │   ├── yolo26-model.md
│       │   ├── export-pipeline.md
│       │   ├── platforms/
│       │   ├── frameworks/
│       │   ├── tasks/
│       │   ├── use-cases/
│       │   └── hardware/
│       ├── scripts/                    # Automation tools
│       │   ├── export_model.py
│       │   ├── benchmark_mobile.py
│       │   └── create_project.py
│       └── evals/
│           └── evals.json
└── yolo-mobile-integration-workspace/  # Evaluation results
```

## License

MIT

## Author

[Marwichmisi](https://github.com/Marwichmisi)
