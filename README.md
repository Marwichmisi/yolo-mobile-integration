<div align="center">

# 🚀 YOLO26 Mobile Integration

### *The Ultimate Skill for Integrating YOLO26 Vision Models into Mobile Applications*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-red.svg)](https://github.com/ultralytics/ultralytics)
[![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)](https://flutter.dev)
[![React Native](https://img.shields.io/badge/React%20Native-0.70+-blue.svg)](https://reactnative.dev)
[![iOS](https://img.shields.io/badge/iOS-15+-black.svg)](https://developer.apple.com/ios/)
[![Android](https://img.shields.io/badge/Android-8+-green.svg)](https://developer.android.com)
[![skills.sh](https://skills.sh/b/Marwichmisi/yolo-mobile-integration)](https://skills.sh/Marwichmisi/yolo-mobile-integration)
[![Stars](https://img.shields.io/github/stars/Marwichmisi/yolo-mobile-integration?style=social)](https://github.com/Marwichmisi/yolo-mobile-integration/stargazers)
[![Forks](https://img.shields.io/github/forks/Marwichmisi/yolo-mobile-integration?style=social)](https://github.com/Marwichmisi/yolo-mobile-integration/network/members)

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-supported-tasks">Supported Tasks</a> •
  <a href="#-export-formats">Export Formats</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-use-cases">Use Cases</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://assets.ultralytics.com/yolo26/banner.png" alt="YOLO26 Mobile Integration" width="800">
</p>

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🎯 Supported Tasks](#-supported-tasks)
- [📦 Export Formats](#-export-formats)
- [📊 Benchmarks](#-benchmarks)
- [💡 Use Cases](#-use-cases)
- [📱 Platform Support](#-platform-support)
- [🔧 Installation](#-installation)
- [📖 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **7 Vision Tasks** | Detection, Segmentation, Semantic, Depth, Classification, Pose, OBB |
| 📱 **Multi-Platform** | iOS (CoreML), Android (LiteRT), Flutter, React Native, Web |
| ⚡ **Real-Time** | Optimized for on-device inference with NMS-free architecture |
| 🔌 **22 Export Formats** | CoreML, LiteRT, NCNN, ExecuTorch, ONNX, TensorRT, and more |
| 🎨 **Production-Ready** | Complete code templates for all major platforms |
| 📚 **Comprehensive Docs** | 25+ reference files with examples and benchmarks |
| 🔧 **Helper Scripts** | Export, benchmark, and project scaffolding tools |
| 🌐 **Edge Ready** | Raspberry Pi, NVIDIA Jetson, Google Coral support |

---

## 🚀 Quick Start

### 📦 Install from skills.sh

The easiest way to install this skill is via the [skills.sh](https://skills.sh) CLI:

```bash
# Install the skill
npx skills add Marwichmisi/yolo-mobile-integration

# Or install to a specific agent (e.g., OpenCode, Claude Code, Codex)
npx skills add Marwichmisi/yolo-mobile-integration -a opencode

# Or install globally
npx skills add Marwichmisi/yolo-mobile-integration -g
```

> **Note**: The skills CLI supports [OpenCode](https://opencode.ai), [Claude Code](https://claude.ai), [Codex](https://openai.com/codex), [Cursor](https://cursor.com), and 72+ more agents.

### 1. Install Ultralytics

```bash
pip install ultralytics
```

### 2. Export YOLO26 for Mobile

```python
from ultralytics import YOLO

# Load YOLO26 model
model = YOLO("yolo26n.pt")

# Export for iOS (CoreML)
model.export(format="coreml", quantize=8)

# Export for Android (LiteRT)
model.export(format="litert", quantize="w8a32")

# Export for cross-platform (NCNN)
model.export(format="ncnn")
```

### 3. Integrate into Your App

Choose your platform:

| Platform | SDK | Documentation |
|----------|-----|---------------|
| **iOS** | [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) | [Guide](skills/yolo-mobile-integration/references/platforms/ios-coreml.md) |
| **Flutter** | [Ultralytics YOLO Flutter Plugin](https://github.com/ultralytics/yolo-flutter-app) | [Guide](skills/yolo-mobile-integration/references/frameworks/flutter-integration.md) |
| **React Native** | Custom Native Modules | [Guide](skills/yolo-mobile-integration/references/frameworks/react-native-integration.md) |
| **Web** | [@ultralytics/yolo NPM](https://www.npmjs.com/package/@ultralytics/yolo) | [Guide](skills/yolo-mobile-integration/references/platforms/web-tfjs.md) |

---

## 🎯 Supported Tasks

| Task | Model Variants | Description | Mobile Use Case |
|------|---------------|-------------|-----------------|
| **Detection** | `yolo26{n,s,m,l,x}.pt` | Object detection with bounding boxes | Security, counting, tracking |
| **Segmentation** | `yolo26{n,s,m,l,x}-seg.pt` | Instance segmentation with pixel masks | Medical, AR, photo editing |
| **Semantic** | `yolo26{n,s,m,l,x}-sem.pt` | Semantic segmentation (per-pixel class) | Scene understanding |
| **Depth** | `yolo26{n,s,m,l,x}-depth.pt` | Monocular depth estimation (meters) | AR, photography, navigation |
| **Classification** | `yolo26{n,s,m,l,x}-cls.pt` | Image classification | Content moderation, sorting |
| **Pose** | `yolo26{n,s,m,l,x}-pose.pt` | 17-keypoint pose estimation | Fitness, sports, animation |
| **OBB** | `yolo26{n,s,m,l,x}-obb.pt` | Oriented bounding boxes | Aerial imagery, documents |

---

## 📦 Export Formats

### Mobile-Optimized Formats

| Format | iOS | Android | Web | Quantization | Best For |
|--------|-----|---------|-----|--------------|----------|
| **CoreML** | ✅ Native | ❌ | ❌ | FP16, INT8, W8A16 | iOS apps |
| **Core AI** | ✅ iOS 27+ | ❌ | ❌ | INT8 | Future iOS |
| **LiteRT** | ✅ | ✅ Native | ✅ LiteRT.js | INT8, W8A16, W8A32 | Universal mobile |
| **NCNN** | ✅ | ✅ Native | ❌ | FP16 | ARM devices |
| **ExecuTorch** | ✅ | ✅ | ❌ | FP32 | PyTorch ecosystem |
| **ONNX** | Via inter. | Via inter. | ✅ | FP16, INT8 | Interchange |
| **MNN** | ✅ | ✅ | ❌ | FP16, INT8 | Alibaba devices |
| **RKNN** | ❌ | ✅ | ❌ | FP16, INT8 | Rockchip NPU |
| **QNN** | ❌ | ✅ | ❌ | W8A16 | Snapdragon NPU |
| **Edge TPU** | ❌ | ✅ | ❌ | INT8 auto | Google Coral |
| **TensorRT** | ❌ | ✅ | ❌ | FP16, INT8 | NVIDIA Jetson |

### Quantization Options

| Value | Type | Size Reduction | Calibration Needed |
|-------|------|----------------|-------------------|
| `quantize=8` | INT8 | ~75% | Yes (static) |
| `quantize=16` | FP16 | ~50% | No |
| `quantize="w8a16"` | INT8 weights + FP16 activations | ~70% | No |
| `quantize="w8a32"` | Dynamic INT8 | ~70% | No |

---

## 📊 Benchmarks

### iPhone 17 Pro (A19 Pro) — CoreML

| Task | CPU (ms) | CPU+ANE (ms) | FPS (ANE) |
|------|----------|--------------|-----------|
| Detect | 9.2 | **3.2** | **312** |
| Segment | 12.6 | **4.8** | **208** |
| Semantic | 9.7 | **4.6** | **217** |
| Depth | 25.0 | **5.3** | **189** |
| Classify | 2.2 | **1.9** | **526** |
| Pose | 11.9 | **3.9** | **256** |
| OBB | 10.6 | **3.4** | **294** |

### Xiaomi 17 (Snapdragon 8 Elite Gen 5) — LiteRT w8a32

| Task | CPU (ms) | GPU (ms) | FPS (GPU) |
|------|----------|----------|-----------|
| Detect | 52.2 | **15.8** | **63** |
| Segment | 73.4 | **33.2** | **30** |
| Depth | 124.4 | **23.0** | **43** |
| Classify | 4.4 | **3.1** | **323** |
| Pose | 57.4 | **16.6** | **60** |
| OBB | 50.3 | **11.7** | **85** |

### Edge Devices

| Device | Format | YOLO26n FPS | YOLO26s FPS |
|--------|--------|-------------|-------------|
| Raspberry Pi 5 | NCNN | ~8 | ~4 |
| Jetson Orin Nano | TensorRT FP16 | ~45 | ~25 |
| Jetson AGX Orin | TensorRT FP16 | ~120 | ~70 |
| Coral USB Accelerator | Edge TPU | ~30 | ~15 |

---

## 💡 Use Cases

| Use Case | Task | Platform | Reference |
|----------|------|----------|-----------|
| 🏋️ **Fitness Coach** | Pose | Flutter | [Guide](skills/yolo-mobile-integration/references/use-cases/fitness-coach.md) |
| 🔒 **Security Alarm** | Detection | iOS/Android | [Guide](skills/yolo-mobile-integration/references/use-cases/security-alarm.md) |
| 🔢 **Object Counting** | Detection | React Native | [Guide](skills/yolo-mobile-integration/references/use-cases/object-counting.md) |
| 🅿️ **Parking Management** | Detection | Flutter | [Guide](skills/yolo-mobile-integration/references/use-cases/parking-management.md) |
| 📏 **Distance Measurement** | Depth | iOS/Android | [Guide](skills/yolo-mobile-integration/references/use-cases/distance-measurement.md) |
| 🔥 **Heatmap Analytics** | Detection | Web | [Guide](skills/yolo-mobile-integration/references/use-cases/heatmap-analytics.md) |
| 🎭 **Object Blurring** | Segmentation | Flutter | [Guide](skills/yolo-mobile-integration/references/use-cases/object-blurring.md) |

---

## 📱 Platform Support

### Framework Integration

| Framework | iOS | Android | Documentation |
|-----------|-----|---------|---------------|
| **Flutter** | ✅ CoreML | ✅ LiteRT | [Guide](skills/yolo-mobile-integration/references/frameworks/flutter-integration.md) |
| **React Native** | ✅ CoreML | ✅ LiteRT | [Guide](skills/yolo-mobile-integration/references/frameworks/react-native-integration.md) |
| **Native Swift** | ✅ CoreML | — | [Guide](skills/yolo-mobile-integration/references/platforms/ios-coreml.md) |
| **Native Kotlin** | — | ✅ LiteRT | [Guide](skills/yolo-mobile-integration/references/platforms/android-litert.md) |
| **Web/PWA** | ✅ LiteRT.js | ✅ LiteRT.js | [Guide](skills/yolo-mobile-integration/references/platforms/web-tfjs.md) |

### Hardware Deployment

| Hardware | Format | Performance | Guide |
|----------|--------|-------------|-------|
| **Raspberry Pi 5** | NCNN | ~8 FPS (YOLO26n) | [Guide](skills/yolo-mobile-integration/references/hardware/raspberry-pi.md) |
| **NVIDIA Jetson Orin** | TensorRT | 30-60 FPS (YOLO26n) | [Guide](skills/yolo-mobile-integration/references/hardware/nvidia-jetson.md) |
| **Google Coral** | Edge TPU | ~30 FPS (YOLO26n) | [Guide](skills/yolo-mobile-integration/references/hardware/edge-tpu.md) |

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- For iOS export: macOS with Xcode
- For Android: Android SDK

### Install Ultralytics

```bash
# Using pip
pip install ultralytics

# Using conda
conda install -c conda-forge ultralytics

# From source
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

### Verify Installation

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Export for mobile
model.export(format="coreml", quantize=8)
```

---

## 📖 Documentation

### Skill Structure

```
skills/yolo-mobile-integration/
├── SKILL.md                          # Main orchestrator
├── references/
│   ├── yolo26-model.md               # Model selection guide
│   ├── export-pipeline.md            # Export instructions
│   ├── frameworks/
│   │   ├── flutter-integration.md    # Flutter integration
│   │   └── react-native-integration.md
│   ├── platforms/
│   │   ├── ios-coreml.md             # iOS CoreML
│   │   ├── android-litert.md         # Android LiteRT
│   │   ├── android-ncnn.md           # Android NCNN
│   │   ├── cross-platform-executorch.md
│   │   └── web-tfjs.md               # Web LiteRT.js
│   ├── tasks/
│   │   ├── detection.md
│   │   ├── segmentation.md
│   │   ├── classification.md
│   │   ├── pose-estimation.md
│   │   ├── obb.md
│   │   ├── semantic.md
│   │   └── depth.md
│   ├── use-cases/
│   │   ├── fitness-coach.md
│   │   ├── security-alarm.md
│   │   ├── object-counting.md
│   │   ├── parking-management.md
│   │   ├── distance-measurement.md
│   │   ├── heatmap-analytics.md
│   │   └── object-blurring.md
│   └── hardware/
│       ├── raspberry-pi.md
│       ├── nvidia-jetson.md
│       └── edge-tpu.md
├── scripts/
│   ├── export_model.py               # Export helper
│   ├── create_project.py             # Project scaffolder
│   └── benchmark_mobile.py           # Benchmark tool
└── evals/
    └── evals.json                    # Test cases
```

### Key References

| Document | Description |
|----------|-------------|
| [YOLO26 Model Reference](skills/yolo-mobile-integration/references/yolo26-model.md) | Complete model guide |
| [Export Pipeline](skills/yolo-mobile-integration/references/export-pipeline.md) | All export formats |
| [Flutter Integration](skills/yolo-mobile-integration/references/frameworks/flutter-integration.md) | Complete Flutter guide |
| [React Native Integration](skills/yolo-mobile-integration/references/frameworks/react-native-integration.md) | Complete RN guide |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

---

## 📄 License

This project is licensed under the [AGPL-3.0 License](LICENSE) - see the [LICENSE](LICENSE) file for details.

### Ultralytics License

YOLO26 models are licensed under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) or [Enterprise License](https://www.ultralytics.com/license).

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics) for the amazing YOLO26 models
- [Apple](https://developer.apple.com/machine-learning/) for CoreML
- [Google](https://www.tensorflow.org/lite) for LiteRT
- [Tencent](https://github.com/Tencent/ncnn) for NCNN
- [PyTorch](https://pytorch.org/) for ExecuTorch

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=ultralytics/ultralytics&type=Date)](https://star-history.com/#ultralytics/ultralytics&Date)

**Made with ❤️ by the YOLO26 Mobile Integration Team**

[⬆ Back to Top](#-yolo26-mobile-integration)

</div>
