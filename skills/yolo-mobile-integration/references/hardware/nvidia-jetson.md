# NVIDIA Jetson - Déploiement Ultralytics YOLO26

## Vue d'ensemble

NVIDIA Jetson est une plateforme de calcul en périphérie (edge computing) avec GPU NVIDIA intégré, optimisée pour l'inférence d'IA en temps réel. TensorRT permet d'optimiser les modèles YOLO26 pour des performances maximales sur le GPU Jetson. Ce guide couvre le déploiement sur Jetson Orin, Nano et Xavier avec caméra GStreamer et benchmarks de performance.

## Comparaison de la série Jetson

| Caractéristique | Jetson Orin Nano | Jetson Orin NX | Jetson AGX Orin | Jetson Nano | Jetson Xavier NX |
|---|---|---|---|---|---|
| GPU | 1024 cœurs Ampere | 1024 cœurs Ampere | 2048 cœurs Ampere | 128 cœurs Maxwell | 384 cœurs Volta |
| Performance IA (TOPS) | 40 | 100 | 275 | 0,5 | 21 |
| CPU | 6× Cortex-A78AE | 8× Cortex-A78AE | 12× Cortex-A78AE | 4× Cortex-A57 | 6× Carmel ARMv8.2 |
| RAM | 4/8 Go LPDDR5 | 8/16 Go LPDDR5 | 32/64 Go LPDDR5 | 4 Go LPDDR4 | 8/16 Go LPDDR4x |
| Consommation | 7-15 W | 10-25 W | 15-60 W | 5-10 W | 10-20 W |
| Format | Module Orin Nano | Module Orin NX | Module AGX | Module Nano | Module Xavier NX |

## Guide de déploiement étape par étape

### Étape 1 - Installer JetPack SDK

Le SDK JetPack inclut L4T (Linux for Tegra), CUDA, cuDNN, TensorRT, OpenCV et GStreamer. Il est préinstallé sur la carte SD officielle NVIDIA.

```bash
# Vérifier la version JetPack
cat /etc/nv_tegra_release
# Vérifier L4T
head -n 1 /etc/nv_tegra_release

# Vérifier TensorRT
dpkg -l | grep TensorRT
# Vérifier CUDA
nvcc --version

# Vérifier OpenCV
python3 -c "import cv2; print(cv2.__version__)"
```

Si vous avez besoin de flasher un nouveau système :

```bash
1. Télécharger NVIDIA SDK Manager sur un PC Linux
2. Flasher la carte SD avec JetPack 6.x (L4T 36.x)
3. L'installation inclut : Ubuntu 22.04, CUDA 12.x, TensorRT 10.x, cuDNN 9.x
```

### Étape 2 - Configurer les modes alimentation et performance

```bash
# Vérifier le mode actuel
sudo nvpmodel -q

# Mode performance maximale (tous les cœurs)
sudo nvpmodel -m 0
sudo jetson_clocks

# Mode économie d'énergie (Orin Nano)
sudo nvpmodel -m 1

# Verrouiller les fréquences CPU/GPU
sudo jetson_clocks --fan
```

### Étape 3 - Installer Python et Ultralytics

```bash
# Créer un environnement virtuel (recommandé)
sudo apt install python3-venv python3-pip -y
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

# Installer PyTorch pour Jetson (version ARM64)
pip install --upgrade pip
pip install numpy

# PyTorch adapté Jetson (voir https://forums.developer.nvidia.com/t/pytorch-for-jetson)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Installer Ultralytics avec support TensorRT
pip install ultralytics[export]

# Vérifier l'installation
python3 -c "from ultralytics import YOLO; print('OK')"
```

### Étape 4 - Exporter le modèle vers TensorRT

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export TensorRT FP16 (recommandé pour Jetson)
model.export(format="engine", half=True, imgsz=640, batch=1)

# Export TensorRT INT8 (nécessite calibration)
model.export(format="engine", int8=True, imgsz=640, data="coco128.yaml")

# Export TensorRT FP32 (précision maximale)
model.export(format="engine", imgsz=640)
```

```bash
# CLI - Export TensorRT FP16
yolo export model=yolo26n.pt format=engine half=True imgsz=640

# CLI - Export TensorRT INT8 avec calibration
yolo export model=yolo26n.pt format=engine int8=True data=coco128.yaml imgsz=640
```

### Étape 5 - Tester l'inférence

```python
from ultralytics import YOLO
import time

model = YOLO("yolo26n.engine")

# Test sur image
results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()

# Benchmark sur 100 images
times = []
for _ in range(100):
    start = time.time()
    model("test_image.jpg", verbose=False)
    times.append(time.time() - start)

avg_latency = sum(times) / len(times) * 1000
fps = 1000 / avg_latency
print(f"Latence moyenne : {avg_latency:.1f} ms")
print(f"FPS : {fps:.1f}")
```

---

## Intégration caméra GStreamer

### Tester la caméra CSI

```bash
# Caméra CSI ( ribbon )
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! 'video/x-raw,format=BGRx' ! \
  videoconvert ! 'video/x-raw,format=BGR' ! \
  fpsdisplaysink video-sink=fakesink sync=false -v

# Vérifier les caméras disponibles
gst-device-monitor-1.0
```

### Inférence temps réel avec GStreamer + OpenCV

```python
import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# Pipeline GStreamer pour caméra CSI
def create_gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Charger le modèle
model = YOLO("yolo26n.engine")

# Ouvrir le flux caméra
pipeline = create_gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERREUR: Impossible d'ouvrir la caméra")
    exit()

print("Caméra ouverte. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inférence YOLO26
    results = model(frame, imgsz=640, verbose=False)

    # Annoter le frame
    annotated = results[0].plot()

    # Afficher
    cv2.imshow("YOLO26 Jetson", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Pipeline GStreamer pour caméra USB

```python
# Pour caméra USB (UVC)
usb_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(usb_pipeline, cv2.CAP_GSTREAMER)
```

### Pipeline GStreamer pour flux RTSP/IP

```python
# Pour caméra IP RTSP
rtsp_pipeline = (
    "rtspsrc location=rtsp://192.168.1.100:554/stream ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(rtsp_pipeline, cv2.CAP_GSTREAMER)
```

---

## Script complet : détection temps réel avec comptage

```python
import cv2
import time
import subprocess
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolo26n.engine"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
VEHICLE_CLASSES = {2: "voiture", 3: "moto", 5: "bus", 7: "camion"}

# --- Pipeline GStreamer ---
def create_gstreamer_pipeline(width=1280, height=720, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1 ! "
        f"nvvidconv ! video/x-raw,width={INPUT_SIZE},height={INPUT_SIZE},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )

# --- Initialisation ---
model = YOLO(MODEL_PATH)
pipeline = create_gstreamer_pipeline()
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la caméra GStreamer")

# --- Ligne de comptage ---
COUNTING_LINE_Y = 400
OFFSET = 20

vehicle_count = 0
track_history = defaultdict(lambda: {"prev_y": None, "counted": False})

def is_crossing(prev_y, curr_y):
    if prev_y is None:
        return False
    return prev_y < COUNTING_LINE_Y - OFFSET and curr_y >= COUNTING_LINE_Y

# --- Boucle principale ---
fps_counter = 0
fps_timer = time.time()

print("Détection en cours. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, imgsz=INPUT_SIZE, conf=CONF_THRESHOLD,
                     iou=IOU_THRESHOLD, verbose=False)
    detections = results[0].boxes

    for box in detections:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cy = (y1 + y2) / 2

        track_id = f"{cls}_{int((x1+x2)/2)}_{int(cy)}"

        if is_crossing(track_history[track_id]["prev_y"], cy) and not track_history[track_id]["counted"]:
            vehicle_count += 1
            track_history[track_id]["counted"] = True
            print(f"[COMPTÉ] {VEHICLE_CLASSES[cls]} #{vehicle_count} (conf: {conf:.2f})")

        track_history[track_id]["prev_y"] = cy

        # Dessin
        label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Ligne de comptage
    cv2.line(frame, (0, COUNTING_LINE_Y), (INPUT_SIZE, COUNTING_LINE_Y), (0, 0, 255), 2)

    # FPS
    fps_counter += 1
    if time.time() - fps_timer >= 2.0:
        fps = fps_counter / (time.time() - fps_timer)
        fps_counter = 0
        fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Véhicules: {vehicle_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO26 Jetson", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Lancement

```bash
# Activer les performances maximales
sudo nvpmodel -m 0
sudo jetson_clocks

# Lancer le script
python3 vehicle_counter.py
```

---

## Benchmarks YOLO26 sur Jetson

### Résultats attendus (TensorRT FP16, 640px)

| Modèle | Jetson Orin Nano | Jetson Orin NX | Jetson AGX Orin | Jetson Nano | Jetson Xavier NX |
|---|---|---|---|---|---|
| YOLO26n | ~45 FPS | ~80 FPS | ~150 FPS | ~10 FPS | ~25 FPS |
| YOLO26s | ~25 FPS | ~45 FPS | ~80 FPS | ~5 FPS | ~15 FPS |
| YOLO26m | ~15 FPS | ~28 FPS | ~50 FPS | ~2 FPS | ~8 FPS |
| YOLO26l | ~10 FPS | ~18 FPS | ~32 FPS | ~1 FPS | ~5 FPS |

### Reproduire les benchmarks

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.benchmark(
    data="coco128.yaml",
    imgsz=640,
    half=True,       # FP16
    device=0,        # GPU
    batch=1,
    verbose=True,
)
```

### Latence typique (ms) - TensorRT FP16 640px

| Modèle | Orin Nano | Orin NX | AGX Orin | Nano | Xavier NX |
|---|---|---|---|---|---|
| YOLO26n | ~22 | ~12 | ~7 | ~100 | ~40 |
| YOLO26s | ~40 | ~22 | ~12 | ~200 | ~67 |
| YOLO26m | ~67 | ~36 | ~20 | ~500 | ~125 |
| YOLO26l | ~100 | ~55 | ~31 | ~1000 | ~200 |

---

## Modes alimentation et thermique

### Modes nvpmodel (Orin Nano)

```bash
# Vérifier les modes disponibles
sudo nvpmodel -q --verbose

# Mode 0 : MAXN - tous les cœurs, fréquence maximale
sudo nvpmodel -m 0

# Mode 1 : 5W - économie d'énergie réduite
sudo nvpmodel -m 1

# Mode 2 : économie d'énergie maximale
sudo nvpmodel -m 2
```

| Mode | CPU | GPU | Consommation | Usage |
|---|---|---|---|---|
| 0 (MAXN) | 6 cœurs @ 1,5 GHz | 1 GHz | 15 W | Inférence temps réel |
| 1 | 4 cœurs @ 1,0 GHz | 600 MHz | 10 W | Usage général |
| 2 | 2 cères @ 600 MHz | 300 MHz | 5 W | Veille, surveillance passive |

### Gestion thermique

```bash
# Vérifier la température
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Vérifier le ventilateur (si applicable)
cat /sys/devices/pwm-fan/cur_pwm

# Définir la vitesse du ventilateur (0-255)
echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm

# Vérifier les fréquences actuelles
sudo jetson_clocks --show
```

### Refroidissement recommandé

| Carte | Refroidissement passif | Refroidissement actif | Température seuil |
|---|---|---|---|
| Orin Nano | Radiateur obligatoire | Recommandé | 85 °C (throttle à 95 °C) |
| Orin NX | Radiateur obligatoire | Ventilateur 40mm | 85 °C |
| AGX Orin | Radiateur recommandé | Ventilateur boîtier | 90 °C |
| Nano | Radiateur recommandé | Non (passif uniquement) | 80 °C |
| Xavier NX | Radiateur obligatoire | Ventilateur recommandé | 85 °C |

---

## Optimisation TensorRT avancée

### Création manuelle d'un moteur TensorRT

```python
from ultralytics import YOLO
import tensorrt as trt

model = YOLO("yolo26n.pt")

# Export ONNX d'abord
model.export(format="onnx", imgsz=640, simplify=True, opset=12)

# Puis conversion TensorRT via trtexec (CLI)
```

```bash
# Optimisation manuelle avec trtexec
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo26n.onnx \
  --saveEngine=yolo26n_fp16.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --workspace=4096 \
  --verbose
```

### Benchmark TensorRT avec Python

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    return builder.build_serialized_network(network, config)

# Sauvegarder le moteur
engine_bytes = build_engine("yolo26n.onnx", "yolo26n_fp16.engine", fp16=True)
with open("yolo26n_fp16.engine", "wb") as f:
    f.write(engine_bytes)
```

### Inférence native TensorRT avec pycuda

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allouer la mémoire GPU
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "name": name})

    def infer(self, image):
        # Prétraitement
        input_blob = self.preprocess(image)

        # Copier vers GPU
        np.copyto(self.inputs[0]["host"], input_blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Inférence
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Récupérer les résultats
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [out["host"].copy() for out in self.outputs]

    def preprocess(self, image, imgsz=640):
        # Redimensionner et normaliser
        img = cv2.resize(image, (imgsz, imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None]  # batch dimension

# Utilisation
trt_model = TensorRTInference("yolo26n_fp16.engine")
outputs = trt_model.infer(frame)
```

---

## Export avec quantification INT8

L'INT8 nécessite un jeu de données de calibration.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export INT8 avec calibration COCO128
model.export(
    format="engine",
    int8=True,
    imgsz=640,
    data="coco128.yaml",
    batch=1,
    workspace=4,  # GB
    verbose=True,
)
```

### Jeu de données de calibration personnalisé

```python
# Préparer un dossier d'images de calibration (100-500 images suffisent)
# data/
#   calibration/
#     0001.jpg
#     0002.jpg
#     ...

model.export(
    format="engine",
    int8=True,
    imgsz=640,
    data="calibration/",  # dossier d'images
    batch=1,
)
```

---

## Déploiement avec Docker

```dockerfile
# Dockerfile pour Jetson avec Ultralytics + TensorRT
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

RUN pip3 install --no-cache-dir \
    numpy opencv-python \
    ultralytics[export] \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace
COPY yolo26n.engine /workspace/
COPY vehicle_counter.py /workspace/

CMD ["python3", "vehicle_counter.py"]
```

```bash
# Construire et exécuter
docker build -t yolo26-jetson .
docker run --runtime nvidia --rm -it \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    yolo26-jetson
```

---

## Meilleures pratiques

1. **Toujours utiliser TensorRT** - jamais de modèle `.pt` brut sur Jetson. L'engine TensorRT offre 5-10x plus de FPS.
2. **FP16 par défaut** - Le mode FP16 offre le meilleur rapport précision/performance. Réservez l'INT8 aux cas où la latence est critique.
3. **Résolution d'entrée** - 640px est optimal. Réduisez à 320px pour plus de FPS, augmentez à 1280px pour la précision.
4. **Batch = 1** - Toujours pour l'inférence en temps réel sur Jetson.
5. **Mode MAXN** - Toujours activer `sudo nvpmodel -m 0` avant l'inférence.
6. **GStreamer hardware** - Utiliser `nvarguscamerasrc` et `nvvidconv` pour le déchargement matériel.
7. **Monitoring** - Surveiller `tegrastats` et la température pendant les tests.

```bash
# Monitoring en temps réel
sudo tegrastats --interval 1000
```

---

## Dépannage

| Problème | Cause | Solution |
|---|---|---|
| `nvpmodel: command not found` | JetPack non installé | Flasher avec le SDK Manager NVIDIA |
| `No module named 'tensorrt'` | TensorRT non dans le PATH | `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH` ou installer via `apt install tensorrt` |
| CUDA out of memory | Modèle trop grand / workspace insuffisant | Réduire `imgsz`, utiliser `half=True`, augmenter `workspace` |
| `nvarguscamerasrc: no camera available` | Caméra CSI non détectée | Vérifier la connexion FPC, exécuter `gst-device-monitor-1.0` |
| Faible FPS avec GStreamer | Pas de déchargement matériel | Vérifier que `nvvidconv` est utilisé, pas `videoconvert` |
| `ModuleNotFoundError: 'ultralytics'` | environnement virtuel non activé | `source ~/yolo_env/bin/activate` |
| Throttling thermique | Surchauffe | Ajouter refroidissement actif, vérifier `cat /sys/devices/virtual/thermal/thermal_zone*/temp` |
| INT8 calibration error | Pas assez d'images de calibration | Utiliser 100+ images représentatives |
| `onnxsim` error during export | ONNX non simplifié | Ajouter `simplify=True` à l'export |

---

## Prochaines étapes

- [Guide NVIDIA Jetson Ultralytics](https://docs.ultralytics.com/fr/guides/nvidia-jetson) - guide officiel
- [Documentation TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/) - API complète
- [GStreamer plugins](https://docs.nvidia.com/metropolis/deepstream/dev-guide/) - pipelines avancés
- [Jetson Benchmark](https://github.com/NVIDIA-AI-IOT/jetson_benchmark) - outils de benchmarking
- [Mode Predict](https://docs.ultralytics.com/fr/modes/predict) - options d'inférence avancées
- [Mode Export](https://docs.ultralytics.com/fr/modes/export) - tous les formats d'export
