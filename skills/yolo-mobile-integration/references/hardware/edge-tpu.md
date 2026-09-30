# Google Coral Edge TPU - Déploiement Ultralytics YOLO26

## Vue d'ensemble

Google Coral Edge TPU est un accélérateur d'IA en périphérie conçu pour l'inférence locale à haute vitesse et faible consommation. Le compilateur Edge TPU convertit les modèles TensorFlow Lite en format binaire optimisé pour l'architecture TPU. Ce guide couvre le déploiement avec Coral USB Accelerator, Raspberry Pi et les plateformes compatibles.

## Matériel Coral

### Comparaison des produits Coral

| Caractéristique | Coral USB Accelerator | Coral Dev Board | Coral SoM |
|---|---|---|---|
| Interfaçage | USB 3.0 | SoM intégré | Module (SoM) |
| Performance | 4 TOPS | 4 TOPS | 4 TOPS |
| Consommation | ~2 W | ~5 W (total) | ~2 W |
| Compatibilité | Tout PC/Mac/Linux | Board dédiée | Boards compatibles |
| Prix indicatif | ~60 € | ~130 € | ~30-50 € |
| Format | Dongle USB | Carte complète | Module |

### Edge TPU runtime - Spécifications

| Caractéristique | Détail |
|---|---|
| Précision supportée | INT8 (quantification post-entraînement) |
| TOPS | 4 TOPS |
| Consommation | ~2 W |
| Température de fonctionnement | 0-60 °C |
| Systèmes supportés | Linux, Windows, macOS |
| Interface | USB 3.0 (SuperSpeed) |

## Limitations Edge TPU

### Contraintes importantes

| Limitation | Détail | Impact |
|---|---|---|
| **INT8 uniquement** | Quantification post-entraînement obligatoire | Perte de précision possible (1-3% mAP) |
| **Pas de FP16/FP32** | Le matériel ne supporte que l'INT8 | Impossible d'utiliser des modèles non quantifiés |
| **Ops limitées** | Certaines opérations non supportées | Le modèle doit utiliser des ops compatibles TPU |
| **Taille d'entrée fixe** | La résolution doit être définie à la compilation | Pas de résolution dynamique |
| **Pas de NMS custom** | Le post-traitement peut nécessiter un support | Vérifier la compatibilité end-to-end |
| **Batch = 1** | Inférence image par image uniquement | Pas de traitement par lot |

### Opérations supportées

| Opération | Support | Remarque |
|---|---|---|
| Conv2D | ✅ | Supporté |
| DepthwiseConv2D | ✅ | Supporté |
| MaxPool2D | ✅ | Supporté |
| AveragePool2D | ✅ | Supporté |
| ReLU / ReLU6 | ✅ | Supporté |
| Softmax | ✅ | Supporté |
| Resize (bilinear) | ✅ | Supporté |
| Pad | ✅ | Supporté |
| Concat | ✅ | Supporté |
| Reshape | ✅ | Supporté |
| Transpose | ✅ | Supporté |
| LeakyReLU | ❌ | Non supporté |
| HardSwish | ❌ | Non supporté |
| Swish | ❌ | Non supporté |

## Guide de déploiement étape par étape

### Étape 1 - Installer le Coral USB Accelerator

#### Sur Raspberry Pi / Linux

```bash
# Ajouter le dépôt Coral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Ajouter la clé GPG
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo apt-key add -

# Mettre à jour et installer
sudo apt update
sudo apt install libedgetpu1-std -y

# Vérifier la détection
lsusb | grep -i google
# Devrait afficher : Google Inc. Coral

# Vérifier le runtime
dpkg -l | grep edgetpu
```

#### Vérification du périphérique USB

```bash
# Vérifier que le dongle est reconnu
ls -la /dev/bus/usb/
# ou
usb-devices | grep -i coral

# Vérifier les droits d'accès
sudo usermod -a -G plugdev $USER
# Se déconnecter et reconnecter pour appliquer
```

### Étape 2 - Installer le runtime Edge TPU

```bash
# Runtime standard (recommandé)
sudo apt install libedgetpu1-sty -y

# Runtime maximum ( performances maximales, consommation accrue)
sudo apt install libedgetpu1-max -y

# Différence :
# - std : 500 MHz (2 TOPS effectifs, ~1,5W)
# - max : 1 GHz (4 TOPS effectifs, ~2W)
```

#### Runtime Python (PyCoral)

```bash
# Installer pycoral (API Python)
sudo apt install python3-pycoral -y

# Ou via pip
pip install pycoral
```

#### Runtime Python alternatif (tflite-runtime)

```bash
# tflite-runtime avec delegate Edge TPU
pip install tflite-runtime

# Vérifier le delegate
python3 -c "
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
interpreter = Interpreter(
    model_path='yolo26n_edgetpu.tflite',
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
)
print('Edge TPU delegate chargé avec succès')
"
```

### Étape 3 - Installer Ultralytics

```bash
# Créer un environnement virtuel
python3 -m venv ~/coral_env
source ~/coral_env/bin/activate

# Installer Ultralytics
pip install --upgrade pip
pip install ultralytics[export]
pip install tflite-runtime

# Vérifier
python3 -c "from ultralytics import YOLO; print('OK')"
```

### Étape 4 - Exporter le modèle YOLO26 vers Edge TPU

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export Edge TPU (compile automatiquement)
model.export(format="edgetpu", imgsz=640)

# Le fichier généré : yolo26n_edgetpu.tflite
```

```bash
# CLI
yolo export model=yolo26n.pt format=edgetpu imgsz=640
```

#### Processus de compilation Edge TPU

L'export `format="edgetpu"` effectue automatiquement ces étapes :

1. Export ONNX depuis PyTorch
2. Conversion ONNX vers TensorFlow SavedModel
3. Conversion vers TensorFlow Lite avec quantification INT8
4. Compilation Edge TPU via `edgetpu_compiler`

```bash
# Vérifier le modèle compilé
edgetpu_compiler --version

# Compiler manuellement (si nécessaire)
edgetpu_compiler yolo26n_quant.tflite -o ./output/

# Le compilateur affiche des avertissements si des ops ne sont pas supportés
```

### Étape 5 - Tester l'inférence

```python
from ultralytics import YOLO

# Charger le modèle Edge TPU
model = YOLO("yolo26n_edgetpu.tflite")

# Inférence sur image
results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()

# Vérifier que le TPU est utilisé
# (affiche device: Edge TPU dans les logs)
```

---

## Intégration Raspberry Pi + Coral

### Installation sur Raspberry Pi

```bash
# Système : Raspberry Pi OS 64 bits (Bullseye ou supérieur)

# 1. Installer les dépendances
sudo apt update
sudo apt install -y python3-pip python3-venv libcamera-dev

# 2. Configurer le dépôt Coral (voir étape 1)

# 3. Installer le runtime
sudo apt install libedgetpu1-sty -y

# 4. Installer Ultralytics
python3 -m venv ~/coral_env
source ~/coral_env/bin/activate
pip install ultralytics[export] tflite-runtime

# 5. Exporter le modèle
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='edgetpu', imgsz=640)
"

# 6. Vérifier le périphérique
lsusb | grep -i google
```

### Caméra CSI + Coral (picamera2)

```python
import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialiser la caméra
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# Charger le modèle Edge TPU
model = YOLO("yolo26n_edgetpu.tflite")

# Boucle d'inférence
fps_counter = 0
fps_timer = time.time()

while True:
    frame = picam2.capture_array()
    results = model(frame, imgsz=640, verbose=False)

    annotated = results[0].plot()
    display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # FPS
    fps_counter += 1
    if time.time() - fps_timer >= 2.0:
        fps = fps_counter / (time.time() - fps_timer)
        print(f"FPS: {fps:.1f}")
        fps_counter = 0
        fps_timer = time.time()

    cv2.imshow("Coral + RPi", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
```

---

## Script complet : détection temps réel avec comptage

```python
import cv2
import time
import numpy as np
from collections import defaultdict
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolo26n_edgetpu.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
VEHICLE_CLASSES = {2: "voiture", 3: "moto", 5: "bus", 7: "camion"}

# --- Initialisation caméra ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# --- Chargement modèle Edge TPU ---
model = YOLO(MODEL_PATH)
print("Modèle Edge TPU chargé. Détection en cours...")

# --- Ligne de comptage ---
COUNTING_LINE_Y = 450
OFFSET = 25

vehicle_count = 0
track_history = defaultdict(lambda: {"prev_y": None, "counted": False})

def is_crossing(prev_y, curr_y):
    if prev_y is None:
        return False
    return prev_y < COUNTING_LINE_Y - OFFSET and curr_y >= COUNTING_LINE_Y

# --- Boucle principale ---
fps_counter = 0
fps_timer = time.time()

while True:
    frame = picam2.capture_array()
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
    cv2.line(frame, (0, COUNTING_LINE_Y), (frame.shape[1], COUNTING_LINE_Y),
             (0, 0, 255), 2)

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

    # Affichage
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Edge TPU Détection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nTotal véhicules comptés : {vehicle_count}")
picam2.stop()
cv2.destroyAllWindows()
```

### Lancement

```bash
# Sur Raspberry Pi
python3 vehicle_counter_coral.py
```

---

## Benchmarks YOLO26 sur Edge TPU

### Résultats attendus (INT8, 640px, Coral USB Accelerator)

| Modèle | Latence (ms) | FPS | mAP (perte INT8) | Taille modèle |
|---|---|---|---|---|
| **YOLO26n** | **~15** | **~65** | 40,1 (-0,5) | ~3,5 Mo |
| YOLO26s | ~35 | ~28 | 48,6 (-1,2) | ~11 Mo |
| YOLO26m | ~80 | ~12 | 53,1 (-1,8) | ~20 Mo |
| YOLO26l | ~150 | ~6 | 55,0 (-2,1) | ~26 Mo |

### Benchmarks selon la plateforme hôte

| Plateforme hôte | YOLO26n FPS | YOLO26s FPS | Goulot d'étranglement |
|---|---|---|---|
| Raspberry Pi 4 | ~45 | ~20 | CPU (pré/post-traitement) |
| Raspberry Pi 5 | ~60 | ~28 | CPU (pré/post-traitement) |
| PC (i5) | ~65 | ~30 | Edge TPU (limite matérielle) |
| Coral Dev Board | ~50 | ~22 | CPU (pré/post-traitement) |
| Laptop (i7) | ~65 | ~30 | Edge TPU (limite matérielle) |

> **Note** : L'Edge TPU lui-même traite chaque image en ~15 ms (YOLO26n). Le goulot d'étranglement est souvent le CPU hôte pour le pré-traitement (redimensionnement, normalisation) et le post-traitement (NMS, dessin).

### Reproduire les benchmarks

```python
from ultralytics import YOLO
import time

model = YOLO("yolo26n_edgetpu.tflite")

# Warmup
for _ in range(10):
    model("test_image.jpg", imgsz=640, verbose=False)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    model("test_image.jpg", imgsz=640, verbose=False)
    times.append(time.time() - start)

avg_ms = sum(times) / len(times) * 1000
fps = 1000 / avg_ms
print(f"Latence moyenne : {avg_ms:.1f} ms")
print(f"FPS : {fps:.1f}")
```

---

## Optimisation des performances

### Réduire le goulot d'étranglement CPU

```python
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Prétraitement optimisé avec OpenCV
def preprocess_fast(image, imgsz=640):
    """Prétraitement optimisé pour Edge TPU."""
    # Redimensionner directement en BGR (évite la conversion)
    img = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    # Normalisation en place
    img = img.astype(np.float32) / 255.0
    # Ajouter dimension batch
    return np.expand_dims(img, axis=0)

# Pipeline parallèle : capture + prétraitement
class OptimizedPipeline:
    def __init__(self, model_path, imgsz=640):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.executor = ThreadPoolExecutor(max_workers=2)

    def process_frame(self, frame):
        # Prétraitement
        input_tensor = preprocess_fast(frame, self.imgsz)
        # Inférence
        results = self.model.predict(input_tensor, verbose=False)
        return results[0]
```

### Utiliser le runtime `std` vs `max`

```bash
# Mode standard (500 MHz) - recommandé pour la plupart des cas
sudo apt install libedgetpu1-sty -y

# Mode maximum (1 GHz) - pour les applications critiques en latence
sudo apt install libedgetpu1-max -y

# Basculement entre les modes
sudo apt remove libedgetpu1-max
sudo apt install libedgetpu1-sty
```

### Réduire la résolution d'entrée

```python
# Exporter avec une résolution inférieure pour plus de FPS
model = YOLO("yolo26n.pt")

# 320px - 2x plus rapide, précision réduite
model.export(format="edgetpu", imgsz=320)

# 416px - compromis vitesse/précision
model.export(format="edgetpu", imgsz=416)

# 512px - bon compromis
model.export(format="edgetpu", imgsz=512)
```

| Résolution | Latence YOLO26n | FPS | mAP |
|---|---|---|---|
| 320 | ~8 ms | ~120 | 35,2 |
| 416 | ~11 ms | ~90 | 38,1 |
| 512 | ~13 ms | ~75 | 39,5 |
| 640 | ~15 ms | ~65 | 40,1 |

---

## Déploiement avec Docker

```dockerfile
# Dockerfile pour Coral USB Accelerator
FROM python:3.11-slim

# Dépendances système
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
RUN pip install --no-cache-dir \
    ultralytics[export] \
    tflite-runtime \
    opencv-python-headless

WORKDIR /workspace
COPY yolo26n_edgetpu.tflite /workspace/
COPY vehicle_counter.py /workspace/

CMD ["python3", "vehicle_counter.py"]
```

```bash
# Construire et exécuter avec accès USB
docker build -t yolo26-coral .
docker run --rm -it \
    --device /dev/bus/usb:/dev/bus/usb \
    --privileged \
    yolo26-coral
```

---

## Déploiement sur Coral Dev Board

```bash
# Le Coral Dev Board a l'Edge TPU intégré (pas de dongle USB)

# 1. Vérifier le TPU intégré
lsusb | grep -i google

# 2. Installer le runtime (Mendel Linux)
sudo apt update
sudo apt install libedgetpu1-sty -y

# 3. Installer Python et Ultralytics
pip3 install ultralytics[export] tflite-runtime

# 4. Exporter et exécuter
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='edgetpu', imgsz=640)
"

python3 vehicle_counter.py
```

---

## Meilleures pratiques

1. **Toujours utiliser le runtime `std`** - Le mode `max` double la consommation pour un gain de ~10% en latence. Réservez-le aux cas critiques.
2. **Résolution 416 ou 512** - Le 640px est souvent inutilement précis. Le 416px offre un excellent compromis vitesse/précision.
3. **Prétraitement optimisé** - Utiliser `cv2.resize` directement en BGR pour éviter les conversions de couleur inutiles.
4. **Éviter les allocations mémoire** - Réutiliser les tableaux numpy entre les frames.
5. **Mode headless** - Ne pas afficher les frames avec `cv2.imshow` en production (goulot d'étranglement).
6. **Surveiller la température** - L'Edge TPU chauffe en fonctionnement continu. Prévoir un radiateur si nécessaire.
7. **USB 3.0 obligatoire** - Le dongle nécessite USB 3.0 pour des performances optimales. L'USB 2.0 réduit les FPS de ~40%.

---

## Dépannage

| Problème | Cause | Solution |
|---|---|---|
| `No Edge TPU device found` | Dongle non détecté | Vérifier `lsusb`, essayer un autre port USB 3.0 |
| `Failed to load delegate` | Runtime non installé | `sudo apt install libedgetpu1-sty` |
| `Insufficient permissions` | Droits USB insuffisants | `sudo usermod -a -G plugdev $USER` puis reconnexion |
| `Model not compiled for Edge TPU` | Modèle TFLite non compilé | Exécuter `edgetpu_compiler model.tflite` |
| `Ops not supported by Edge TPU` | Ops incompatibles dans le modèle | Vérifier les ops supportées, réduire la complexité |
| Faible FPS | CPU hôte lent | Réduire la résolution, optimiser le prétraitement |
| `USB transfer error` | Port USB 2.0 ou câble défectueux | Utiliser un port USB 3.0 (bleu), vérifier le câble |
| `ModuleNotFoundError: 'pycoral'` | pycoral non installé | `sudo apt install python3-pycoral` ou `pip install pycoral` |
| `edgetpu_compiler: command not found` | Compilateur non installé | `sudo apt install edgetpu-compiler` |
| Perte de précision importante | Quantification INT8 agressive | Vérifier le jeu de calibration, envisager le FP16 sur CPU |

---

## Prochaines étapes

- [Documentation Coral AI](https://coral.ai/docs/) - guides officiels
- [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/) - compilation et optimisation
- [PyCoral API](https://coral.ai/docs/reference/py/) - API Python avancée
- [Mode Predict](https://docs.ultralytics.com/fr/modes/predict) - options d'inférence avancées
- [Mode Export](https://docs.ultralytics.com/fr/modes/export) - tous les formats d'export
- [Guide Raspberry Pi](raspberry-pi.md) - déploiement sans Edge TPU
