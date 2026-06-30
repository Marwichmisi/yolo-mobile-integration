# Raspberry Pi — Déploiement Ultralytics YOLO26

## Vue d'ensemble

Raspberry Pi est un ordinateur compact et abordable capable d'exécuter Ultralytics YOLO26 pour la détection d'objets en temps réel en périphérie (edge) — sans GPU nécessaire. Ce guide couvre le déploiement sur Raspberry Pi 4 et 5 : installation, exportation, inférence avec caméra, benchmarks et scripts prêts à l'emploi.

## Comparaison de la série Raspberry Pi

| Caractéristique | Raspberry Pi 3 | Raspberry Pi 4 | Raspberry Pi 5 |
|---|---|---|---|
| CPU | BCM2837, Cortex-A53 64 bits | BCM2711, Cortex-A72 64 bits | BCM2712, Cortex-A76 64 bits |
| Fréquence max CPU | 1,4 GHz | 1,8 GHz | 2,4 GHz |
| GPU | VideoCore IV | VideoCore VI | VideoCore VII |
| Fréquence max GPU | 400 MHz | 500 MHz | 800 MHz |
| Mémoire | 1 Go LPDDR2 | 1/2/4/8 Go LPDDR4-3200 | 4/8 Go LPDDR4X-4267 |
| PCIe | N/A | N/A | Interface 1xPCIe 2.0 |
| Consommation max | 2,5 A@5V | 3 A@5V | 5 A@5V (PD activé) |

## Guide de déploiement étape par étape

### Étape 1 — Exportation du modèle

Sur une machine de développement (PC/Mac), exporter le modèle YOLO26n au format ONNX :

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

> Pour NCNN (meilleur sur ARM) : `model.export(format="ncnn")`

### Étape 2 — Transférer le modèle vers le RPi

```bash
scp yolo26n.onnx user@raspberrypi:~/yolo26n.onnx
# ou pour NCNN :
scp -r yolo26n_ncnn_model user@raspberrypi:~/yolo26n_ncnn_model
```

### Étape 3 — Installer les dépendances sur le RPi

#### Avec Docker (recommandé pour démarrer rapidement)

```bash
t=ultralytics/ultralytics:latest-arm64
sudo docker pull $t && sudo docker run -it --ipc=host $t
```

#### Sans Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-opencv libgl1 -y
pip install -U pip ultralytics[export]
sudo reboot
```

### Étape 4 — Tester l'inférence

```python
from ultralytics import YOLO

model = YOLO("yolo26n.onnx")
results = model("https://ultralytics.com/images/bus.jpg", imgsz=640)
results[0].show()
```

### Étape 5 — Configurer la caméra et lancer

Voir les sections dédiées ci-dessous.

---

## Utilisation de la caméra Raspberry Pi

### Test de la caméra

```bash
rpicam-hello
# ou sur les anciennes versions :
libcamera-hello
```

> Le Raspberry Pi 5 utilise des connecteurs CSI plus petits (15 broches vs 22 broches sur RPi 4). Un câble adaptateur 15→22 broches est nécessaire.

### Installation de picamera2

```bash
sudo apt install python3-picamera2 -y
```

### Initialisation et capture de frames avec picamera2

```python
from picamera2 import Picamera2
import time

picam2 = Picamera2()

# Configuration : résolution 1280x720, format RGB888
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)

picam2.start()
time.sleep(2)  # laisser le capteur s'initialiser

frame = picam2.capture_array()
print(f"Frame capturée : {frame.shape}")  # (720, 1280, 3)

picam2.stop()
```

### Configuration avancée avec buffers

```python
from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"NoiseReductionMode": 0}  # désactiver le bruit pour plus de FPS
)
picam2.configure(config)
picam2.start()

frame = picam2.capture_array()

picam2.stop()
```

### Inférence en temps réel avec picamera2

```python
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialiser Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Charger le modèle YOLO26
model = YOLO("yolo26n.onnx")

while True:
    frame = picam2.capture_array()
    results = model(frame, imgsz=640)
    annotated_frame = results[0].plot()

    # Convertir RGB → BGR pour OpenCV
    annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", annotated_bgr)

    if cv2.waitKey(1) == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
```

### Inférence via flux TCP

```bash
# Terminal 1 : démarrer le serveur vidéo
rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
```

```python
# Terminal 2 : exécuter l'inférence
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("tcp://127.0.0.1:8888")
```

---

## Script complet : compteur de véhicules sur RPi5

Ce script détecte les véhicules (voitures, motos, bus, camions) et compte les passages dans une zone définie.

```python
import cv2
import time
from collections import defaultdict
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolo26n.onnx"
CAMERA_RESOLUTION = (1280, 720)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Ligne de comptage (y en pixels) — ajuster selon la scène
COUNTING_LINE_Y = 450
OFFSET_TOLERANCE = 25  # pixels de tolérance autour de la ligne

# --- Initialisation caméra ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# --- Chargement modèle ---
model = YOLO(MODEL_PATH)
print("Modèle chargé. Détection en cours...")

# --- État du compteur ---
vehicle_count = 0
track_positions = defaultdict(lambda: {"prev_y": None, "counted": False})

def is_crossing_line(prev_y, curr_y):
    """Vérifie si l'objet traverse la ligne de comptage (haut vers bas)."""
    if prev_y is None:
        return False
    return prev_y < COUNTING_LINE_Y - OFFSET_TOLERANCE and curr_y >= COUNTING_LINE_Y

# --- Boucle principale ---
fps_counter = 0
fps_timer = time.time()

while True:
    frame = picam2.capture_array()
    results = model(frame, imgsz=640, verbose=False)

    detections = results[0].boxes

    for box in detections:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        track_id = f"{cls}_{int(cx)}_{int(cy)}"

        # Comptage
        prev = track_positions[track_id]["prev_y"]
        if is_crossing_line(prev, cy) and not track_positions[track_id]["counted"]:
            vehicle_count += 1
            track_positions[track_id]["counted"] = True
            print(f"[COMPTÉ] {VEHICLE_CLASSES[cls]} #{vehicle_count} (conf: {conf:.2f})")

        track_positions[track_id]["prev_y"] = cy

        # Dessin
        label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Ligne de comptage
    cv2.line(frame, (0, COUNTING_LINE_Y), (CAMERA_RESOLUTION[0], COUNTING_LINE_Y),
             (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicules: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # FPS
    fps_counter += 1
    if time.time() - fps_timer >= 2.0:
        fps = fps_counter / (time.time() - fps_timer)
        print(f"FPS: {fps:.1f} | Véhicules comptés: {vehicle_count}")
        fps_counter = 0
        fps_timer = time.time()

    # Affichage
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Compteur de Vehicules", display)

    if cv2.waitKey(1) == ord("q"):
        break

print(f"\nTotal véhicules comptés : {vehicle_count}")
picam2.stop()
cv2.destroyAllWindows()
```

### Lancement

```bash
python3 vehicle_counter.py
```

---

## Améliorations YOLO26 vs YOLO11

YOLO26n sur Raspberry Pi 5 (ONNX, 640px) :
- **+15 % images/seconde** : 6,79 → 7,79 FPS
- **mAP plus élevé** : 40,1 vs 39,5
- **Latence** : ~128 ms par frame (ONNX, FP32)

## Benchmarks détaillés sur Raspberry Pi 5

Testés avec Ultralytics 8.4.1 en FP32, taille d'entrée 640.

| Modèle | Format | FPS | Latence (ms) | mAP | Recommandé ? |
|---|---|---|---|---|---|
| **YOLO26n** | ONNX | **7,79** | **~128** | 40,1 | Oui |
| YOLO26n | NCNN | ~8,5 | ~118 | 40,1 | Oui (meilleur choix) |
| **YOLO26s** | ONNX | ~4,2 | ~238 | 46,8 | Oui (plus précis) |
| YOLO26s | NCNN | ~4,8 | ~208 | 46,8 | Oui |
| YOLO26m | ONNX | ~1,8 | ~555 | 50,2 | Non (trop lent) |
| YOLO26l | ONNX | ~0,9 | ~1111 | 52,8 | Non |

> Seuls **YOLO26n** et **YOLO26s** sont recommandés — les autres tailles sont trop volumineuses pour un usage temps réel sur RPi.

### Reproduire les benchmarks

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.benchmark(data="coco128.yaml", imgsz=640)
```

---

## Overclocking Raspberry Pi 5

L'overclocking peut augmenter les FPS de 20 à 30 % sur RPi 5.

### Étape 1 — Mettre à jour le firmware

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-update
```

### Étape 2 — Vérifier la température avant overclocking

```bash
vcgencmd measure_temp
```

> Installer un radiateur + ventilateur actif est **obligatoire** avant d'overclocker.

### Étape 3 — Éditer la configuration

```bash
sudo nano /boot/firmware/config.txt
```

Ajouter à la fin :

```ini
# Overclocking RPi 5
arm_freq=2800
gpu_freq=900
over_voltage_delta=50000
force_turbo=0
```

> **Valeurs conservatrices** : arm_freq=2800, gpu_freq=900.  
> **Valeurs agressives** (refroidissement liquide requis) : arm_freq=3000, gpu_freq=1000, force_turbo=1.

### Étape 4 — Redémarrer et vérifier

```bash
sudo reboot
# Après redémarrage :
vcgencmd measure_clock arm
vcgencmd measure_temp
```

### Étape 5 — Stabilité test

```bash
sudo apt install stress-ng -y
stress-ng --cpu 4 --timeout 300s --metrics-brief
```

> Si le système plante, réduire arm_freq par incréments de 100 MHz.

---

## Setup NVMe SSD pour fonctionnement continu

Pour une utilisation 24h/7j, les cartes SD s'usent vite par les écritures. Un SSD NVMe est fortement recommandé.

### Matériel nécessaire

- Raspberry Pi 5
- Adaptateur HAT NVMe (ex: Pimoroni NVMe Base, Waveshare NVMe HAT)
- SSD NVMe M.2 2230 ou 2242

### Installation

```bash
# 1. Installer le SSD sur le HAT NVMe et le monter sur le RPi 5

# 2. Flasher Raspberry Pi OS Lite sur le SSD (pas sur la SD)
#    Utiliser le tool Raspberry Pi Imager → sélectionner le SSD comme cible

# 3. Modifier le bootloader pour booter sur NVMe
sudo rpi-eeprom-config --edit
# Modifier BOOT_ORDER pour mettre le NVMe en premier (ex: BOOT_ORDER=0x641)
# 6 = NVMe, 4 = USB, 1 = SD

# 4. Redémarrer — le RPi boot maintenant sur le SSD
sudo reboot

# 5. Vérifier
lsblk
# Le SSD devrait apparaître comme /dev/nvme0n1
```

### Réduire les écritures sur le SSD

```bash
# Monter en mode relâché (réduit les écritures)
sudo nano /etc/fstab
# Ajouter : defaults,noatime,nodiratime

# Désactiver le swap (si 8 Go de RAM suffisent)
sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo systemctl disable dphys-swapfile
```

### Optimisation pour la caméra + NVMe

```bash
# Réserver de la mémoire GPU minimale (RPi 5 gère automatiquement)
# Pas besoin de modifier gpu_mem sur RPi 5

# Augmenter les handles USB si des périphériques USB sont connectés
sudo nano /boot/firmware/cmdline.txt
# Ajouter : usb-storage.quirks=152d:0578:u
```

---

## Meilleures pratiques

1. **Raspberry Pi OS Lite** — Ne pas installer l'environnement de bureau pour économiser de la RAM et du CPU.
2. **Refroidissement actif** — Utiliser un ventilateur ou un boîtier avec radiateur, surtout si overclocking.
3. **Alimentation** — Utiliser un adaptateur USB-C officiel (5V/5A) pour le RPi 5.
4. **Swap désactivé** — Avec 4+ Go de RAM, désactiver le swap pour éviter les écritures sur le SSD.
5. **Mode headless** — Configurer SSH et le Wi-Fi avant le premier boot pour éviter de brancher un clavier/écran.

```bash
# Créer les fichiers pour le setup headless (avant premier boot)
touch /boot/ssh
nano /boot/wpa_supplicant.conf
```

```ini
country=FR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="VOTRE_SSID"
    psk="VOTRE_MOT_DE_PASSE"
}
```

---

## Prochaines étapes

- [Mode Predict](https://docs.ultralytics.com/fr/modes/predict) — options d'inférence avancées
- [Mode Export](https://docs.ultralytics.com/fr/modes/export) — formats supplémentaires (NCNN, TFLite, CoreML)
- [Guide NVIDIA Jetson](https://docs.ultralytics.com/fr/guides/nvidia-jetson) — pour plus de puissance en périphérie
- [Optimisation ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/arm-execution-provider.html) — execution provider ARM pour performances accrues
