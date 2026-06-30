# Déploiement Web avec LiteRT.js (ex-TF.js) — Guide Complet YOLO26

---

## QUAND UTILISER CE GUIDE

Utilisez ce guide quand vous devez :

- **Déployer YOLO26 dans un navigateur** (Chrome, Firefox, Edge, Safari)
- **Créer une PWA de détection d'objets** installable sur mobile/desktop
- **Utiliser la caméra en temps réel** pour la détection dans le navigateur
- **Faire tourner le modèle hors ligne** sans serveur backend
- **Intégrer YOLO dans un React Native WebView** (voir section dédiée)
- **Atteindre les meilleures performances** via WebGPU ou WASM

**Ne pas utiliser ce guide pour** : déploiement mobile natif (Android/iOS), côté serveur (Python/Node), ou React Native avec accès natif direct — consultez les autres fichiers de la section `platforms/`.

---

## Vue d'ensemble

**Important** : Depuis Ultralytics 8.4.83, le format `tfjs` est obsolète et remplacé par **LiteRT**. Le format `litert` exporte un modèle `.tflite` qui s'exécute dans le navigateur via [LiteRT.js](https://github.com/nicknisi/nicknisi/nicknisi/nicknisi) avec accélération WebGPU/WASM.

### Avantages principaux
- **Un seul modèle** : Même fichier `.tflite` pour mobile, embarqué et web
- **Exécution hors ligne** : Pas de connexion internet requise
- **Accélération GPU** : WebGPU et WASM pour des performances rapides
- **Confidentialité** : Les données restent sur l'appareil
- **Multi-plateforme** : Navigateur, Node.js, React Native

---

## Exportation du modèle

### Commandes d'exportation

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export LiteRT (remplace TF.js)
model.export(format="litert")  # Crée 'yolo26n.tflite'

# Export INT8 dynamique
model.export(format="litert", quantize="w8a32")  # Crée 'yolo26n_w8a32.tflite'

# Export INT8 statique (meilleure précision)
model.export(format="litert", quantize=8, data="coco8.yaml")  # Crée 'yolo26n_int8.tflite'

# Export W8A16
model.export(format="litert", quantize="w8a16", data="coco8.yaml")  # Crée 'yolo26n_w8a16.tflite'

# Export avec résolution web optimisée
model.export(format="litert", imgsz=320)  # Entrée 320x320 — recommandé pour le web
```

```bash
# CLI - Export basique
yolo export model=yolo26n.pt format=litert

# CLI - Export INT8 dynamique
yolo export model=yolo26n.pt format=litert quantize=w8a32

# CLI - Export pour web (résolution réduite — recommandé)
yolo export model=yolo26n.pt format=litert imgsz=320
```

### Options d'exportation

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `format` | `str` | `'litert'` | Format cible (remplace `tfjs`) |
| `imgsz` | `int` ou `tuple` | `640` | Taille d'entrée. Utilisez `320` pour le web |
| `quantize` | `int` ou `str` | `None` | Précision : `8`, `'w8a16'`, `'w8a32'` |
| `batch` | `int` | `1` | Taille du lot |
| `data` | `str` | `'coco8.yaml'` | Dataset pour calibration INT8 |

### Validation Python

```python
from ultralytics import YOLO

model = YOLO("yolo26n.tflite")
results = model("https://ultralytics.com/images/bus.jpg")
metrics = model.val(data="coco8.yaml")
```

---

## Exemple complet fonctionnel (servable avec `python -m http.server`)

Voici un projet web complet. Créez la structure suivante et lancez `python -m http.server 8000` depuis la racine du projet.

### Structure des fichiers

```
yolo-web/
├── index.html
├── sw.js
├── manifest.json
└── models/
    └── yolo26n.tflite   ← exportez avec : yolo export model=yolo26n.pt format=litert imgsz=320
```

### index.html — Application complète de détection en caméra

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>YOLO26 — Détection en temps réel</title>
  <link rel="manifest" href="manifest.json">
  <meta name="theme-color" content="#000000">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black">
  <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📷</text></svg>">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #111; color: #eee; font-family: -apple-system, system-ui, sans-serif; overflow: hidden; }
    #app { position: relative; width: 100vw; height: 100vh; }
    video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
    canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
    #hud {
      position: absolute; top: 10px; left: 10px; right: 10px;
      display: flex; justify-content: space-between; align-items: flex-start;
      pointer-events: none; z-index: 10;
    }
    .badge {
      background: rgba(0,0,0,0.6); padding: 6px 12px; border-radius: 8px;
      font-size: 13px; backdrop-filter: blur(4px);
    }
    #status { border-left: 3px solid #f59e0b; }
    #status.ready { border-left-color: #10b981; }
    #status.error { border-left-color: #ef4444; }
    #perf { text-align: right; }
    #overlay-rects { pointer-events: none; }
    #btn-cam {
      position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
      padding: 12px 28px; font-size: 16px; font-weight: 600;
      background: #3b82f6; color: #fff; border: none; border-radius: 12px;
      cursor: pointer; z-index: 20; display: none;
    }
    #btn-cam:active { background: #2563eb; }
    #install-prompt {
      position: absolute; bottom: 70px; left: 50%; transform: translateX(-50%);
      padding: 10px 20px; font-size: 14px;
      background: #8b5cf6; color: #fff; border: none; border-radius: 10px;
      cursor: pointer; z-index: 20; display: none;
    }
  </style>
</head>
<body>
  <div id="app">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay-rects"></canvas>
    <div id="hud">
      <div id="status" class="badge">Initialisation…</div>
      <div id="perf" class="badge">— FPS</div>
    </div>
    <button id="btn-cam">Activer la caméra</button>
    <button id="install-prompt">Installer l'app</button>
  </div>

  <script type="module">
    // ============================================================
    // 1. CONFIGURATION
    // ============================================================
    const MODEL_PATH = './models/yolo26n.tflite';
    const IMG_SIZE   = 320;  // Correspond à l'export : imgsz=320
    const CONF_THRES = 0.5;
    const NMS_IOU    = 0.45;

    const statusEl = document.getElementById('status');
    const perfEl   = document.getElementById('perf');
    const video    = document.getElementById('video');
    const canvas   = document.getElementById('overlay-rects');
    const ctx      = canvas.getContext('2d');
    const btnCam   = document.getElementById('btn-cam');

    // ============================================================
    // 2. SÉLECTION DU BACKEND (WebGPU > WASM > CPU)
    // ============================================================
    async function selectBackend() {
      // WebGPU
      if (navigator.gpu) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            console.log('Backend WebGPU disponible :', adapter.name);
            return 'webgpu';
          }
        } catch (e) {
          console.warn('WebGPU indisponible, fallback WASM');
        }
      }
      // WASM (fallback par défaut, threads automatiques)
      console.log('Backend WASM utilisé');
      return 'wasm';
    }

    // ============================================================
    // 3. CHARGEMENT DU MODÈLE LiteRT.js
    // ============================================================
    let interpreter = null;
    let backendUsed = '';

    async function loadModel() {
      statusEl.textContent = 'Chargement du modèle…';

      const backend = await selectBackend();
      backendUsed = backend;

      const response = await fetch(MODEL_PATH);
      if (!response.ok) throw new Error(`Modèle introuvable : ${MODEL_PATH}`);
      const buffer = await response.arrayBuffer();

      // Import dynamique de LiteRT.js
      // Option A — CDN (recommandé pour usage simple) :
      //   Ajoutez <script src="https://cdn.jsdelivr.net/npm/@google-ai-edge/litert/dist/litert.js"></script>
      //   dans le <head>, puis utilisez window.litert ci-dessous.
      // Option B — ESM (si bundled avec un outil) :
      //   import * as litert from '@google-ai-edge/litert';

      const litert = window.litert; // Disponible via le CDN script tag
      if (!litert) {
        throw new Error(
          'LiteRT.js non chargé. Ajoutez dans le <head> :\n' +
          '<script src="https://cdn.jsdelivr.net/npm/@google-ai-edge/litert/dist/litert.js"><\/script>'
        );
      }

      const options = {};
      if (backend === 'webgpu') {
        options.backend = 'webgpu';
      } else {
        options.backend = 'wasm';
        // Nbre de threads = tous les cœurs disponibles
        if (navigator.hardwareConcurrency) {
          options.numThreads = navigator.hardwareConcurrency;
        }
      }

      interpreter = await litert.Interpreter.create(buffer, options);
      statusEl.textContent = `Prêt — ${backend.toUpperCase()}`;
      statusEl.classList.add('ready');
    }

    // ============================================================
    // 4. PRÉTRAITEMENT DE L'IMAGE
    // ============================================================
    function preprocess(source) {
      const c = document.createElement('canvas');
      c.width = IMG_SIZE;
      c.height = IMG_SIZE;
      const cx = c.getContext('2d');
      cx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);
      const pixels = cx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

      // Float32Array CHW : [R0,G0,B0, R1,G1,B1, …]
      const floatData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
      for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        const p = i * 4;
        floatData[i * 3]     = pixels[p]     / 255.0; // R
        floatData[i * 3 + 1] = pixels[p + 1] / 255.0; // G
        floatData[i * 3 + 2] = pixels[p + 2] / 255.0; // B
      }
      return floatData;
    }

    // ============================================================
    // 5. POST-TRAITEMENT (NMS inclus)
    // ============================================================
    function postprocess(output, origW, origH) {
      const detections = [];
      const numClasses = output.length / (IMG_SIZE * IMG_SIZE) - 4;

      // Adaptatif : format [1, 4+numClasses, numBoxes] ou [numBoxes, 4+numClasses]
      const isTransposed = output.length > IMG_SIZE * IMG_SIZE;

      if (isTransposed) {
        // Format transposé — traitement par colonnes (simplifié)
        const numBoxes = IMG_SIZE * IMG_SIZE;
        for (let i = 0; i < Math.min(numBoxes, 8400); i++) {
          let maxConf = 0, maxIdx = 0;
          for (let c = 4; c < 4 + numClasses; c++) {
            const conf = output[c * numBoxes + i];
            if (conf > maxConf) { maxConf = conf; maxIdx = c - 4; }
          }
          if (maxConf > CONF_THRES) {
            detections.push({
              x: output[0 * numBoxes + i],
              y: output[1 * numBoxes + i],
              w: output[2 * numBoxes + i],
              h: output[3 * numBoxes + i],
              classId: maxIdx,
              confidence: maxConf
            });
          }
        }
      } else {
        // Format plat : chaque détection = [x, y, w, h, ...classes]
        const stride = 4 + numClasses;
        const numBoxes = output.length / stride;
        for (let i = 0; i < numBoxes; i++) {
          const off = i * stride;
          let maxConf = 0, maxIdx = 0;
          for (let c = 4; c < stride; c++) {
            if (output[off + c] > maxConf) { maxConf = output[off + c]; maxIdx = c - 4; }
          }
          if (maxConf > CONF_THRES) {
            detections.push({
              x: output[off + 0],
              y: output[off + 1],
              w: output[off + 2],
              h: output[off + 3],
              classId: maxIdx,
              confidence: maxConf
            });
          }
        }
      }

      // Non-Maximum Suppression simple
      detections.sort((a, b) => b.confidence - a.confidence);
      const keep = [];
      for (const d of detections) {
        let dominated = false;
        for (const k of keep) {
          if (d.classId === k.classId && iou(d, k) > NMS_IOU) { dominated = true; break; }
        }
        if (!dominated) keep.push(d);
      }

      // Coordonnées → écran
      const scaleX = origW / IMG_SIZE;
      const scaleY = origH / IMG_SIZE;
      return keep.map(d => ({
        x: d.x * scaleX,
        y: d.y * scaleY,
        w: d.w * scaleX,
        h: d.h * scaleY,
        classId: d.classId,
        confidence: d.confidence
      }));
    }

    function iou(a, b) {
      const x1 = Math.max(a.x - a.w / 2, b.x - b.w / 2);
      const y1 = Math.max(a.y - a.h / 2, b.y - b.h / 2);
      const x2 = Math.min(a.x + a.w / 2, b.x + b.w / 2);
      const y2 = Math.min(a.y + a.h / 2, b.y + b.h / 2);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      return inter / (a.w * a.h + b.w * b.h - inter + 1e-6);
    }

    // ============================================================
    // 6. DESSIN DES DÉTECTIONS
    // ============================================================
    const COLORS = [
      '#ef4444','#f59e0b','#10b981','#3b82f6','#8b5cf6',
      '#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'
    ];

    function draw(detections) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      detections.forEach(det => {
        const color = COLORS[det.classId % COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(det.x, det.y, det.w, det.h);
        ctx.fillStyle = color;
        ctx.font = 'bold 13px monospace';
        ctx.fillText(
          `${det.classId} ${(det.confidence * 100).toFixed(0)}%`,
          det.x + 4,
          det.y - 6
        );
      });
    }

    // ============================================================
    // 7. BOUCLE DE DÉTECTION
    // ============================================================
    let running = false;
    let lastTime = 0;
    let frameCount = 0;
    let fpsDisplay = 0;

    async function detectLoop() {
      if (!running || !interpreter || video.paused || video.ended) return;

      // FPS
      frameCount++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        fpsDisplay = frameCount;
        frameCount = 0;
        lastTime = now;
        perfEl.textContent = `${fpsDisplay} FPS — ${backendUsed.toUpperCase()}`;
      }

      // Prétraiter
      const inputData = preprocess(video);

      // Remplir le tenseur d'entrée
      const inputTensor = interpreter.getInputTensor(0);
      inputTensor.setData(inputData);

      // Inférence
      await interpreter.invoke();

      // Récupérer la sortie
      const outputTensor = interpreter.getOutputTensor(0);
      const output = outputTensor.getData();

      // Post-traiter et dessiner
      const dets = postprocess(output, canvas.width, canvas.height);
      draw(dets);

      requestAnimationFrame(detectLoop);
    }

    // ============================================================
    // 8. INITIALISATION CAMÉRA
    // ============================================================
    async function startCamera() {
      const constraints = {
        video: {
          facingMode: { ideal: 'environment' },
          width:  { ideal: 1280 },
          height: { ideal: 720 }
        }
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        btnCam.style.display = 'none';
        running = true;
        detectLoop();
      } catch (err) {
        statusEl.textContent = `Erreur caméra : ${err.message}`;
        statusEl.classList.add('error');
      }
    }

    // ============================================================
    // 9. POINT D'ENTRÉE
    // ============================================================
    async function main() {
      try {
        await loadModel();
        // Sur mobile, la caméra nécessite un geste utilisateur
        if (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) {
          btnCam.style.display = 'block';
          btnCam.addEventListener('click', startCamera);
        } else {
          await startCamera();
        }
      } catch (err) {
        statusEl.textContent = `Erreur : ${err.message}`;
        statusEl.classList.add('error');
        console.error(err);
      }
    }

    main();
  </script>

  <!-- LiteRT.js — CDN (chargé en dehors du module pour disponibilité globale) -->
  <script src="https://cdn.jsdelivr.net/npm/@google-ai-edge/litert/dist/litert.js"></script>
</body>
</html>
```

### manifest.json — PWA installable

```json
{
  "name": "YOLO26 Détection",
  "short_name": "YOLO26",
  "description": "Détection d'objets en temps réel avec YOLO26 dans le navigateur",
  "start_url": "./index.html",
  "display": "standalone",
  "background_color": "#111111",
  "theme_color": "#3b82f6",
  "orientation": "portrait",
  "icons": [
    {
      "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📷</text></svg>",
      "sizes": "any",
      "type": "image/svg+xml"
    }
  ]
}
```

### sw.js — Service Worker pour le cache hors ligne

```javascript
const CACHE_NAME = 'yolo26-v1';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './manifest.json',
  './models/yolo26n.tflite'
];

// Installation : pré-cacher les assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Pré-cache des assets');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  // Activer immédiatement sans attendre les anciens SW
  self.skipWaiting();
});

// Activation : nettoyer les anciens caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Stratégie : Cache-First pour les modèles, Network-First pour le reste
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Modèles .tflite → Cache First (priorité au cache)
  if (url.pathname.endsWith('.tflite')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        });
      })
    );
    return;
  }

  // Autres ressources → Network First avec fallback cache
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
```

### Enregistrement du Service Worker dans index.html

Ajoutez ce bloc à la fin du `<body>` dans `index.html` (après le `<script type="module">`) :

```html
<script>
  // Enregistrement du Service Worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').then((reg) => {
      console.log('[App] Service Worker enregistré, scope :', reg.scope);
    }).catch((err) => {
      console.warn('[App] Échec enregistrement SW :', err);
    });
  }

  // Prompt d'installation PWA
  let deferredPrompt;
  const installBtn = document.getElementById('install-prompt');
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn.style.display = 'block';
  });
  installBtn.addEventListener('click', () => {
    deferredPrompt.prompt();
    deferredPrompt.userChoice.then((choice) => {
      if (choice.outcome === 'accepted') installBtn.style.display = 'none';
      deferredPrompt = null;
    });
  });
</script>
```

### Lancement

```bash
cd yolo-web/
python -m http.server 8000
# → Ouvrir http://localhost:8000
```

> **Note** : Le Service Worker nécessite un serveur HTTP (pas une simple ouverture de fichier). `python -m http.server` fonctionne parfaitement.

---

## Sélection WebGPU / WASM — Code réutilisable

### Détection automatique du meilleur backend

```javascript
async function selectBestBackend() {
  const backends = [];

  // Test WebGPU
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.requestAdapterInfo?.() || {};
        backends.push({
          name: 'webgpu',
          priority: 1,
          info: info.description || adapter.name || 'GPU'
        });
      }
    } catch (e) {
      console.warn('WebGPU non disponible');
    }
  }

  // WASM toujours disponible
  backends.push({
    name: 'wasm',
    priority: 2,
    info: `${navigator.hardwareConcurrency || '?'} threads`
  });

  // Trier par priorité
  backends.sort((a, b) => a.priority - b.priority);

  return backends[0] || { name: 'wasm', priority: 99 };
}

async function createInterpreterWithBestBackend(modelBuffer) {
  const best = await selectBestBackend();
  console.log(`Backend sélectionné : ${best.name} (${best.info})`);

  const options = {};
  if (best.name === 'webgpu') {
    options.backend = 'webgpu';
  } else {
    options.backend = 'wasm';
    if (navigator.hardwareConcurrency) {
      options.numThreads = navigator.hardwareConcurrency;
    }
  }

  const litert = window.litert || await import('@google-ai-edge/litert');
  return litert.Interpreter.create(modelBuffer, options);
}
```

### Benchmark comparatif des backends

```javascript
async function benchmarkBackends(modelBuffer) {
  const litert = window.litert || await import('@google-ai-edge/litert');
  const results = {};

  // Créer un tenseur d'entrée factice
  const fakeInput = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

  const configs = [
    { name: 'wasm', options: { backend: 'wasm', numThreads: navigator.hardwareConcurrency || 4 } },
  ];

  // Ajouter WebGPU si disponible
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        configs.unshift({ name: 'webgpu', options: { backend: 'webgpu' } });
      }
    } catch (e) { /* ignoré */ }
  }

  for (const config of configs) {
    try {
      const interp = await litert.Interpreter.create(modelBuffer, config.options);
      const times = [];

      // Warm-up
      const inputTensor = interp.getInputTensor(0);
      inputTensor.setData(fakeInput);
      await interp.invoke();

      // Mesure
      for (let i = 0; i < 20; i++) {
        const start = performance.now();
        inputTensor.setData(fakeInput);
        await interp.invoke();
        times.push(performance.now() - start);
      }

      const avg = times.reduce((a, b) => a + b) / times.length;
      results[config.name] = {
        avgMs: avg.toFixed(2),
        fps: (1000 / avg).toFixed(1),
        min: Math.min(...times).toFixed(2),
        max: Math.max(...times).toFixed(2)
      };
    } catch (e) {
      results[config.name] = { error: e.message };
    }
  }

  console.table(results);
  return results;
}
```

---

## Performance — Guide d'optimisation

### Hiérarchie de performance (du plus rapide au plus lent)

| Rang | Backend | Description | Cas d'usage |
|------|---------|-------------|-------------|
| 1 | **WebGPU** | GPU natif via le navigateur | Chrome 113+, appareils performants |
| 2 | **WASM + threads** | WebAssembly multi-threadé | Tout navigateur récent |
| 3 | **WASM seul** | Mono-thread | Navigateurs sans SharedArrayBuffer |
| 4 | **CPU/JS** | JavaScript pur | Fallback extrême, très lent |

### Conseils d'optimisation

```javascript
// 1. Réduire la taille d'entrée (impact majeur !)
//    320px au lieu de 640px ≈ 4x plus rapide, précision suffisante pour la plupart des cas
model.export(format="litert", imgsz=320)

// 2. Quantifier le modèle (réduit la taille et accélère)
model.export(format="litert", quantize="w8a32")  // INT8 dynamique
model.export(format="litert", quantize="w8a16", data="coco8.yaml")  // INT8 statique

// 3. Limiter le FPS côté client (évite de surcharger le CPU)
function throttle(fn, maxFps = 20) {
  let last = 0;
  const interval = 1000 / maxFps;
  return (...args) => {
    const now = Date.now();
    if (now - last >= interval) {
      last = now;
      return fn(...args);
    }
  };
}

// 4. Détecter la puissance de l'appareil et adapter
function getDeviceProfile() {
  const memory = navigator.deviceMemory || 4;
  const cores  = navigator.hardwareConcurrency || 4;

  if (memory <= 2 || cores <= 2) return { imgsz: 160, fps: 10 }; // Basique
  if (memory <= 4 || cores <= 4) return { imgsz: 320, fps: 15 }; // Mobile standard
  return { imgsz: 320, fps: 20 }; // Desktop / mobile haut de gamme
}

// 5. Utiliser SharedArrayBuffer si disponible (WASM multi-thread)
//    Nécessite les en-têtes Cross-Origin-Isolation :
//    Cross-Origin-Embedder-Policy: require-corp
//    Cross-Origin-Opener-Policy: same-origin
```

### Taille du modèle vs performance

| Modèle | Taille | FPS (WASM 4c) | FPS (WebGPU) |
|--------|--------|----------------|--------------|
| YOLO26n FP32 640px | ~6 MB | 15-20 | 25-30 |
| YOLO26n INT8 640px | ~3 MB | 20-25 | 30-40 |
| YOLO26n FP32 320px | ~2 MB | 25-30 | 35-45 |
| YOLO26n INT8 320px | ~1.5 MB | 30-40 | 40-60 |

---

## Intégration WebView pour React Native

### Composant React Native avec WebView

```javascript
// YOLOWebView.js
import React, { useRef, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { WebView } from 'react-native-webview';

const YOLOWebView = ({ onDetection }) => {
  const webViewRef = useRef(null);

  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://cdn.jsdelivr.net/npm/@google-ai-edge/litert/dist/litert.js"></script>
      <style>
        body { margin: 0; padding: 0; background: #000; }
        #container { position: relative; width: 100%; height: 100vh; }
        video { width: 100%; height: 100%; object-fit: cover; }
        canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
      </style>
    </head>
    <body>
      <div id="container">
        <video id="video" autoplay playsinline></video>
        <canvas id="overlay"></canvas>
      </div>

      <script>
        const IMG_SIZE = 320;
        let detector = null;

        async function initYOLO() {
          const response = await fetch('/models/yolo26n.tflite');
          const buffer = await response.arrayBuffer();

          // Sélection du backend
          const backend = (navigator.gpu) ? 'webgpu' : 'wasm';
          detector = await litert.Interpreter.create(buffer, { backend });

          const video = document.getElementById('video');
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
          });

          video.srcObject = stream;
          video.onloadedmetadata = () => {
            const canvas = document.getElementById('overlay');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectLoop();
          };
        }

        async function detectLoop() {
          if (!detector) return;

          const video = document.getElementById('video');
          const canvas = document.getElementById('overlay');
          const ctx = canvas.getContext('2d');

          // Prétraiter
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = IMG_SIZE;
          tempCanvas.height = IMG_SIZE;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
          const imageData = tempCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

          const input = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
          for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            const p = i * 4;
            input[i * 3]     = imageData.data[p]     / 255.0;
            input[i * 3 + 1] = imageData.data[p + 1] / 255.0;
            input[i * 3 + 2] = imageData.data[p + 2] / 255.0;
          }

          // Inférence
          detector.getInputTensor(0).setData(input);
          await detector.invoke();
          const output = detector.getOutputTensor(0).getData();

          // Post-traiter (simplifié)
          const detections = [];
          const stride = 84;
          for (let i = 0; i < output.length / stride; i++) {
            const off = i * stride;
            let maxConf = 0, maxIdx = 0;
            for (let j = 4; j < stride; j++) {
              if (output[off + j] > maxConf) { maxConf = output[off + j]; maxIdx = j - 4; }
            }
            if (maxConf > 0.5) {
              detections.push({
                x: output[off], y: output[off + 1],
                w: output[off + 2], h: output[off + 3],
                classId: maxIdx, confidence: maxConf
              });
            }
          }

          // Dessiner
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const sx = canvas.width / IMG_SIZE, sy = canvas.height / IMG_SIZE;
          detections.forEach(d => {
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
            ctx.fillStyle = '#00ff00';
            ctx.font = '14px Arial';
            ctx.fillText(d.classId + ': ' + (d.confidence * 100).toFixed(0) + '%',
              d.x * sx, d.y * sy - 5);
          });

          window.ReactNativeWebView.postMessage(JSON.stringify(detections));
          requestAnimationFrame(detectLoop);
        }

        initYOLO();
      <\/script>
    </body>
    </html>
  `;

  const handleMessage = useCallback((event) => {
    try {
      const detections = JSON.parse(event.nativeEvent.data);
      onDetection?.(detections);
    } catch (error) {
      console.error('Erreur parsing:', error);
    }
  }, [onDetection]);

  return (
    <View style={{ flex: 1 }}>
      <WebView
        ref={webViewRef}
        source={{ html: htmlContent }}
        style={{ flex: 1 }}
        onMessage={handleMessage}
        javaScriptEnabled={true}
        mediaPlaybackRequiresUserAction={false}
        allowsInlineMediaPlayback={true}
      />
    </View>
  );
};

export default YOLOWebView;
```

---

## Problèmes courants et solutions

### 1. "WebGPU not supported"

```javascript
// Fallback automatique — ne jamais forcer WebGPU
async function createInterpreterSafe(modelBuffer) {
  const litert = window.litert;
  const options = { backend: 'wasm' }; // défaut sûr

  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) options.backend = 'webgpu';
    } catch (e) { /* rester en WASM */ }
  }

  return litert.Interpreter.create(modelBuffer, options);
}
```

### 2. Modèle trop lent

```bash
# Exporter en 320px + INT8
yolo export model=yolo26n.pt format=litert imgsz=320 quantize=w8a32
```

### 3. "Out of memory" dans le navigateur

```javascript
// Réduire la résolution, libérer la mémoire, limiter le FPS
const profile = { imgsz: 160, fps: 10 }; // Profil ultra-léger
```

### 4. Latence au premier chargement

```javascript
// Le Service Worker (sw.js) pré-cache le modèle
// Au rechargement, le modèle est servit depuis le cache local
// Vérifier : Chrome DevTools → Application → Cache Storage → yolo26-v1
```

### 5. Caméra ne démarre pas

```javascript
// Vérifier les permissions et le contexte sécurisé
if (!window.isSecureContext) {
  console.error('Nécessite HTTPS ou localhost pour accéder à la caméra');
}
// Les Service Workers ne fonctionnent que dans un contexte sécurisé
```

---

## Flux de travail

```
Entraîner YOLO26
       ↓
Exporter vers LiteRT (imgsz=320 recommandé)
       ↓
  yolo26n.tflite
       ↓
┌──────┼──────────────────────────────┐
│      │                              │
▼      ▼                              ▼
Web    React Native WebView    Mobile natif
       (ce guide)              (voir guides Android/iOS)
│
├── LiteRT.js + WebGPU/WASM
├── Service Worker (offline)
├── PWA (installable)
└── python -m http.server pour le test
```

---

## Ressources

- [LiteRT.js — GitHub officiel](https://github.com/google-ai-edge/LiteRT.js)
- [LiteRT.js — Documentation Google](https://developers.google.com/edge/litert/web)
- [WebGPU — MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [Ultralytics — Export LiteRT](https://docs.ultralytics.com/integrations/litert)
- [React Native WebView](https://github.com/nicknisi/react-native-webview)
