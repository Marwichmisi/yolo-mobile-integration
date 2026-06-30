#!/usr/bin/env python3
"""
YOLO26 Mobile Benchmark Script
Run benchmarks on exported models to measure latency, FPS, and accuracy.

Usage:
    python benchmark_mobile.py --model yolo26n.mlpackage --device iphone
    python benchmark_mobile.py --model yolo26n.onnx --device cpu
    python benchmark_mobile.py --model yolo26n.tflite --device android-gpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Erreur: ultralytics n'est pas installe.")
    print("Installez-le avec: pip install ultralytics")
    sys.exit(1)


# Device-specific configurations
DEVICE_CONFIGS = {
    "iphone": {
        "format": "coreml",
        "device": "cpu",
        "notes": "CoreML utilise le Neural Engine automatiquement sur iPhone",
    },
    "android-cpu": {
        "format": "litert",
        "device": "cpu",
        "notes": "LiteRT avec delegate CPU",
    },
    "android-gpu": {
        "format": "litert",
        "device": "gpu",
        "notes": "LiteRT avec GPU delegate",
    },
    "android-nnapi": {
        "format": "litert",
        "device": "nnapi",
        "notes": "LiteRT avec NNAPI delegate (NPU si disponible)",
    },
    "rpi": {
        "format": "ncnn",
        "device": "cpu",
        "notes": "NCNN optimise pour ARM",
    },
    "jetson": {
        "format": "engine",
        "device": "gpu",
        "notes": "TensorRT sur GPU NVIDIA",
    },
    "cpu": {
        "format": "onnx",
        "device": "cpu",
        "notes": "ONNX Runtime sur CPU",
    },
    "gpu": {
        "format": "onnx",
        "device": "cuda",
        "notes": "ONNX Runtime sur GPU CUDA",
    },
}


def detect_format_from_path(model_path: str) -> str:
    """Detect export format from file extension."""
    ext = Path(model_path).suffix.lower()
    format_map = {
        ".mlpackage": "coreml",
        ".mlmodel": "coreml",
        ".tflite": "litert",
        ".onnx": "onnx",
        ".engine": "engine",
        ".plan": "engine",
        ".torchscript": "torchscript",
        ".pt": "pytorch",
    }
    return format_map.get(ext, "unknown")


def benchmark_model(
    model_path: str,
    device: str = "cpu",
    imgsz: int = 640,
    num_runs: int = 100,
    warmup_runs: int = 10,
    half: bool = False,
    data: str = None,
) -> dict:
    """
    Benchmark a YOLO model.

    Args:
        model_path: Path to the exported model file
        device: Target device configuration
        imgsz: Input image size
        num_runs: Number of inference runs to average
        warmup_runs: Number of warmup runs before timing
        half: Use FP16
        data: Dataset YAML for validation

    Returns:
        dict with benchmark results
    """
    print(f"Benchmark de {model_path}...")
    print(f"Appareil: {device}")
    print(f"Taille d'entree: {imgsz}")
    print(f"Lancements: {num_runs} (avec {warmup_runs} warmup)")

    # Get device config
    config = DEVICE_CONFIGS.get(device, DEVICE_CONFIGS["cpu"])

    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur de chargement: {e}")
        return {"success": False, "error": str(e)}

    # Create a dummy image for benchmarking
    import numpy as np
    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print("Echauffement...")
    for _ in range(warmup_runs):
        _ = model(dummy_image, verbose=False)

    # Benchmark
    print("Benchmark en cours...")
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_image, verbose=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    fps = 1000.0 / avg_latency

    # Sort for percentiles
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

    results = {
        "success": True,
        "model": model_path,
        "device": device,
        "format": config["format"],
        "imgsz": imgsz,
        "num_runs": num_runs,
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "fps": round(fps, 1),
        "notes": config["notes"],
    }

    # Run validation if dataset provided
    if data:
        print("Validation en cours...")
        try:
            val_results = model.val(data=data)
            results["mAP50"] = round(val_results.box.map50, 4)
            results["mAP50_95"] = round(val_results.box.map, 4)
        except Exception as e:
            print(f"Erreur de validation: {e}")
            results["val_error"] = str(e)

    return results


def print_results(results: dict):
    """Pretty-print benchmark results."""
    if not results["success"]:
        print(f"\nEchec: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("RESULTATS DU BENCHMARK")
    print("=" * 60)
    print(f"Modele:    {results['model']}")
    print(f"Appareil:  {results['device']}")
    print(f"Format:    {results['format']}")
    print(f"Entree:    {results['imgsz']}x{results['imgsz']}")
    print("-" * 60)
    print(f"Latence moyenne:  {results['avg_latency_ms']} ms")
    print(f"Latence min:      {results['min_latency_ms']} ms")
    print(f"Latence max:      {results['max_latency_ms']} ms")
    print(f"P50:              {results['p50_latency_ms']} ms")
    print(f"P95:              {results['p95_latency_ms']} ms")
    print(f"P99:              {results['p99_latency_ms']} ms")
    print(f"FPS:              {results['fps']}")
    if "mAP50_95" in results:
        print(f"mAP50-95:         {results['mAP50_95']}")
        print(f"mAP50:            {results['mAP50']}")
    print("-" * 60)
    print(f"Note: {results['notes']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO26 pour appareils mobiles"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Chemin vers le modele exporte",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=list(DEVICE_CONFIGS.keys()),
        help="Appareil cible (defaut: cpu)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'entree (defaut: 640)",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=100,
        name="num_runs",
        help="Nombre de lancements (defaut: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Lancements d'echauffement (defaut: 10)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Utiliser la demi-precision",
    )
    parser.add_argument(
        "--data",
        help="Dataset YAML pour la validation",
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier de sortie JSON",
    )

    args = parser.parse_args()

    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Erreur: Modele non trouve: {model_path}")
        sys.exit(1)

    # Run benchmark
    results = benchmark_model(
        model_path=str(model_path),
        device=args.device,
        imgsz=args.imgsz,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        half=args.half,
        data=args.data,
    )

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResultats sauvegardes dans: {output_path}")


if __name__ == "__main__":
    main()
