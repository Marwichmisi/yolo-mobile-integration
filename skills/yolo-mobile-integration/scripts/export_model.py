#!/usr/bin/env python3
"""
YOLO26 Mobile Export Script
Automates model export with format detection and quantization.

Usage:
    python export_model.py --model yolo26n --format coreml --quantize int8
    python export_model.py --model yolo26n --format litert --quantize w8a32
    python export_model.py --model yolo26s-seg --format ncnn
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Erreur: ultralytics n'est pas installé.")
    print("Installez-le avec: pip install ultralytics")
    sys.exit(1)


# Format mapping: friendly name → ultralytics format name
FORMAT_MAP = {
    "coreml": "coreml",
    "tflite": "litert",
    "litert": "litert",
    "ncnn": "ncnn",
    "onnx": "onnx",
    "torchscript": "torchscript",
    "tensorrt": "engine",
    "openvino": "openvino",
    "edgetpu": "edgetpu",
    "tfjs": "tfjs",
    "mnn": "mnn",
    "rknn": "rknn",
    "qnn": "qnn",
    "executorch": "executorch",
    "paddle": "paddle",
}

# Task detection from model filename
TASK_MAP = {
    "-seg": "segment",
    "-cls": "classify",
    "-pose": "pose",
    "-obb": "obb",
    "-sem": "segment",
}

# Platform recommendations
PLATFORM_FORMATS = {
    "ios": ["coreml", "ncnn", "onnx"],
    "android": ["litert", "ncnn", "onnx"],
    "both": ["ncnn", "executorch", "onnx"],
    "web": ["tfjs", "onnx"],
    "rpi": ["ncnn", "onnx"],
    "jetson": ["engine", "onnx"],
    "coral": ["edgetpu"],
}


def detect_task(model_path: str) -> str:
    """Detect task type from model filename."""
    for suffix, task in TASK_MAP.items():
        if suffix in model_path:
            return task
    return "detect"


def get_model_info(model_path: str) -> dict:
    """Get model information."""
    model = YOLO(model_path)
    task = detect_task(model_path)
    names = model.names if hasattr(model, "names") else {}
    return {
        "model": model,
        "task": task,
        "num_classes": len(names),
        "class_names": names,
    }


def export_model(
    model_path: str,
    format_name: str,
    quantize: str = None,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    data: str = None,
    output_dir: str = None,
) -> dict:
    """
    Export a YOLO model to the specified format.

    Args:
        model_path: Path to the .pt model file
        format_name: Target format (coreml, litert, ncnn, onnx, etc.)
        quantize: Quantization type (int8, fp16, w8a32, w8a16)
        imgsz: Input image size
        half: Use FP16 quantization
        dynamic: Dynamic input shapes
        data: Dataset YAML for INT8 calibration
        output_dir: Output directory

    Returns:
        dict with export results
    """
    # Resolve format name
    ultralytics_format = FORMAT_MAP.get(format_name, format_name)

    # Validate format for platform
    print(f"Export de {model_path} vers {format_name}...")

    # Load model
    model = YOLO(model_path)
    info = get_model_info(model_path)
    print(f"Tache detectee: {info['task']}")
    print(f"Classes: {info['num_classes']}")

    # Build export kwargs
    export_kwargs = {
        "format": ultralytics_format,
        "imgsz": imgsz,
        "half": half,
        "dynamic": dynamic,
    }

    # Handle quantization
    if quantize:
        q = quantize.lower()
        if q in ("int8", "8"):
            export_kwargs["int8"] = True
            if data:
                export_kwargs["data"] = data
            print("Quantification INT8 activee")
        elif q in ("fp16", "half", "16"):
            export_kwargs["half"] = True
            print("Quantification FP16 activee")
        elif q in ("w8a32",):
            export_kwargs["int8"] = True
            print("Quantification W8A32 (dynamique) activee")
        elif q in ("w8a16",):
            export_kwargs["half"] = True
            export_kwargs["int8"] = True
            print("Quantification W8A16 activee")

    if output_dir:
        export_kwargs["exist_ok"] = True

    # Run export
    try:
        result = model.export(**export_kwargs)
        print(f"Export reussi: {result}")
        return {
            "success": True,
            "output": str(result),
            "format": format_name,
            "task": info["task"],
        }
    except Exception as e:
        print(f"Erreur lors de l'export: {e}")
        return {
            "success": False,
            "error": str(e),
            "format": format_name,
        }


def recommend_format(platform: str, task: str = "detect") -> list:
    """Recommend export formats for a target platform."""
    formats = PLATFORM_FORMATS.get(platform, ["onnx"])
    print(f"Formats recommandes pour {platform}: {', '.join(formats)}")
    return formats


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO26 vers des formats mobiles"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Chemin vers le modele .pt (ex: yolo26n.pt)",
    )
    parser.add_argument(
        "--format", "-f",
        required=True,
        choices=list(FORMAT_MAP.keys()),
        help="Format cible",
    )
    parser.add_argument(
        "--quantize", "-q",
        choices=["int8", "fp16", "w8a32", "w8a16"],
        help="Type de quantification",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'entree (defaut: 640)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Utiliser la demi-precision (FP16)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Tailles d'entree dynamiques",
    )
    parser.add_argument(
        "--data",
        help="Fichier YAML du dataset pour calibration INT8",
    )
    parser.add_argument(
        "--output", "-o",
        help="Repertoire de sortie",
    )
    parser.add_argument(
        "--recommend",
        help="Recommander des formats pour une plateforme (ios, android, both, web, rpi, jetson, coral)",
        choices=list(PLATFORM_FORMATS.keys()),
    )

    args = parser.parse_args()

    # Handle recommendation mode
    if args.recommend:
        recommend_format(args.recommend)
        return

    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Erreur: Modele non trouve: {model_path}")
        sys.exit(1)

    # Run export
    result = export_model(
        model_path=str(model_path),
        format_name=args.format,
        quantize=args.quantize,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        data=args.data,
        output_dir=args.output,
    )

    if result["success"]:
        print(f"\nExport termine avec succes!")
        print(f"Format: {result['format']}")
        print(f"Sortie: {result['output']}")
    else:
        print(f"\nEchec de l'export: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
