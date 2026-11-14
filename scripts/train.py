"""Fine-tune a YOLO model for a mission profile and export it for edge deployment.

Examples
--------
# Recon: aerial people/vehicles (VisDrone ships as a built-in Ultralytics dataset)
python scripts/train.py --data VisDrone.yaml --model yolo11n.pt --imgsz 960 --epochs 100

# Counter-UAS: your own drone dataset in YOLO format (one class: drone)
python scripts/train.py --data data/drone.yaml --model yolo11n.pt --epochs 150 --export onnx engine

Exported weights plug straight into configs/*.yaml via `detector.model`.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset YAML (YOLO format)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--export", nargs="*", default=["onnx"],
                    help="export formats: onnx engine (TensorRT) openvino tflite ...")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project,
        # small, fast-moving aerial objects: keep mosaic, add scale/rotation jitter
        mosaic=1.0, scale=0.6, degrees=10.0, fliplr=0.5, close_mosaic=10,
    )
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")
    for fmt in args.export:
        path = model.export(format=fmt, imgsz=args.imgsz, half=fmt == "engine")
        print(f"exported {fmt}: {path}")


if __name__ == "__main__":
    main()
