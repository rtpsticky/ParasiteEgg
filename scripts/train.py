"""Train a YOLO detector on the prepared dataset.

  python scripts/train.py --data data/yolo/data.yaml --model yolo11n.pt --epochs 100

Nano (n) or small (s) models are the right size for ONNX Runtime Web.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt / yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (GPU only)")
    ap.add_argument("--device", default=None, help="0, cpu, ... (default: auto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="parasite")
    ap.add_argument("--resume", action="store_true", help="resume runs/<name>/weights/last.pt")
    args = ap.parse_args()

    if args.resume:
        YOLO(f"runs/detect/{args.name}/weights/last.pt").train(resume=True)
        return

    batch = args.batch
    if batch == -1 and args.device == "cpu":
        batch = 8  # auto-batch needs CUDA

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project="runs/detect",
        exist_ok=True,
        patience=20,
        cos_lr=True,
        # Microscope fields have no "up": rotations/flips are valid augmentations.
        # Colour jitter covers stain and phone camera white-balance differences.
        flipud=0.5,
        fliplr=0.5,
        degrees=90,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
