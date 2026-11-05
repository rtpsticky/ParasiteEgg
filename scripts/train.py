"""Train a YOLO detector on the prepared dataset.

  python scripts/train.py --data data/yolo/data.yaml --model yolo11n.pt --epochs 100

Nano (n) or small (s) models are the right size for ONNX Runtime Web.
Output always goes to ./runs/detect/<name>/ (relative to the current directory).
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# Absolute on purpose: newer ultralytics puts a *relative* project under its own
# settings runs_dir (runs/detect/runs/detect/<name>), which is easy to lose.
PROJECT = Path("runs/detect").absolute()


def retarget_checkpoint(last: Path, run_dir: Path, data: str | None):
    """Point a checkpoint made elsewhere (Colab, Kaggle, ...) at this machine's paths."""
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args") or {}
    ta["project"], ta["name"] = str(run_dir.parent), run_dir.name
    if "save_dir" in ta:
        ta["save_dir"] = str(run_dir)
    if data:
        ta["data"] = str(Path(data).absolute())
    ckpt["train_args"] = ta
    torch.save(ckpt, last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt / yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (GPU only)")
    ap.add_argument("--device", default=None, help="0, cpu, mps (Apple Silicon), ... (default: auto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="parasite")
    ap.add_argument("--resume", action="store_true", help="resume runs/detect/<name>/weights/last.pt")
    args = ap.parse_args()

    run_dir = PROJECT / args.name
    print(f"run dir: {run_dir}")

    batch = args.batch
    if batch == -1 and args.device in ("cpu", "mps"):
        batch = 8 if args.device == "cpu" else 16  # auto-batch needs CUDA

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        retarget_checkpoint(last, run_dir, args.data if Path(args.data).exists() else None)
        # device/batch/workers may differ from the original run, e.g. resume a
        # Colab GPU checkpoint on a local CPU (ultralytics allows these overrides).
        over = {"workers": args.workers}
        if args.device is not None:
            over["device"] = args.device
        if batch != -1:
            over["batch"] = batch
        YOLO(str(last)).train(resume=True, **over)
        return

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=str(PROJECT),
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
