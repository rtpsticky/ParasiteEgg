"""Export trained weights to ONNX for ONNX Runtime Web, then evaluate on the test split.

  python scripts/export_onnx.py --weights runs/detect/parasite/weights/best.pt

Output tensor: [1, 4 + num_classes, N]  (cx, cy, w, h in input pixels, then class
scores). NMS is NOT baked in; do it in JS (see web/index.html).
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/parasite/weights/best.pt")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="web/model")
    args = ap.parse_args()

    model = YOLO(args.weights)

    test_dir = Path(args.data).parent / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        m = model.val(data=args.data, split="test", imgsz=args.imgsz)
        print(f"TEST mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")

    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=17,
                             simplify=True, dynamic=False, half=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, out / "parasite.onnx")
    names = [model.names[i] for i in range(len(model.names))]
    (out / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"wrote {out / 'parasite.onnx'} and classes.json")


if __name__ == "__main__":
    main()
