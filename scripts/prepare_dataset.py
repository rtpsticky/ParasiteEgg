"""Convert Chula-ParasiteEgg-11 (COCO json) into an Ultralytics YOLO dataset.

- Finds the COCO label json and images anywhere under --src (zip layout agnostic).
- Stratified train/val split from the official training set.
- Optional held-out test split from the test zip + test_labels_200.json.
- Downscales large images (phone captures are up to 4032px) to --max-side to
  cut disk usage and data-loading time. Boxes are normalised, so they stay valid.

Usage:
  python scripts/prepare_dataset.py --src data/raw/train --out data/yolo \
      [--test-src data/raw/test --test-json data/raw/test_labels_200.json]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_coco_json(root: Path) -> Path:
    cands = [p for p in root.rglob("*.json") if not p.name.startswith("._")]
    for p in cands:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and {"images", "annotations", "categories"} <= d.keys():
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"No COCO-style json found under {root}")


def index_images(root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"):
            idx.setdefault(p.name, p)
    return idx


def process_one(job):
    src, dst_img, dst_lbl, W, H, boxes, max_side = job
    try:
        im = Image.open(src)
        # Normalise boxes against the size the annotator saw (COCO width/height).
        if im.size != (W, H) and im.size == (H, W):
            im = ImageOps.exif_transpose(im)
        if im.size != (W, H):
            # Size mismatch we can't explain: trust the json, scale boxes to actual.
            sx, sy = im.size[0] / W, im.size[1] / H
            boxes = [(c, x * sx, y * sy, w * sx, h * sy) for c, x, y, w, h in boxes]
            W, H = im.size
        im = im.convert("RGB")
        s = max_side / max(W, H)
        if s < 1:
            im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        im.save(dst_img, quality=95)  # saved without EXIF -> no surprise rotation
        lines = []
        for c, x, y, w, h in boxes:
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            lines.append(
                f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
            )
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        return None
    except Exception as e:  # keep going; report at the end
        return f"{src}: {e}"


def build_jobs(coco: dict, img_idx: dict, cat_map: dict, out: Path, split_of, max_side: int):
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append((cat_map[a["category_id"]], *a["bbox"]))
    jobs, missing = [], 0
    for im in coco["images"]:
        name = Path(im["file_name"]).name
        src = img_idx.get(name)
        if src is None:
            missing += 1
            continue
        split = split_of(im["id"])
        stem = Path(name).stem
        dst_img = out / "images" / split / f"{stem}.jpg"
        dst_lbl = out / "labels" / split / f"{stem}.txt"
        jobs.append((src, dst_img, dst_lbl, im["width"], im["height"], anns[im["id"]], max_side))
    return jobs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="extracted training zip")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--test-src", type=Path, help="extracted test zip (optional)")
    ap.add_argument("--test-json", type=Path, help="test_labels_200.json (optional)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jpath = find_coco_json(args.src)
    print(f"labels: {jpath}")
    coco = json.loads(jpath.read_text(encoding="utf-8"))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    print(f"{len(names)} classes: {names}")

    # Stratify by the first annotation's class (each image is one egg type).
    first_cls = {}
    for a in coco["annotations"]:
        first_cls.setdefault(a["image_id"], cat_map[a["category_id"]])
    by_cls = defaultdict(list)
    for im in coco["images"]:
        by_cls[first_cls.get(im["id"], -1)].append(im["id"])
    rng = random.Random(args.seed)
    val_ids = set()
    for ids in by_cls.values():
        rng.shuffle(ids)
        val_ids.update(ids[: round(len(ids) * args.val_frac)])

    jobs, missing = build_jobs(
        coco, index_images(args.src), cat_map, args.out,
        lambda i: "val" if i in val_ids else "train", args.max_side,
    )

    if args.test_src and args.test_json:
        tcoco = json.loads(args.test_json.read_text(encoding="utf-8"))
        tcat = {c["id"]: cat_map[c["id"]] for c in tcoco["categories"] if c["id"] in cat_map}
        tj, tm = build_jobs(tcoco, index_images(args.test_src), tcat, args.out,
                            lambda i: "test", args.max_side)
        jobs += tj
        missing += tm

    for sp in ("train", "val", "test"):
        (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    print(f"converting {len(jobs)} images (missing on disk: {missing}) ...")
    errors = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, err in enumerate(ex.map(process_one, jobs, chunksize=16), 1):
            if err:
                errors.append(err)
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}")
    for e in errors[:20]:
        print("ERROR", e)
    print(f"done, {len(errors)} errors")

    has_test = any((args.out / "images" / "test").iterdir())
    yaml = [f"path: {args.out.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if has_test:
        yaml.append("test: images/test")
    yaml.append("names:")
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (args.out / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
