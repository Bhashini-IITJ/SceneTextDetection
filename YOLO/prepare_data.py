"""Convert BSTD (and optionally MIST) annotations into an Ultralytics YOLO detection dataset.

- Only images whose BSTD split is "train" are used; the BSTD test split is never touched.
- The train images are split 90/10 into train/val, stratified by BSTD folder (seed 20260930).
  With the default arguments this reproduces the 4,737 / 526 split used for the released models.
- Every annotation polygon (>= 3 points) becomes its enclosing axis-aligned box (single class "text"),
  clipped to the image. EXIF-rotated images whose coordinates were stored in raw sensor orientation
  are corrected so the boxes match the image as cv2 reads it.
- --mist_zip adds every N-th frame (default 2) of each MIST training video to train (BSTD+MIST model).
  MIST test frames are never used.

Example:
  python prepare_data.py --gt /data/BSTD/BSTD_v17.57.json --image_root /data/BSTD --out data/bstd
  python prepare_data.py --gt ... --image_root ... --mist_zip /data/mist.zip --out data/bstd_mist
"""
import argparse
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some BSTD photos exceed Pillow's default limit; only EXIF is read

EXIF_MAP = {  # raw (x, y) -> displayed (x, y) for EXIF orientation tags
    2: lambda x, y, w, h: (w - 1 - x, y), 3: lambda x, y, w, h: (w - 1 - x, h - 1 - y),
    4: lambda x, y, w, h: (x, h - 1 - y), 5: lambda x, y, w, h: (y, x),
    6: lambda x, y, w, h: (h - 1 - y, x), 7: lambda x, y, w, h: (h - 1 - y, w - 1 - x),
    8: lambda x, y, w, h: (y, w - 1 - x),
}


def split_keys(rows, seed, val_fraction):
    groups = defaultdict(list)
    for key, row in rows:
        groups[row.get("folderName", key.split("_", 1)[0])].append(key)
    rng, assign = random.Random(seed), {}
    for g in sorted(groups):
        items = sorted(groups[g])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
        for i, k in enumerate(items):
            assign[k] = "val" if i < n_val else "train"
    return assign


def yolo_line(pts, w, h):
    x1, y1 = max(0.0, min(p[0] for p in pts)), max(0.0, min(p[1] for p in pts))
    x2, y2 = min(w - 1.0, max(p[0] for p in pts)), min(h - 1.0, max(p[1] for p in pts))
    if x2 <= x1 or y2 <= y1:
        return None
    return f"0 {(x1 + x2) / (2 * w):.7f} {(y1 + y2) / (2 * h):.7f} {(x2 - x1) / w:.7f} {(y2 - y1) / h:.7f}"


def bstd(gt_path, image_root, out, seed, val_fraction):
    data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    rows = [(k, v) for k, v in data.items() if v.get("split") == "train"]
    assign = split_keys(rows, seed, val_fraction)
    n = defaultdict(int)
    for i, (key, row) in enumerate(rows, 1):
        src = Path(image_root) / row["image_name"]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {src}")
        h, w = img.shape[:2]
        with Image.open(src) as im:
            raw_w, raw_h = im.size
            orient = int(im.getexif().get(274, 1))
        anns = [a for a in row.get("annotations", {}).values() if len(a.get("coordinates") or []) >= 3]
        polys = [[(float(x), float(y)) for x, y in a["coordinates"]] for a in anns]
        pts = [p for poly in polys for p in poly]
        fit_raw = sum(0 <= x < raw_w and 0 <= y < raw_h for x, y in pts)
        fit_disp = sum(0 <= x < w and 0 <= y < h for x, y in pts)
        if orient in EXIF_MAP and fit_raw > fit_disp:  # coordinates are in raw orientation
            polys = [[EXIF_MAP[orient](x, y, raw_w, raw_h) for x, y in poly] for poly in polys]
        lines = [l for l in (yolo_line(p, w, h) for p in polys) if l]
        split = assign[key]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "images" / split / f"{key}{src.suffix.lower() or '.jpg'}")
        (out / "labels" / split / f"{key}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n[split] += 1; n[f"{split}_boxes"] += len(lines)
        if i % 500 == 0:
            print(f"BSTD {i}/{len(rows)}", flush=True)
    return n


def mist(mist_zip, out, every):
    z = zipfile.ZipFile(mist_zip)
    names = [x for x in z.namelist() if x.startswith("Images/Train/") and "__dup" not in x]
    videos = defaultdict(list)
    for x in names:
        m = re.match(r"(.+)_image_(\d+)", Path(x).stem)
        if m:
            videos[m.group(1)].append((int(m.group(2)), x))
    n = defaultdict(int)
    for frames in videos.values():
        for i, (_, x) in enumerate(sorted(frames)):
            if i % every:
                continue
            raw = z.read(x)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(x).stem
            shapes = json.loads(z.read(f"gt/Train/{stem}.json")).get("shapes", [])
            polys = [s.get("points", []) for s in shapes if len(s.get("points", [])) >= 3]
            lines = [l for l in (yolo_line(p, w, h) for p in polys) if l]
            (out / "images/train" / f"mist__{stem}.png").write_bytes(raw)
            (out / "labels/train" / f"mist__{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n["mist_train"] += 1; n["mist_boxes"] += len(lines)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="BSTD annotation JSON (e.g. BSTD_v17.57.json)")
    p.add_argument("--image_root", required=True, help="folder that image_name paths in the JSON are relative to")
    p.add_argument("--out", required=True, help="output dataset folder (must not exist)")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--mist_zip", default=None, help="optional MIST zip (Images/Train, gt/Train) for the BSTD+MIST model")
    p.add_argument("--mist_every", type=int, default=2, help="use every N-th frame of each MIST training video")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    counts = bstd(a.gt, a.image_root, out, a.seed, a.val_fraction)
    if a.mist_zip:
        counts.update(mist(a.mist_zip, out, a.mist_every))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n")
    print(json.dumps(dict(counts), indent=1))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
