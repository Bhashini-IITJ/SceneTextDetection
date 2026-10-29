"""Evaluate a YOLO text detector on BSTD (detection precision / recall / F1).

Every BSTD annotation polygon with >= 3 points is a ground-truth word. A prediction is a true positive
when its polygon IoU with an unmatched GT polygon is >= --iou_thr (greedy one-to-one matching by
IoU). Reported numbers are pooled over all words of the split. Predictions can be saved as JSON.

Example:
  python test.py --gt /data/BSTD/BSTD_v17.57.json --image_root /data/BSTD --split test
  python test.py --gt ... --image_root ... --split test --model_name yolo11n_bstd_mist_1920 --save_json preds.json
"""
import argparse
import json
import os

from shapely.geometry import Polygon

from yolo_detector import DEFAULT_MODEL, YOLO_detector, model_info


def poly(pts):
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def match(preds, gts, thr):
    P, G = [poly(p) for p in preds], [poly(g) for g in gts]
    pairs = []
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            if p.area <= 0 or g.area <= 0 or not p.intersects(g):
                continue
            inter = p.intersection(g).area
            iou = inter / (p.area + g.area - inter)
            if iou >= thr:
                pairs.append((iou, i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j)
    return len(used_p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="BSTD annotation JSON")
    p.add_argument("--image_root", required=True)
    p.add_argument("--split", default="test", help='BSTD split to evaluate ("test" or "train")')
    p.add_argument("--keys", default=None, help="optional text file with one image key per line (e.g. a val list)")
    p.add_argument("--model_name", default=DEFAULT_MODEL, choices=list(model_info))
    p.add_argument("--model_checkpoint", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--save_json", default=None, help="save {image_key: {'detections', 'scores'}}")
    a = p.parse_args()

    data = json.load(open(a.gt, encoding="utf-8"))
    keys = [l.strip() for l in open(a.keys) if l.strip()] if a.keys else sorted(k for k, v in data.items() if v.get("split") == a.split)
    det = YOLO_detector(a.model_name, a.model_checkpoint, a.device, a.imgsz, a.conf)
    tp = n_pred = n_gt = 0
    saved = {}
    for i, k in enumerate(keys, 1):
        row = data[k]
        out = det.detect(os.path.join(a.image_root, row["image_name"]))
        anns = row.get("annotations", {})
        anns = anns.values() if isinstance(anns, dict) else anns
        gts = [g["coordinates"] for g in anns if len(g.get("coordinates") or []) >= 3]
        tp += match(out["detections"], gts, a.iou_thr)
        n_pred += len(out["detections"]); n_gt += len(gts)
        if a.save_json:
            saved[k] = out
        if i % 100 == 0 or i == len(keys):
            print(f"{i}/{len(keys)} images", flush=True)
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gt if n_gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print(json.dumps({"model": a.model_checkpoint or a.model_name, "images": len(keys), "gt_words": n_gt,
                      "predictions": n_pred, "true_positives": tp, "precision": round(P, 4),
                      "recall": round(R, 4), "f1": round(F, 4), "iou_thr": a.iou_thr}, indent=1))
    if a.save_json:
        json.dump(saved, open(a.save_json, "w"))


if __name__ == "__main__":
    main()
