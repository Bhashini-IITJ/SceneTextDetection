"""Fine-tune YOLO11n for text detection with the exact recipe of the released checkpoints.

Both released models were trained with these settings; only --data differs
(BSTD only, or BSTD + every 2nd MIST training frame; see prepare_data.py).

Example:
  python train.py --data data/bstd/data.yaml --init_weights weights/yolo11n-text-init.pt --name bstd_1920
"""
import argparse

RECIPE = dict(
    epochs=60, patience=12, batch=8, imgsz=1920, optimizer="SGD", lr0=0.001, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3, cos_lr=True, close_mosaic=10, amp=False, seed=20260930,
    deterministic=True, cache=False, workers=4, save_period=1,
    # augmentation (no flips: text is not mirror-symmetric)
    degrees=3.0, translate=0.1, scale=0.35, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0,
    mosaic=0.5, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml written by prepare_data.py")
    p.add_argument("--init_weights", default="yolo11n.pt",
                   help="starting checkpoint. The released models started from a YOLO11n text-detection "
                        "checkpoint (see README); 'yolo11n.pt' (COCO) also works but will not reproduce them exactly")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo11n_text_1920")
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    p.add_argument("--batch", type=int, default=RECIPE["batch"])
    p.add_argument("--imgsz", type=int, default=RECIPE["imgsz"])
    a = p.parse_args()

    from ultralytics import YOLO
    cfg = dict(RECIPE, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz)
    model = YOLO(a.init_weights)
    model.train(data=a.data, project=a.project, name=a.name, device=a.device, **cfg)
    print(f"best checkpoint: {a.project}/{a.name}/weights/best.pt (selected by validation fitness)")


if __name__ == "__main__":
    main()
