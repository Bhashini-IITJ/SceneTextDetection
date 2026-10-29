"""YOLO11n text detector fine-tuned on the Bharat Scene Text Dataset (BSTD).

Two checkpoints, both trained and run at imgsz=1920, both trained without any BSTD test image:

- ``yolo11n_bstd_1920`` (default): BSTD train split only. Best on the BSTD test set.
- ``yolo11n_bstd_mist_1920``: BSTD train split + every 2nd MIST training video frame.
  More general (much better on MIST); the detector used by IndicPhotoOCR 2.0.

The detector predicts axis-aligned word boxes, returned as 4-point polygons in the same
``{'detections': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...]}`` format as the EAST and
TextBPN++ detectors in this repository.
"""
import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RELEASE = "https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/YOLO"
model_info = {
    "yolo11n_bstd_1920": {
        "filename": "yolo11n_bstd_1920.pt",
        "sha256": "3b9216b50239a8d4f2ba39f1099f68ab0c4499661ca343acdbfc2e28a0c21fc3",
        "imgsz": 1920,
    },
    "yolo11n_bstd_mist_1920": {
        "filename": "yolo11n_bstd_mist_1920.pt",
        "sha256": "027951606ab3307653b532b8094eba38783723221d8683bd4ff2e11b8db49e2d",
        "imgsz": 1920,
    },
}
DEFAULT_MODEL = "yolo11n_bstd_1920"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint(model_name):
    """Return the local path of a released checkpoint, downloading and verifying it if needed."""
    info = model_info[model_name]
    path = WEIGHTS_DIR / info["filename"]
    if not path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"{RELEASE}/{info['filename']}"
        print(f"Downloading {url} ...")
        fd, tmp = tempfile.mkstemp(dir=WEIGHTS_DIR, suffix=".tmp")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if sha256(path) != info["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}; delete it and download again")
    return path


class YOLO_detector:
    def __init__(self, model_name=DEFAULT_MODEL, model_checkpoint=None, device="cuda:0",
                 imgsz=None, conf=0.25, iou=0.7, max_det=300):
        """
        :param model_name: one of model_info (used to download the checkpoint and pick imgsz).
        :param model_checkpoint: optional local .pt file; overrides the download.
        :param device: 'cuda', 'cuda:N' or 'cpu'.
        :param imgsz: inference size (long side); default = the size the model was trained at.
        :param conf: box confidence threshold. iou: NMS IoU threshold.
        """
        from ultralytics import YOLO
        if model_name not in model_info:
            raise ValueError(f"Unknown model_name '{model_name}'; choose from {list(model_info)}")
        self.checkpoint = model_checkpoint or ensure_checkpoint(model_name)
        self.model = YOLO(str(self.checkpoint))
        self.device = device
        self.imgsz = imgsz or model_info[model_name]["imgsz"]
        self.conf, self.iou, self.max_det = conf, iou, max_det

    def detect(self, image_path):
        """
        Perform text detection on the given image.
        :param image_path: Path to the input image.
        :return: {'detections': [4-point polygons], 'scores': [confidences]}.
        """
        if cv2.imread(image_path) is None:
            raise ValueError(f"Failed to read the image at {image_path}")
        result = self.model.predict(image_path, device=self.device, imgsz=self.imgsz, conf=self.conf,
                                    iou=self.iou, max_det=self.max_det, verbose=False)[0]
        bbox_result_dict = {"detections": [], "scores": []}
        if result.boxes is not None and len(result.boxes):
            for (x1, y1, x2, y2), s in zip(result.boxes.xyxy.cpu().numpy().tolist(),
                                           result.boxes.conf.cpu().numpy().tolist()):
                bbox_result_dict["detections"].append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                bbox_result_dict["scores"].append(float(s))
        return bbox_result_dict

    def visualize_detections(self, image_path, bbox_result_dict, output_path="output.png"):
        """Draw the detections on the image and save it to output_path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read the image at {image_path}")
        for bbox in bbox_result_dict.get("detections", []):
            cv2.polylines(image, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text detection using fine-tuned YOLO11n")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, choices=list(model_info),
                        help="Released checkpoint to use (downloaded to weights/ on first use)")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Optional local checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda:0", help='e.g. "cpu", "cuda" or "cuda:0"')
    parser.add_argument("--imgsz", type=int, default=None, help="Inference size; default = training size (1920)")
    parser.add_argument("--conf", type=float, default=0.25, help="Box confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save a visualization")
    args = parser.parse_args()

    detector = YOLO_detector(args.model_name, args.model_checkpoint, args.device, args.imgsz, args.conf)
    result = detector.detect(args.image_path)
    print(result)
    if args.output:
        detector.visualize_detections(args.image_path, result, args.output)
