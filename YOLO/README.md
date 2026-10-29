
## Scene Text Detection - YOLO
YOLO11n (Ultralytics) fine-tuned for word-level text detection on BSTD, trained and run at 1920 px.
To get started create a virtual env and install the PyTorch version > 2.4.
### Installation
```commandline
conda create -n yolo_text python=3.10
conda activate yolo_text
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
cd SceneTextDetection/YOLO/
pip install -r requirements.txt
```

### Models
Two checkpoints are provided. Neither has seen any BSTD test image.

| `--model_name` | Training data | Use when |
|---|---|---|
| `yolo11n_bstd_1920` (default) | BSTD train split | best numbers on BSTD |
| `yolo11n_bstd_mist_1920` | BSTD train split + every 2nd MIST training frame | more general real-world/video frames; detector of IndicPhotoOCR 2.0 |

Model checkpoints can also be accessed from github [assets](https://github.com/Bhashini-IITJ/SceneTextDetection/releases/tag/YOLO). They are downloaded to `weights/` (and SHA-256 verified) on first use.

### Inference

The script ```yolo_detector.py``` shall be used for inference. Get more details about using CLI ```python yolo_detector.py -h```.
```
python yolo_detector.py --image_path ../demo_images/image_90.jpg
python yolo_detector.py --image_path ../demo_images/image_90.jpg --model_name yolo11n_bstd_mist_1920 --device cpu --output out.jpg
```
Output format is the same as the other detectors in this repository: `{'detections': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...], 'scores': [...]}`.

From Python:
```python
from yolo_detector import YOLO_detector
detector = YOLO_detector("yolo11n_bstd_1920", device="cuda:0")
result = detector.detect("../demo_images/image_90.jpg")
```

### Training
1. Build the YOLO dataset from the BSTD annotation JSON. This uses only the BSTD train split, held-out 10% stratified by folder (seed 20260930, the split of the released models):
```
python prepare_data.py --gt <BSTD>/BSTD_v17.57.json --image_root <BSTD> --out data/bstd
# BSTD + MIST model: add every 2nd MIST training frame
python prepare_data.py --gt <BSTD>/BSTD_v17.57.json --image_root <BSTD> --mist_zip <MIST>/mist.zip --out data/bstd_mist
```
2. Train with the recipe of the released checkpoints (1920 px, SGD lr0 1e-3 cosine, 60 epochs, patience 12, batch 8, no flips, AMP off, seed 20260930). The best epoch is chosen by validation fitness:
```
python train.py --data data/bstd/data.yaml --init_weights <yolo11n text checkpoint or yolo11n.pt> --name bstd_1920
```
The released models were fine-tuned from a YOLO11n checkpoint pre-trained for text detection at 640 px. Starting from `yolo11n.pt` (COCO) also works, but will not reproduce the released numbers exactly.

### Testing
```
python test.py --gt <BSTD>/BSTD_v17.57.json --image_root <BSTD> --split test
python test.py --gt <BSTD>/BSTD_v17.57.json --image_root <BSTD> --split test --model_name yolo11n_bstd_mist_1920 --save_json preds.json
```
Reports detection precision, recall and F1 (polygon IoU ≥ 0.5, one-to-one matching), and can save the predictions per image.

### Acknowledgement
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
