## Scene Text Detection - TextBPN++
To get started create a virtual env and install the PyTorch version > 2.4.
### Installation
```commandline
conda create -n textbpn python=3.12
conda activate textbpn
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

cd SceneTextDetection/TextBPNpp/
pip install opencv-python==4.10.0.82 
```

### Inference 

The script ```textbpnpp_detector.py``` shall be used for inference. Get more details about using CLI ```python textbpnpp_detector.py -h```.

Model checkpoints can also be accessed from github [assets](https://github.com/Bhashini-IITJ/SceneTextDetection/releases/tag/TextBPN%2B%2B).
```
python textbpnpp_detector.py --image_path ../demo_images/image_90.jpg  --model_name textbpn_mlt
```

### Acknowledgement
TextBPN++ re-implemenation [repository](https://github.com/GXYM/TextBPN-Plus-Plus). 