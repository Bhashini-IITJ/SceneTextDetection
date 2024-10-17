

## Scene Text Detection - East
To get started create a virtual env and install the PyTorch version > 2.4.
### Installation
```commandline
conda create -n east_infer python=3.12
conda activate east_infer
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
cd SceneTextDetection/East/
pip install -r requirements.txt 
```

### Inference 

The script ```infer.py``` shall be used for inference. Get more details about using CLI ```python infer.py -h```.

Model checkpoints can also be accessed from github [assets](https://github.com/Bhashini-IITJ/SceneTextDetection/releases/tag/EAST).
```
python infer.py --image_path ../demo_images/image_90.jpg --model_checkpoint tmp/epoch_990_checkpoint.pth.tar
```

### Acknowledgement
EAST re-implemenation [repository](https://github.com/foamliu/EAST). 