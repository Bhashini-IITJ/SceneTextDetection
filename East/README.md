

## Scene Text Detection - East

### Clone Repository 

```
git clone https://github.com/Bhashini-IITJ/SceneTextDetection.git   
```

### Installation
```commandline
conda create -n east_infer python=3.12
conda activate east_infer
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
cd SceneTextDetection/East/
pip install -r requirements.txt 
```

### Inference 

```
python infer.py --image_path test/image.jpg --device cpu --model_checkpoint tmp/epoch_990_checkpoint.pth.tar
```

