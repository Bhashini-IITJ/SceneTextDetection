

## East Inference Setup 

### Clone Repository 

```
git clone https://github.com/Bhashini-IITJ/SceneTextDetection.git   
```

### Setup
```commandline
conda create -n east_infer python=3.12
conda activate east_infer
cd SceneTextDetection/East/
pip install -r requirements.txt 
mkdir results
```

### Inference 

```
python infer.py --image_path test/image.jpg --device cpu --model_checkpoint tmp/epoch_990_checkpoint.pth.tar
```

