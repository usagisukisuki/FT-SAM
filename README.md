# FT-SAM
This repo is the implementation for Fine Turning Segment Anything Model.

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in FT-SAM.
1. Clone this repository.
   ```
   git clone https://github.com/usagisukisuki/FT-SAM.git
   cd FT-SAM
   ```
2. Install Pytorch and TorchVision. (you can follow the instructions here)
3. Install other dependencies.
   ```
   pip install -r requirements.txt
   ```

## Checkpoints
We use checkpoint of SAM in [vit_b](https://github.com/facebookresearch/segment-anything) version.
Additionally, we also use checkpoint of MobileSAM.
Please download from [SAM](https://github.com/facebookresearch/segment-anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and extract them under "models/Pretrained_model".
```
models
├── Pretrained_model
    ├── sam_vit_b_01ec64.pth
    ├── mobile_sam.pt
```

## Dataset
We can evaluate two biological segmentation datasets: ISBI2012 abd ssTEM.
## Training
### Binary segmentation (ISBI2012)
If you prepared the dataset, you can directly run the following code to train the model.

### Multi-class segmentation (ssTEM)
If you prepared the dataset, you can directly run the following code to train the model.


## Testing


