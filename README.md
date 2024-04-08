# GroundingDINO-SAM

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
```
export CUDA_HOME=/path/to/cuda-11.3
```

```bash
conda create -n grounding python=3.10
conda activate grounding
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install GroundingDINO
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -e .
```
Install Segment Anything
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

**NOTE**: To glue all the elements of our demo together we will use the `supervision` pip package, which will help us process, filter and visualize our detections as well as to save our dataset. A lower version of the supervision was installed with Grounding DINO. However, in this demo we need the functionality introduced in the new versions. Therefore, we uninstall the current supervsion version and install version `0.6.0`.
```bash
pip install supervision==0.6.0
```

Install [Roboflow](https://roboflow.com/)
```bash
pip install -q roboflow
```

## <a name="GettingStarted"></a>Getting Started
First download SAM and GroundingDINO [model checkpoints](#model-checkpoints). 

The models can also be efficiently downloaded from the command line.
```
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
## Demo

```
python demo.py
```

<p float="left">
  <img src="data/dog-3.jpeg?raw=true" width="49.1%" />
  <img src="dog-3.png?raw=true" width="48.9%" />
</p>



## <a name="Models"></a>Model Checkpoints

### GroundingDINO model
- `GroundingDINO-T`: [Swin-T.](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- `GroundingDINO-B`: [Swin-B.](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth)
### Segment Anything model
Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Acknowledgements

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)

## Supplementary Instruction
If you also encounter this error：
`We couldn't connect to 'https://huggingface.co' to load this file`，
You can avoid this problem by doing the following：
```
pip install -U huggingface_hub
# Linux
export HF_ENDPOINT=https://hf-mirror.com
# Windows Powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download --resume-download bert-base-uncased	

```
