# offsite-tuning
The official implementation of Offsite-Tuning.

## Setup

```bash
conda create -n offsite python
conda activate offsite
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets evaluate wandb scikit-learn scipy timm
pip install lm-eval

python setup.py develop
```