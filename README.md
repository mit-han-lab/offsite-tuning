# offsite-tuning
The official implementation of Offsite-Tuning.

## Abstract

Foundation models have become increasingly popular due to their general performance on downstream tasks.
Although some foundation models can make zero-shot predictions, transfer learning (e.g. fine-tuning) is still important for adaption to specific downstream tasks.
However, many foundation models are proprietary, so users must share their data with model owners to fine-tune the models, which is costly and raise privacy concerns. Moreover, fine-tuning foundation models is computation-intensive and impractical for most downstream users.
In this paper, we propose Offsite-Tuning, a privacy-preserving and efficient transfer learning framework that can adapt foundation models to downstream data without full model access.
In offsite-tuning, the model owner sends a light-weight adapter and a lossy compressed emulator to the data owner, who then fine-tunes the adapter on the downstream data with the emulator's assistance.
The fine-tuned adapter is then returned to the model owner, who plugs it into the full model to create an adapted foundation model for users. 
Offsite-tuning preserves both parties' privacy and is more computationally efficient than existing fine-tuning methods that require full model weights.
We demonstrate the effectiveness of offsite-tuning on various large language and vision foundation models.
We show that offsite-tuned foundation models can achieve comparable accuracy as full model fine-tuning while being privacy-preserving and efficient, with a 6.5x speedup and a 5.6x memory reduction.

## Setup

```bash
conda create -n offsite python
conda activate offsite
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets evaluate wandb scikit-learn scipy timm
pip install lm-eval

python setup.py develop
```