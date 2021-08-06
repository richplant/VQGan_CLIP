# VQGan_CLIP

Generate images from a list of prompts with VQGan and Clip locally. 

This repo is based off the original VQGan + CLIP (z+quantize method) Colab Notebook by Katherine Crowson. [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

# Install

Clone the repo, create a new virtual environment and install all prequisites.

```
git clone git@github.com:RyanGoslingsBugle/VQGan_CLIP.git
cd VQGan_CLIP
python -m venv <venv>
source <venv>/bin/activate
pip install -r requirements
```

# Configure

The only required parameter is the input CSV, set with "--input" or "-i".

Otherwise, other params are configurable as follows:

```
python generate.py -h
usage: generate.py [-h] --input INPUT [--save_freq SAVE_FREQ] [--max_epochs MAX_EPOCHS] [--seed SEED] [--size SIZE] [--vqgan_config {checkpoints/vqgan_imagenet_f16_1024.yaml,checkpoints/vqgan_imagenet_f16_16384.yaml}]
                   [--vqgan_checkpoint {checkpoints/vqgan_imagenet_f16_1024.ckpt,checkpoints/vqgan_imagenet_f16_16384.ckpt}] [--cutn CUTN] [--cut_pow CUT_POW] [--step_size STEP_SIZE] [--init_weight INIT_WEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to input CSV.
  --save_freq SAVE_FREQ, -s SAVE_FREQ
                        Save an updated image every SAVE_FREQ number of epochs.
  --max_epochs MAX_EPOCHS, -m MAX_EPOCHS
                        Maximum number of epochs to train for.
  --seed SEED           Seed for randomisation.
  --size SIZE           Output image size.
  --vqgan_config {checkpoints/vqgan_imagenet_f16_1024.yaml,checkpoints/vqgan_imagenet_f16_16384.yaml}
                        Pretrained VQGan config to load.
  --vqgan_checkpoint {checkpoints/vqgan_imagenet_f16_1024.ckpt,checkpoints/vqgan_imagenet_f16_16384.ckpt}
                        Pretrained VQGAn model to load.
  --cutn CUTN           Number of cuts to make.
  --cut_pow CUT_POW     Cut power.
  --step_size STEP_SIZE
                        Learning rate.
  --init_weight INIT_WEIGHT
                        Value to initialize weights to.

```

# Generate

Run the following command to generate from a list of prompts (check sample data.csv for format):

```
python generate.py -i data.csv
```
