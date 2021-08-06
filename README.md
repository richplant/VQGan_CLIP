# VQGan_CLIP

Generate images from a list of prompts with VQGan and Clip locally. 

This repo is based off the original VQGan + CLIP (z+quantize method) Colab Notebook by Katherine Crowson. [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1L8oL-vLJXVcRzCFbPwOoMkPKJ8-aYdPN>
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

The only required parameter is the input CSV. Otherwise, other params are configurable as follows:

```
python generate.py -h
usage: generate.py [-h] [--save_freq SAVE_FREQ] [--max_epochs MAX_EPOCHS] [--seed SEED] [--size SIZE] 
                   [--vqgan_config {checkpoints/vqgan_imagenet_f16_1024.yaml,checkpoints/vqgan_imagenet_f16_16384.yaml}]
                   [--vqgan_checkpoint {checkpoints/vqgan_imagenet_f16_1024.ckpt,checkpoints/vqgan_imagenet_f16_16384.ckpt}] 
                   [--cutn CUTN] [--cut_pow CUT_POW] [--step_size STEP_SIZE] [--init_weight INIT_WEIGHT] 
                   [--clip_model {RN50,RN101,RN50x4,RN50x16,ViT-B/32,ViT-B/16}] [--init_image INIT_IMAGE] 
                   [--noise_prompt_seeds NOISE_PROMPT_SEEDS] [--noise_prompt_weights NOISE_PROMPT_WEIGHTS] 
                   [--image_prompts IMAGE_PROMPTS]
                   input

positional arguments:
  input                 Path to input CSV.

optional arguments:
  -h, --help            show this help message and exit
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
  --clip_model {RN50,RN101,RN50x4,RN50x16,ViT-B/32,ViT-B/16}
                        CLIP model to load.
  --init_image INIT_IMAGE
                        Path to initialization image.
  --noise_prompt_seeds NOISE_PROMPT_SEEDS
                        List of noise prompt seeds.
  --noise_prompt_weights NOISE_PROMPT_WEIGHTS
                        List of noise prompt weights.
  --image_prompts IMAGE_PROMPTS
                        List of image prompts.
```

# Generate

Run the following command to generate from a list of prompts (check sample data.csv for format):

```
python generate.py data.csv
```
