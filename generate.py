import argparse
from tqdm import tqdm
from model import VQGanCLIP
from utils import Utils
import logging


def run(args):
    model = VQGanCLIP(args)
    i = 0
    try:
        with tqdm() as pbar:
            while i < args.max_epochs:
                model.train(i)
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass


def main(args):
    Utils.download()
    prompts = Utils.load_prompts(args.input)
    for title, prompt in prompts.items():
        prompt = Utils.prompt_split(prompt)
        args.prompts = prompt
        args.name = title
        run(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Path to input CSV."
    )
    parser.add_argument(
        "--save_freq",
        "-s",
        type=int,
        default=500,
        help="Save an updated image every SAVE_FREQ number of epochs."
    )
    parser.add_argument(
        "--max_epochs",
        "-m",
        type=int,
        default=5_000,
        help="Maximum number of epochs to train for."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for randomisation."
    )
    parser.add_argument(
        "--size",
        type=list,
        default=[256, 340],
        help="Output image size."
    )
    parser.add_argument(
        "--vqgan_config",
        choices=["checkpoints/vqgan_imagenet_f16_1024.yaml", "checkpoints/vqgan_imagenet_f16_16384.yaml"],
        type=str,
        default="checkpoints/vqgan_imagenet_f16_1024.yaml",
        help="Pretrained VQGan config to load."
    )
    parser.add_argument(
        "--vqgan_checkpoint",
        choices=["checkpoints/vqgan_imagenet_f16_1024.ckpt", "checkpoints/vqgan_imagenet_f16_16384.ckpt"],
        type=str,
        default="checkpoints/vqgan_imagenet_f16_1024.ckpt",
        help="Pretrained VQGAn model to load."
    )
    parser.add_argument(
        "--cutn",
        type=int,
        default=64,
        help="Number of cuts to make."
    )
    parser.add_argument(
        "--cut_pow",
        type=float,
        default=1.0,
        help="Cut power."
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.05,
        help="Learning rate."
    )
    parser.add_argument(
        "--init_weight",
        type=float,
        default=0.0,
        help="Value to initialize weights to."
    )
    args = parser.parse_args()
    args_dict = {
        "name": "",
        "prompts": [],
        "image_prompts": [],
        "noise_prompt_seeds": [],
        "noise_prompt_weights": [],
        "init_image": None,
        "clip_model": 'ViT-B/32',
    }
    for key, val in args_dict.items():
        vars(args)[key] = val

    main(args)
