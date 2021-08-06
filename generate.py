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
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV."
    )
    parser.add_argument(
        "--save_freq",
        "-s",
        type=int,
        default=500,
        help="Save an updated image every [x] number of epochs."
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
    args = parser.parse_args()
    args_dict = {
        "name": "",
        "prompts": [],
        "image_prompts": [],
        "noise_prompt_seeds": [],
        "noise_prompt_weights": [],
        "init_image": None,
        "init_weight": 0.,
        "clip_model": 'ViT-B/32',
        "vqgan_config": 'checkpoints/vqgan_imagenet_f16_1024.yaml',
        "vqgan_checkpoint": 'checkpoints/vqgan_imagenet_f16_1024.ckpt',
        "step_size": 0.05,
        "cutn": 64,
        "cut_pow": 1.,
    }
    for key, val in args_dict.items():
        vars(args)[key] = val

    main(args)
