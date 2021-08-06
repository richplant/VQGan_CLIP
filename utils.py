import requests
from pathlib import Path
import io
from PIL import Image
from tqdm import tqdm
import logging
import csv
import errno
import os


class Utils:
    @staticmethod
    def load_prompts(filename):
        if Path(filename).exists():
            with open(filename, encoding='utf-8', mode='r') as f:
                reader = csv.DictReader(f)
                return {row['Title']: row['Text'] for row in reader}
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    @staticmethod
    def prompt_split(text):
        if len(text) > 64:
            words = text.split(" ")
            prompt = []
            current = words[0]
            for word in words[1:]:
                if len(current) + len(word) + 1 < 64:
                    current = f"{current} {word}"
                else:
                    prompt.append(current)
                    current = word
        else:
            prompt = [text]
        return prompt

    @staticmethod
    def download_file(url, filename):
        # https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
        with requests.get(url, stream=True) as r:
            total = int(r.headers.get('content-length', 0))
            logging.info(f"Downloading: {url} to {str(filename)}")
            with open(filename, 'wb') as f, tqdm(
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024) as bar:
                for data in r.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

    @staticmethod
    def download():
        chk_dir = Path("checkpoints/")
        chk_dir.mkdir(parents=True, exist_ok=True)
        if not chk_dir.joinpath("vqgan_imagenet_f16_1024.yaml").exists():
            Utils.download_file(
                'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
                chk_dir.joinpath('vqgan_imagenet_f16_1024.yaml'))
        if not chk_dir.joinpath("vqgan_imagenet_f16_1024.ckpt").exists():
            Utils.download_file(
                'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
                chk_dir.joinpath('vqgan_imagenet_f16_1024.ckpt'))
        if not chk_dir.joinpath("vqgan_imagenet_f16_16384.yaml").exists():
            Utils.download_file(
                'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
                chk_dir.joinpath('vqgan_imagenet_f16_16384.yaml'))
        if not chk_dir.joinpath("vqgan_imagenet_f16_16384.ckpt").exists():
            Utils.download_file('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast'
                                '.ckpt&dl=1',
                                chk_dir.joinpath('vqgan_imagenet_f16_16384.ckpt'))

    @staticmethod
    def fetch(url_or_path):
        if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
            r = requests.get(url_or_path)
            r.raise_for_status()
            fd = io.BytesIO()
            fd.write(r.content)
            fd.seek(0)
            return fd
        return open(url_or_path, 'rb')

    @staticmethod
    def parse_prompt(prompt):
        if prompt.startswith('http://') or prompt.startswith('https://'):
            vals = prompt.rsplit(':', 3)
            vals = [vals[0] + ':' + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    @staticmethod
    def resize_image(image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
        return image.resize(size, Image.LANCZOS)
