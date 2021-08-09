from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from CLIP import clip
from torch import optim
from torchvision import transforms
from utils import Utils
from helpers import Prompt, Ops, MakeCutouts, ClampWithGrad
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from pathlib import Path
from tqdm import tqdm


class VQGanCLIP:
    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.args = args
        self.model = self.load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(self.device)
        self.perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)

        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        self.sideX, self.sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if args.init_image:
            pil_image = Image.open(Utils.fetch(args.init_image)).convert('RGB')
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=args.step_size)

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []

        for prompt in self.args.prompts:
            txt, weight, stop = Utils.parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for prompt in self.args.image_prompts:
            path, weight, stop = Utils.parse_prompt(prompt)
            img = Utils.resize_image(Image.open(Utils.fetch(path)).convert('RGB'), (self.sideX, self.sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(self.device))

    def load_vqgan_model(self, config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del model.loss
        return model

    def synth(self):
        z_q = Ops.vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return ClampWithGrad.apply(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses):
        name = self.args.name.lower().replace("'", "").replace(" ", "_")
        out_dir = Path(self.args.output).joinpath(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        name_str = f"{out_dir}/progress_{i}.png"
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        out = self.synth()
        TF.to_pil_image(out[0].cpu()).save(name_str)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')

    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def train(self, i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        if i % self.args.save_freq == 0:
            self.checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
