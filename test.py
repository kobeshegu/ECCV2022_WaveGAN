import os
import argparse
import random
import shutil
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from trainer import Trainer
from utils import get_config, get_model_list, get_loaders, write_image

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default="results/vggface_lofganours")
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

_, test_dataloader = get_loaders(config)

if __name__ == '__main__':

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    out_dir = os.path.join(args.name, 'test')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    trainer = Trainer(config)
    last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()

    with torch.no_grad():
        for it, (imgs, _) in tqdm(enumerate(test_dataloader)):
            imgs = imgs.cuda()
            fake_xs = []
            for i in range(config['num_generate']):
                fake_xs.append(trainer.generate(imgs).unsqueeze(1))
            fake_xs = torch.cat(fake_xs, dim=1)

            write_image(it, out_dir, imgs, fake_xs, format='png')


