import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from trainer import Trainer
from utils import get_config, unloader, get_model_list


def fid(real, fake, gpu):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    command = 'python -m pytorch_fid {} {}'.format(real, fake)
    os.system(command)


def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            # print(i, end='\r')
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    print(np.mean(res))


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default="results/flower_wavegan_base_index")
parser.add_argument('--dataset', type=str, default="flower")
parser.add_argument('--real_dir', type=str, default="results/flower_wavegan_base_index/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_wavegan_base_index/tests")
parser.add_argument('--ckpt', type=str, default="gen_00100000.pt")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=3)
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    real_dir = args.real_dir
    fake_dir = os.path.join(args.name, args.fake_dir)
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir, exist_ok=True)

    data = np.load(config['data_root'])
    if args.dataset == 'flower':
        data = data[85:]
        num = 10
    elif args.dataset == 'animal':
        data = data[119:]
        num = 10
    elif args.dataset == 'vggface':
        data = data[1802:]
        num = 30

    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, num:, :, :, :]

    if not os.path.exists(real_dir):
        os.makedirs(real_dir, exist_ok=True)
        for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
            for i in range(data_for_fid.shape[1]):
                idx = i
                real_img = data_for_fid[cls, idx, :, :, :]
                if args.dataset == 'vggface':
                    real_img *= 255
                real_img = Image.fromarray(np.uint8(real_img))
                real_img.save(os.path.join(real_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    if os.path.exists(fake_dir):
        trainer = Trainer(config)
        if args.ckpt:
            last_model_name = os.path.join(args.name, 'checkpoints', args.ckpt)
        else:
            last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
        trainer.load_ckpt(last_model_name)
        trainer.cuda()
        trainer.eval()
        for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
            for i in range(128):
                idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
                imgs = data_for_gen[cls, idx, :, :, :]
                imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
                fake_x = trainer.generate(imgs)
                output = unloader(fake_x[0].cpu())
                output.save(os.path.join(fake_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    fid(real_dir, fake_dir, args.gpu)
    LPIPS(fake_dir)
