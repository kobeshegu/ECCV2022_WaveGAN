import os
import random
import shutil
import sys
import argparse
import torch
import time
import numpy as np
from tensorboardX import SummaryWriter

from trainer import Trainer
from utils import make_result_folders, write_image, write_loss, get_config, get_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str,default="./configs/flower_lofgan.yaml")
parser.add_argument('--output_dir', type=str,default="./results/flower_wavegan_base_index")
parser.add_argument('-r', "--resume", action="store_true")
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

config = get_config(args.conf)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

output_directory = args.output_dir
remove_first = not args.resume
checkpoint_directory, image_directory, log_directory = make_result_folders(output_directory, remove_first=remove_first)
shutil.copy(args.conf, os.path.join(output_directory, 'configs.yaml'))
train_writer = SummaryWriter(log_directory)
max_iter = config['max_iter']

train_dataloader, test_dataloader = get_loaders(config)

if __name__ == '__main__':

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    start = time.time()

    trainer = Trainer(config)
    trainer.cuda()

    imgs_test, _ = iter(test_dataloader).next()
    iterations = trainer.resume(checkpoint_directory) if args.resume else 0
    while True:
        with torch.autograd.set_detect_anomaly(True):

            for it, (imgs, label) in enumerate(train_dataloader):
                trainer.update_lr(iterations, max_iter)
                imgs = imgs.cuda()
                label = label.cuda()
                trainer.dis_update(imgs, label)
                trainer.gen_update(imgs, label)

                if (iterations + 1) % config['snapshot_log_iter'] == 0:
                    end = time.time()
                    print("Iteration: [%06d/%06d], time: %d, loss_adv_dis: %04f, loss_adv_gen: %04f"
                          % (iterations + 1, max_iter, end-start, trainer.loss_adv_dis, trainer.loss_adv_gen))
                    write_loss(iterations, trainer, train_writer)

                if (iterations + 1) % config['snapshot_val_iter'] == 0:
                    with torch.no_grad():
                        imgs_test = imgs_test.cuda()
                        fake_xs = []
                        for i in range(config['num_generate']):
                            fake_xs.append(trainer.generate(imgs_test).unsqueeze(1))
                        fake_xs = torch.cat(fake_xs, dim=1)
                        write_image(iterations, image_directory, imgs_test.detach(), fake_xs.detach())

                if (iterations + 1) % config['snapshot_save_iter'] == 0:
                    trainer.save(checkpoint_directory, iterations)
                    print('Saved model at iteration %d' % (iterations + 1))

                iterations += 1
                if iterations >= max_iter:
                    print("Finish Training")
                    sys.exit(0)

