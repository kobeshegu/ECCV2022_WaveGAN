import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import networks
from utils import get_model_list


class Trainer(nn.Module):
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.lr_dis = conf['lr_dis']
        self.lr_gen = conf['lr_gen']
        self.model = getattr(networks, conf['model'])(conf)
        if 'VGG' in conf['model']:
            for param in self.model.gen.encoder.parameters():
                param.requires_grad = False
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=conf['lr_dis'], weight_decay=conf['weight_decay'])
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=conf['lr_gen'], weight_decay=conf['weight_decay'])

        self.apply(weights_init(conf['init']))

    def gen_update(self, xs, y):
        self.gen_opt.zero_grad()
        losses = self.model(xs, y, 'gen_update')
        for item in losses.keys():
            self.__setattr__(item, losses[item])
        self.gen_opt.step()
        torch.cuda.empty_cache()

    def dis_update(self, xs, y):
        self.dis_opt.zero_grad()
        losses = self.model(xs, y, 'dis_update')
        for item in losses.keys():
            self.__setattr__(item, losses[item])
        self.dis_opt.step()
        torch.cuda.empty_cache()

    def resume(self, checkpoint_dir):

        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.model.gen.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.model.dis.load_state_dict(state_dict)

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus=False):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(this_model.gen.state_dict(), gen_name)
        torch.save(this_model.dis.state_dict(), dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def load_ckpt(self, ckpt_name):
        print('Load checkpoint')
        print("\tPath: %s" % ckpt_name)
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict)
        print('Load success')

    def generate(self, xs):
        return self.model.generate(xs)

    def update_lr(self, iterations, max_iter):
        if iterations > (max_iter // 2):
            self.gen_opt.param_groups[0]['lr'] -= (self.lr_gen / (max_iter // 2))
            self.dis_opt.param_groups[0]['lr'] -= (self.lr_dis / (max_iter // 2))


class CrossTrainer(nn.Module):
    def __init__(self, conf):
        super(CrossTrainer, self).__init__()
        self.lr_dis = conf['lr_dis']
        self.lr_gen = conf['lr_gen']
        self.model = getattr(networks, conf['model'])(conf)
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=conf['lr_dis'], weight_decay=conf['weight_decay'])
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=conf['lr_gen'], weight_decay=conf['weight_decay'])

        self.apply(weights_init(conf['init']))

    def gen_update(self, xs, ss, y):
        self.gen_opt.zero_grad()
        losses = self.model(xs, ss, y, 'gen_update')
        for item in losses.keys():
            self.__setattr__(item, losses[item])
        self.gen_opt.step()

    def dis_update(self, xs, ss, y):
        self.dis_opt.zero_grad()
        losses = self.model(xs, ss, y, 'dis_update')
        for item in losses.keys():
            self.__setattr__(item, losses[item])
        self.dis_opt.step()

    def resume(self, checkpoint_dir):

        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.model.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.model.dis.load_state_dict(state_dict['dis'])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        print('Resume from iteration %d' % iterations)

    def save(self, snapshot_dir, iterations, multigpus=False):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(this_model.gen.state_dict(), gen_name)
        torch.save(this_model.dis.state_dict(), dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def load_ckpt(self, ckpt_name):
        print('Load checkpoint')
        print("\tPath: %s" % ckpt_name)
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict)
        print('Load success')

    def generate(self, xs, ss):
        return self.model.generate(xs, ss)

    def update_lr(self, iterations, max_iter):
        if iterations > (max_iter // 2):
            self.gen_opt.param_groups[0]['lr'] -= (self.lr_gen / (max_iter // 2))
            self.dis_opt.param_groups[0]['lr'] -= (self.lr_dis / (max_iter // 2))


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun




