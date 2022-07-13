import torch
import torch.utils.data as data
import numpy as np

from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


def single_channel_loader(path):
    return Image.open(path)


def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list


class FSGDataset(data.Dataset):
    def __init__(self, root, mode, num_for_seen, n_sample, transform=None):
        assert mode in ['train', 'test']
        data = np.load(root)
        num_cls_total = data.shape[0]
        if mode == 'train':
            self.data = data[:num_for_seen]
            num_cls = num_for_seen
        else:
            self.data = data[num_for_seen:]
            num_cls = num_cls_total - num_for_seen
        self.n_smaple = n_sample
        self.transform = transform

        self.n_class = self.data.shape[0]
        self.img_per_class = self.data.shape[1]
        self.length = self.n_class * self.img_per_class

        print('load data from:', root)
        print('mode:', mode)
        print('num of class:', num_cls)
        print('data per class', self.img_per_class)

    def __getitem__(self, item):
        np.random.seed(item)
        cls = np.random.choice(self.n_class, 1)
        idx = np.random.choice(self.img_per_class, self.n_smaple, replace=False)
        x = self.data[cls, idx, :, :, :]
        if self.transform is not None:
            x = torch.cat([self.transform(img).unsqueeze(0) for img in x], dim=0)
        return x, cls

    def __len__(self):
        return self.length