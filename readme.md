## Official codes for WaveGAN: An Frequency-aware GAN for High-Fidelity Few-shot Image Generation (ECCV2022)
[[Paper](https://arxiv.org/abs/2207.07288)]

### Requirements
```
imageio==2.9.0
lmdb==1.2.1
opencv-python==4.5.3
pillow==8.3.2
scikit-image==0.17.2
scipy==1.5.4
tensorboard==2.7.0
tensorboardx==2.4
torch==1.7.0+cu110
torchvision==0.8.1+cu110
tqdm==4.62.3
```

### Please refer to ./networks/lofgan for core codes of our WaveGAN:

```python
def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
        
        class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError
      
```
## Notice
The code of lofgan_baseindex.py covers both WaveGAN-B and WaveGAN-M version of our method.
lofgan.py only contains the WaveGAN-M version. 

##  Training:

First, please download datasets from the repo of LofGAN and put them in the datasets folder.

`python train.py `

The results will be stored in  ./results/flower_wavegan_base_index

## Testing:

` python main_metric.py`

The generated images are stored in ./results/fakes

To quantitative evaluate the generated images,  pytorch-fid is required to be installed by `pip install pytorch-fid`.

If you use our WaveGAN-B version to train the model, remember to comment bellow codes in the `lofgan_baseindex.py' :
```
        LH1, HL1, HH1 = LH1.view(8, 3,c, h, w), HL1.view(8, 3,c, h, w), HH1.view(8, 3,c, h, w)
        LH1, HL1, HH1 = LH1[:,base_index,:,:,:], HL1[:,base_index,:,:,:], HH1[:,base_index,:,:,:]

        LH2, HL2, HH2 = LH2.view(8, 3, c, h, w), HL2.view(8, 3, c, h, w), HH2.view(8, 3, c, h, w)
        LH2, HL2, HH2 = LH2[:, base_index, :, :, :], HL2[:, base_index, :, :, :], HH2[:, base_index, :, :, :]

        LH3, HL3, HH3 = LH3.view(8, 3, c, h, w), HL3.view(8, 3, c, h, w), HH3.view(8, 3, c, h, w)
        LH3, HL3, HH3 = LH3[:, base_index, :, :, :], HL3[:, base_index, :, :, :], HH3[:, base_index, :, :, :]

        LH4, HL4, HH4 = LH4.view(8, 3, c, h, w), HL4.view(8, 3, c, h, w), HH4.view(8, 3, c, h, w)
        LH4, HL4, HH4 = LH4[:, base_index, :, :, :], HL4[:, base_index, :, :, :], HH4[:, base_index, :, :, :]
```

Feel free to contact kobeshegu@gmail.com if you have any question!
Our code is heavily based on [LoFGAN](https://github.com/edward3862/LoFGAN-pytorch), where you can download the datasets we used in this paper. We thanks a lot for their great work!


## Citation:
```
@inproceedings{Yang2022WaveGAN,
  title     = {WaveGAN: An Frequency-aware GAN for High-Fidelity Few-shot Image Generation},
  author    = {Mengping Yang, and Zhe Wang, and Ziqiu Chi, and Wenyi Feng},
  booktitle = {ECCV},
  year      = {2022}
}
```
