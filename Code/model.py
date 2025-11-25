# models.py
import torch
import torch.nn as nn

# ---------------------------
# Utils
# ---------------------------
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

# ---------------------------
# Resnet Generator (CycleGAN style)
# ---------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        # downsample
        mult = 1
        cur = ngf
        for i in range(2):
            model += [nn.Conv2d(cur, cur*2, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(cur*2),
                      nn.ReLU(True)]
            cur *= 2
        # resblocks
        for i in range(n_blocks):
            model += [ResnetBlock(cur)]
        # upsample
        for i in range(2):
            model += [nn.ConvTranspose2d(cur, cur//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(cur//2),
                      nn.ReLU(True)]
            cur //= 2
        model += [nn.Conv2d(cur, output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# PatchGAN Discriminator
# ---------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Replay Buffer (for discriminator stability)
# ---------------------------
import random
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        returned = []
        for img in images:
            img = torch.unsqueeze(img.data, 0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                returned.append(img)
            else:
                if random.uniform(0,1) > 0.5:
                    idx = random.randint(0, self.max_size-1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = img
                    returned.append(tmp)
                else:
                    returned.append(img)
        return torch.cat(returned, dim=0)
