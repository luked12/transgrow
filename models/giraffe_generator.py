'''
code reference: https://github.com/autonomousvision/giraffe
Several functions of this github were copied and only slightly modified
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import randn
from kornia.filters import filter2d
from math import log2

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

'''
code reference: https://github.com/autonomousvision/giraffe/blob/fc8d4503538fde8654785c90ec4d6191fa5d11e5/im2scene/layers.py
'''
# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

'''
code reference: https://github.com/autonomousvision/giraffe/blob/fc8d4503538fde8654785c90ec4d6191fa5d11e5/im2scene/layers.py
'''
class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout,
                                3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def actvn(self, x):
        out = F.leaky_relu(x, 2e-1)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

'''
code reference: https://github.com/autonomousvision/giraffe/blob/fc8d4503538fde8654785c90ec4d6191fa5d11e5/im2scene/layers.py
'''
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
    
'''    
code ref: https://github.com/autonomousvision/giraffe/blob/main/im2scene/gan2d/models/generator.py    
'''
class Generator(nn.Module):
    '''
    ACKNOWLEDGEMENT: This code is largely adopted from:
    https://github.com/LMescheder/GAN_stability
    '''
    def __init__(self, z_dim, size=64, nfilter=64,
                 nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(z.shape[0], z.shape[1])
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)
        out = self.conv_img(out)
        # out = self.conv_img(actvn(out))
        # out = torch.tanh(out)

        return out
    
'''
code reference: https://github.com/autonomousvision/giraffe/blob/main/im2scene/giraffe/models/neural_renderer.py
'''
class NeuralRenderer(nn.Module):
    ''' Neural renderer class
    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=32, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False,
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size) - 4)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        net = self.conv_in(x)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb    



class GiraffeGenBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None, channel_rgb=3):
        super().__init__()
        self.upsampler_nn = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampler_bl = nn.UpsamplingBilinear2d(scale_factor=2)
        channel_out = channel_out if channel_out else channel_in // 2
        self.enrich_conv = nn.Conv2d(channel_in, channel_out, 3, padding=1)
        self.activation = nn.LeakyReLU()
        self.cross_conv = nn.Conv2d(channel_out, channel_rgb, 3, padding=1)

    def forward(self, input_feat, input_rgb):
        output_feat = self.upsampler_nn(input_feat)
        output_rgb = self.upsampler_bl(input_rgb)
        output_feat = self.enrich_conv(output_feat)
        output_feat = self.activation(output_feat)

        cross = self.cross_conv(output_feat)
        output_rgb = output_rgb + cross

        return output_feat, output_rgb


class GiraffeGen1(nn.Module):
    '''
    intial point for upsampling is [dim,1,1]
    '''
    def __init__(self, channel_in, img_size, channel_rgb=3):
        super().__init__()
        self.img_size = img_size
        # intial ConvTranspose [1x1]->[4x4]
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_in * 2, 4),
            nn.BatchNorm2d(channel_in * 2),
            nn.GLU(dim = 1)
        )
        self.cross_conv = nn.Conv2d(channel_in, channel_rgb, 3, padding=1)
        self.block1 = GiraffeGenBlock(channel_in, channel_rgb=channel_rgb)
        self.block2 = GiraffeGenBlock(channel_in // 2, channel_rgb=channel_rgb)
        self.block3 = GiraffeGenBlock(channel_in // 4, channel_rgb=channel_rgb)
        self.block4 = GiraffeGenBlock(channel_in // 8, channel_rgb=channel_rgb)
        self.block5 = GiraffeGenBlock(channel_in // 16, channel_rgb=channel_rgb)
        self.block6 = GiraffeGenBlock(channel_in // 32, channel_rgb=channel_rgb)
        # self.final_activation = th.nn.Sigmoid()

    def forward(self, input):
        initial = self.initial_conv(input)
        output_rgb = self.cross_conv(initial)
        output_feat, output_rgb = self.block1(initial, output_rgb)
        output_feat, output_rgb = self.block2(output_feat, output_rgb)
        output_feat, output_rgb = self.block3(output_feat, output_rgb)
        output_feat, output_rgb = self.block4(output_feat, output_rgb)
        if self.img_size==128:
            output_feat, output_rgb = self.block5(output_feat, output_rgb)
        if self.img_size==256:
            output_feat, output_rgb = self.block5(output_feat, output_rgb)
            output_feat, output_rgb = self.block6(output_feat, output_rgb)
        return output_rgb
    
    
class GiraffeGen2(nn.Module):
    '''
    for large latent dimension >> 512:
    intial point for upsampling is [dim//4,2,2] and not [dim,1,1] !
    '''
    def __init__(self, channel_in, img_size, channel_rgb=3):
        super().__init__()
        self.img_size = img_size
        # intial ConvTranspose [1x1]->[4x4]
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_in * 2, 3),
            nn.BatchNorm2d(channel_in * 2),
            nn.GLU(dim = 1)
        )
        self.cross_conv = nn.Conv2d(channel_in, channel_rgb, 3, padding=1)
        self.block1 = GiraffeGenBlock(channel_in, channel_rgb=channel_rgb)
        self.block2 = GiraffeGenBlock(channel_in // 2, channel_rgb=channel_rgb)
        self.block3 = GiraffeGenBlock(channel_in // 4, channel_rgb=channel_rgb)
        self.block4 = GiraffeGenBlock(channel_in // 8, channel_rgb=channel_rgb)
        self.block5 = GiraffeGenBlock(channel_in // 16, channel_rgb=channel_rgb)
        self.block6 = GiraffeGenBlock(channel_in // 32, channel_rgb=channel_rgb)
        # self.final_activation = th.nn.Sigmoid()

    def forward(self, input):
        initial = self.initial_conv(input.view(input.shape[0],-1,2,2))
        output_rgb = self.cross_conv(initial)
        output_feat, output_rgb = self.block1(initial, output_rgb)
        output_feat, output_rgb = self.block2(output_feat, output_rgb)
        output_feat, output_rgb = self.block3(output_feat, output_rgb)
        output_feat, output_rgb = self.block4(output_feat, output_rgb)
        if self.img_size==128:
            output_feat, output_rgb = self.block5(output_feat, output_rgb)
        if self.img_size==256:
            output_feat, output_rgb = self.block5(output_feat, output_rgb)
            output_feat, output_rgb = self.block6(output_feat, output_rgb)
        return output_rgb    
