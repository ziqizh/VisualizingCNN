import torch
import torch.nn as nn
import torchvision.models as models

from models.VGG16N import VGG16

import sys

class Vgg16Deconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """
    def __init__(self):
        super(Vgg16Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),

            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            
            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            
            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            
            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1)
        )

        self.conv2deconv_indices = {
                0:36, 2:34, 5:31, 7:29,
                10:26, 12:24, 14:22, 16:20, 
                19:17, 21:15, 23:13, 25:11,
                28:8, 30:6, 32:4, 34:2
                }

        self.unpool2pool_indices = {
                32:4, 27:9, 18:18, 9:27, 0:36
                }

        self.init_weight()

    def init_weight(self):

        # vgg16_pretrained = models.vgg16(pretrained=True)
        # checkpoint_path = "/home/ziqizh/code/adversarial-learning-research/interpretebility/checkpoints/vgg_statedict.pt"
        # checkpoint_path = "checkpoints/vgg_statedict.pt"
        checkpoint_path = "checkpoints/vgg-celeb-gender.epoch-7.checkpoint.pth.tar"
        # vgg16_pretrained = VGG16(num_classes=8)
        # vgg16_pretrained.load_state_dict(torch.load(checkpoint_path))
        # checkpoint_path = "/home/ziqizh/code/expGAN/pytorch_GAN_zoo/checkpoints/vgg_statedict.pt"
        vgg16_pretrained = VGG16(num_features=128*128, num_classes=2)
        vgg16_pretrained.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        base = 0
        for idx, layer in enumerate(vgg16_pretrained.block_1):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[base + idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[base + idx]].bias.data = layer.bias.data

        base = 5
        for idx, layer in enumerate(vgg16_pretrained.block_2):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[base + idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[base + idx]].bias.data = layer.bias.data

        base = 10
        for idx, layer in enumerate(vgg16_pretrained.block_3):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[base + idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[base + idx]].bias.data = layer.bias.data

        base = 19
        for idx, layer in enumerate(vgg16_pretrained.block_4):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[base + idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[base + idx]].bias.data = layer.bias.data

        base = 28
        for idx, layer in enumerate(vgg16_pretrained.block_5):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[base + idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[base + idx]].bias.data = layer.bias.data
        
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                # if isinstance(self.features[idx], nn.ConvTranspose2d):
                #     print(x)
                #     print(self.features[idx].bias.shape)
                x = self.features[idx](x)
        return x
