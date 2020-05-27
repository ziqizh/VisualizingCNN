import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

# from models.VGG16 import VGG16
from models.VGG16N import VGG16

from collections import OrderedDict

class Vgg16Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=1000):
        """
        Input
            number of class, default is 1k.
        """
        super(Vgg16Conv, self).__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_cls),
            nn.Softmax(dim=1)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        # vgg16_pretrained = models.vgg16()
        # vgg16_pretrained.classifier[0].in_features = 512*4*4
        # vgg16_pretrained.classifier[6].out_features = 8
        # checkpoint_path = "checkpoints/vgg_statedict.pt"
        checkpoint_path = "checkpoints/vgg-celeb-gender.epoch-7.checkpoint.pth.tar"
        vgg16_pretrained = VGG16(num_features=128*128, num_classes=2)
        vgg16_pretrained.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        # fine-tune Conv2d
        base = 0
        for idx, layer in enumerate(vgg16_pretrained.block_1):
            if isinstance(layer, nn.Conv2d):
                self.features[base + idx].weight.data = layer.weight.data
                self.features[base + idx].bias.data = layer.bias.data
        base = 5
        for idx, layer in enumerate(vgg16_pretrained.block_2):
            if isinstance(layer, nn.Conv2d):
                self.features[base + idx].weight.data = layer.weight.data
                self.features[base + idx].bias.data = layer.bias.data
        base = 10
        for idx, layer in enumerate(vgg16_pretrained.block_3):
            if isinstance(layer, nn.Conv2d):
                self.features[base + idx].weight.data = layer.weight.data
                self.features[base + idx].bias.data = layer.bias.data
        base = 19
        for idx, layer in enumerate(vgg16_pretrained.block_4):
            if isinstance(layer, nn.Conv2d):
                self.features[base + idx].weight.data = layer.weight.data
                self.features[base + idx].bias.data = layer.bias.data
        base = 28
        for idx, layer in enumerate(vgg16_pretrained.block_5):
            if isinstance(layer, nn.Conv2d):
                self.features[base + idx].weight.data = layer.weight.data
                self.features[base + idx].bias.data = layer.bias.data

        # fine-tune Linear
        # for idx, layer in enumerate(vgg16_pretrained.classifier):
        #     if isinstance(layer, nn.Linear):
        #         self.classifier[idx].weight.data = layer.weight.data
        #         self.classifier[idx].bias.data = layer.bias.data
        self.classifier[0].weight.data = vgg16_pretrained.fc1.weight.data
        self.classifier[0].bias.data =  vgg16_pretrained.fc1.bias.data

        self.classifier[2].weight.data = vgg16_pretrained.fc2.weight.data
        self.classifier[2].bias.data =  vgg16_pretrained.fc2.bias.data

        self.classifier[4].weight.data = vgg16_pretrained.fc3.weight.data
        self.classifier[4].bias.data =  vgg16_pretrained.fc3.bias.data

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)
        print(x.size())
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output

# if __name__ == '__main__':
#     model = models.vgg16(pretrained=True)
#     print(model)
