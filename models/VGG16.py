import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(torch.nn.Module):	

    def __init__(self, num_classes):	
        super(VGG16, self).__init__()	

        # calculate same padding:	
        # (w - k + 2*p)/s + 1 = o	
        # => p = (s(o-1) - w + k)/2	

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
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
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
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(	
                nn.Linear(512 * 4 * 4, 4096),	
                nn.ReLU(),   	
                nn.Linear(4096, 4096),	
                nn.ReLU(),	
                nn.Linear(4096, num_classes)	
        )	

        for idx, layer in enumerate(self.features):	
            if isinstance(layer, torch.nn.Conv2d):	
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels	
                #m.weight.data.normal_(0, np.sqrt(2. / n))	
                layer.weight.detach().normal_(0, 0.05)	
                if layer.bias is not None:	
                    layer.bias.detach().zero_()	
            elif isinstance(layer, torch.nn.Linear):	
                layer.weight.detach().normal_(0, 0.05)	
                layer.bias.detach().detach().zero_()

    def forward(self, x):	
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)
        
        # reshape to (1, 512 * 4 * 4)
        x = x.view(x.size()[0], -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)	

        return logits, probas