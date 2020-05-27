import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          # (1(32-1)- 32 + 3)/2 = 1
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU()
        )

        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU()
        )

        self.block_3 = nn.Sequential(
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU()
        )


        self.block_4 = nn.Sequential(
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU()
        )

        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU()
        )

        self.fc1 = nn.Linear(512*4*4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # self.classifier = nn.Sequential(
        #         nn.Linear(512*4*4, 4096),
        #         nn.ReLU(),
        #         nn.Linear(4096, 4096),
        #         nn.ReLU(),
        #         nn.Linear(4096, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()

    def latent_space(self, x, layer, pool=True):
        #1
        x = self.block_1(x)
        if layer == 1 and not pool:
            return x
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        if layer == 1 and pool:
            return x

        #2
        x = self.block_2(x)
        if layer == 2 and not pool:
            return x
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        if layer == 2 and pool:
            return x

        #3
        x = self.block_3(x)
        if layer == 3 and not pool:
            return x
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        if layer == 3 and pool:
            return x

        #4
        x = self.block_4(x)
        if layer == 4 and not pool:
            return x
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        if layer == 4 and pool:
            return x

        #5
        x = self.block_5(x)
        if layer == 5 and not pool:
            return x
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        if layer == 5 and pool:
            return x

        #6-fc
        x = F.relu(self.fc1(x.view(-1, 512*4*4)))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        #6-fc
        # logits = self.classifier(x.view(-1, 512*4*4))

        probas = F.log_softmax(logits, dim=1)
        return x

    def forward(self, x):
        latent_space_list = []
        x = self.block_1(x)
        latent_space_list.append(x) #0
        # print(f'1 shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        latent_space_list.append(x) #1
        # print(f'1p shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = self.block_2(x)
        latent_space_list.append(x) #2 524288
        # print(f'2 shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        latent_space_list.append(x) #3 131072
        # print(f'2p shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = self.block_3(x)
        latent_space_list.append(x) #4 262144
        # print(f'3 shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        latent_space_list.append(x) #5 65536
        # print(f'3p shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = self.block_4(x)
        latent_space_list.append(x) #6 131072
        # print(f'4 shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        latent_space_list.append(x) #7 32768
        # print(f'4p shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = self.block_5(x)
        latent_space_list.append(x) #8 32768
        # print(f'5 shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        latent_space_list.append(x) #9 8192
        # print(f'5p shape {x.shape} {x.shape[1] * x.shape[2] * x.shape[3]}')

        # logits = self.classifier(x.view(-1, 512*4*4))

        x = F.relu(self.fc1(x.view(-1, 512*4*4)))
        latent_space_list.append(x) #10

        x = F.relu(self.fc2(x))
        latent_space_list.append(x) #11

        logits = self.fc3(x)
        latent_space_list.append(x) #12

        probas = F.softmax(logits, dim=1)
        latent_space_list.append(F.softmax(logits, dim=1)) #13 2
        #print(f'9 shape {probas.shape}')
        # import sys
        # sys.exit()

        return logits, probas, latent_space_list

    def forward_1(self, x):
        x = self.block_1(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.block_2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.block_3(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.block_4(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.block_5(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        x = F.relu(self.fc1(x.view(-1, 512*4*4)))
        x = F.relu(self.fc2(x))

        logits = self.fc3(x)

        probas = F.log_softmax(logits, dim=1)
        #print(f'9 shape {probas.shape}')
        # import sys
        # sys.exit()

        return logits, probas, []