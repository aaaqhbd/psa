import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()
        self.dim=[512]
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(512, 20, 1, bias=False)
        self.fc8sub = nn.Conv2d(512, 20*10, 1, bias=False)
        self.conv8=nn.Conv2d(4096, 512, 3, padding=1,  bias=False)
        # torch.nn.init.normal_(self.fc8.weight, mean=0, std=0.01)
        # torch.nn.init.normal_(self.conv8.weight, mean=0, std=0.01)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc8sub.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8sub]


    def forward(self, x):
        x = super().forward(x)
        x = self.conv8(x)
        x = self.dropout7(x)

        xf = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(xf)
        x = x.view(x.size(0), -1)
        xsub = self.fc8sub(xf)
        xsub = xsub.view(x.size(0), -1)

        return xf,x,xsub

    def forward_cam(self, x):
        x = super().forward(x)
        x = self.conv8(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups