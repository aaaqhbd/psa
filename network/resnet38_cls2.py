import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]


    def forward(self, x,w,label):
        if self.f==1:
            x=self.forward1(x,w,label)
        if self.f==2:
            x = self.forward2(x,w,label)
        if self.f==3:
            x = self.forward3(x,w,label)
        if self.f==3:
            x = self.forward4(x,w,label)
        return x

    def init_w(self,i):
        # self.weight=torch.from_numpy(self.fc8.weight.detach().cpu().numpy()).cuda()
        # self.weight =torch.nn.Parameter(w)
        # self.weight.requires_grad=False
        # self.weight=self.fc8.weight
        # self.weight=w
        self.f=i
    def forward1(self, x,w,label):
        x = super().forward(x)
        # print(w.shape,self.fc8.weight.shape)
        k,c=w.size()
        w=w.view(k,c,1,1)
        # print(w[:,0])
        xcam = F.conv2d(x, w).detach()
        xcam = F.relu(xcam)
        # xcamm=torch.max(xcam,dim=[2,3],keepdim=True)
        xcamm =F.max_pool2d(
            xcam, kernel_size=(x.size(2), x.size(3)), padding=0)
        xcam=xcam/xcamm
        b, c = label.size()
        label = label.view(b, c, 1, 1)
        # print(xcam.shape,label.shape)
        xcam=xcam*label
        xcam=torch.sum(xcam,dim=[1],keepdim=True)
        xcam=torch.clamp(xcam,max=1)
        xcam=1-xcam*0.5

        x = self.dropout7(x)
        x = x * xcam
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        return x
    def forward3(self, x,w,label):
        x = super().forward(x)
        # print(w.shape,self.fc8.weight.shape)
        k,c=w.size()
        w=w.view(k,c,1,1)
        # print(w[:,0])
        xcam = F.conv2d(x, w).detach()
        xcam = F.relu(xcam)
        # xcamm=torch.max(xcam,dim=[2,3],keepdim=True)
        xcamm =F.max_pool2d(
            xcam, kernel_size=(x.size(2), x.size(3)), padding=0)
        xcam=xcam/xcamm
        b, c = label.size()
        label = label.view(b, c, 1, 1)
        # print(xcam.shape,label.shape)
        xcam=xcam*label
        xcam=torch.sum(xcam,dim=[1],keepdim=True)
        xcam=torch.clamp(xcam,max=1)
        xcam=1-xcam*0.5

        # x = self.dropout7(x)
        x = x * xcam
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        return x
    def forward2(self, x,w,label):
        x = super().forward(x)
        k, c = w.size()
        w = w.view(k, c, 1, 1)
        # print(w[:, 0])
        xcam = F.conv2d(x, w).detach()
        xcam = F.relu(xcam)
        xcamm = F.max_pool2d(
            xcam, kernel_size=(x.size(2), x.size(3)), padding=0)
        xcam=xcam/xcamm
        b,c=label.size()
        label=label.view(b,c,1,1)
        xcam=xcam*label
        xcam=torch.sum(xcam,dim=[1],keepdim=True)
        xcam=torch.clamp(xcam,max=1)
        xcam=1-(xcam>0.5).float()

        x = self.dropout7(x)
        x = x * xcam
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        return x
    def forward4(self, x,w,label):
        x = super().forward(x)
        k, c = w.size()
        w = w.view(k, c, 1, 1)
        # print(w[:, 0])
        xcam = F.conv2d(x, w).detach()
        xcam = F.relu(xcam)
        xcamm = F.max_pool2d(
            xcam, kernel_size=(x.size(2), x.size(3)), padding=0)
        xcam=xcam/xcamm
        b,c=label.size()
        label=label.view(b,c,1,1)
        xcam=xcam*label
        xcam=torch.sum(xcam,dim=[1],keepdim=True)
        xcam=torch.clamp(xcam,max=1)
        xcam=1-(xcam>0.5).float()

        # x = self.dropout7(x)
        x = x * xcam
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        return x
    def forward_cam(self, x):
        x = super().forward(x)

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