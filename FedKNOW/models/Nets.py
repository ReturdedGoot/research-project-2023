# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import models
from FedKNOW.models.ResNet import resnet18,wide_resnet50_2,resnext50_32x4d,resnet152
from FedKNOW.models.ResNetWEIT import resnetWEIT18
import json
import numpy as np
from FedKNOW.models.mobilenet import mobilenet_v2
from FedKNOW.models.language_utils import get_word_emb_arr
from FedKNOW.models.layer import DecomposedConv,DecomposedLinear
from FedKNOW.models.inception_v3 import inception_v3
from FedKNOW.models.shufflenetv2 import shufflenet_v2_x0_5
from FedKNOW.models.Densenet import DenseNet


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
class RepTailSENet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = resnet18()
        pre_dict = torch.load('../pre_train/resnet18.pth')
        pre_dict.pop('fc.weight')
        pre_dict.pop('fc.bias')
        state_dict = self.feature_net.state_dict()
        model_dict = {k:v for k,v in pre_dict.items() if k in state_dict.keys()}
        state_dict.update(model_dict)
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(512, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class RepTailDensnet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = DenseNet()
        self.last = torch.nn.Linear(512, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'last' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class RepTailResNet18(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = resnet18()
        state_dict = torch.load('../pre_train/resnet18.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net.outlen, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class RepTailResNet152(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = resnet152()
        state_dict = torch.load('../pre_train/resnet152.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net.outlen, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class RepTailResNext(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = resnext50_32x4d()
        state_dict = torch.load('pre_train/resnext50_32x4d-7cdf4587.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net.outlen, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class RepTailMobilenet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = mobilenet_v2()
        state_dict = torch.load('pre_train/mobilenet_v2-b0353104.pth')
        state_dict.pop('classifier.1.weight')
        state_dict.pop('classifier.1.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net.last_channel, output)
        self.weight_keys = []
        self.n_outputs = output
        self.nc_per_task = nc_per_task
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class RepTailWideResNet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = wide_resnet50_2()
        state_dict = torch.load('pre_train/wide_resnet50_2-95faca4d.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net.outlen, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class RepTailshufflenet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = shufflenet_v2_x0_5()
        state_dict = torch.load('pre_train/shufflenetv2_x0.5-f707e7126e.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(self.feature_net._stage_out_channels[-1], output)
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        self.weight_keys = []
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class RepTailInception_v3(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = inception_v3()
        state_dict = torch.load('pre_train/inception_v3_google-1a9a5a14.pth')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.feature_net.load_state_dict(state_dict)
        self.last = torch.nn.Linear(2048, output)
        self.weight_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x,t,pre=False,is_con=True):
        if self.training:
            h,_ = self.feature_net(x)
        else:
            h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class RepTail(nn.Module):
    def __init__(self,inputsize,output=100,nc_per_task = 10):
        super().__init__()
        self.feature_net = Cifar100Net(inputsize)
        #self.feature_net = Cifar100WEIT(inputsize,output,nc_per_task)
        self.last = nn.Linear(1024, output)
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        self.weight_keys = [['feature_net.conv1.weight'], ['feature_net.conv1.bias'], ['feature_net.conv2.weight'], ['feature_net.conv2.bias'],
         ['feature_net.conv3.weight'], ['feature_net.conv3.bias'], ['feature_net.conv4.weight'], ['feature_net.conv4.bias'],
         ['feature_net.conv5.weight'], ['feature_net.conv5.bias'], ['feature_net.conv6.weight'], ['feature_net.conv6.bias'],
         ['feature_net.fc1.weight'], ['feature_net.fc1.bias'], ['last.weight'], ['last.bias']]

    def forward(self,x,t,pre=False,is_con=True,avg_act=False):
        h = self.feature_net(x,avg_act)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class Cifar100Net(nn.Module):
    def __init__(self, inputsize):
        super().__init__()

        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(size, 3, padding=1)  # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = nn.Linear(s*s*128, 1024)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
    def forward(self, x, avg_act=False):
        if x.size(1) !=3:
            bsz = x.size(0)
            x = x.view(bsz,3,32,32)
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        # h = self.fc2(h)
        self.grads = {}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
                return hook
        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))
        return h

class Cifar100WEIT(nn.Module):
    def __init__(self, inputsize,n_ouputs=100,nc_per_task=5, window = "sliding"):
        super().__init__()
        self.nc_per_task = nc_per_task
        self.n_outputs = n_ouputs
        self.window = window
        ncha, size, _ = inputsize
        self.conv1 = DecomposedConv(ncha, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(size, 3, padding=1)  # 32
        self.conv2 = DecomposedConv(32, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = DecomposedConv(32, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = DecomposedConv(64, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = DecomposedConv(64, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = DecomposedConv(128, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = DecomposedLinear(s * s * 128, 256)  # 2048
        self.last = nn.Linear(256, self.n_outputs)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
        self.layer_keys = [['conv1'],['conv2'],['conv3'],['conv4'],['conv5'],['conv6'],['fc1'],['last']]
    def set_sw(self, glob_weights):
        self.conv1.sw = Parameter(glob_weights[0])
        self.conv2.sw = Parameter(glob_weights[1])
        self.conv3.sw = Parameter(glob_weights[2])
        self.conv4.sw = Parameter(glob_weights[3])
        self.conv5.sw = Parameter(glob_weights[4])
        self.conv6.sw = Parameter(glob_weights[5])
        self.fc1.sw = glob_weights[6]
    def set_knowledge(self,t,from_kbs):
        self.conv1.set_atten(t,from_kbs[0].size(-1))
        self.conv1.set_knlwledge(from_kbs[0])
        self.conv2.set_atten(t, from_kbs[1].size(-1))
        self.conv2.set_knlwledge(from_kbs[1])
        self.conv3.set_atten(t, from_kbs[2].size(-1))
        self.conv3.set_knlwledge(from_kbs[2])
        self.conv4.set_atten(t, from_kbs[3].size(-1))
        self.conv4.set_knlwledge(from_kbs[3])
        self.conv5.set_atten(t, from_kbs[4].size(-1))
        self.conv5.set_knlwledge(from_kbs[4])
        self.conv6.set_atten(t, from_kbs[5].size(-1))
        self.conv6.set_knlwledge(from_kbs[5])
        self.fc1.set_atten(t, from_kbs[6].size(-1))
        self.fc1.set_knlwledge(from_kbs[6])
    def get_weights(self):
        weights = []
        w = self.conv1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)
        w = self.conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        self.conv3.get_weight().detach()
        w = self.conv3.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.conv4.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.conv5.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.conv6.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.fc1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        return weights
    def forward(self, x, t,avg_act=False):
        if x.size(1) !=3:
            bsz = x.size(0)
            x = x.view(bsz,3,32,32)
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        output = self.last(h)
        # make sure we predict classes within the current task
        offset1 = 0 if self.window != "sliding" else int(t * self.nc_per_task)
        # offset1 = int(t * self.nc_per_task)
        offset2 = self.n_outputs if self.window == "full" else int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

        # h = self.fc2(h)
class WEITResNet(nn.Module):
    def __init__(self,output=100,nc_per_task = 10):
        super().__init__()

        self.feature_net = resnetWEIT18()
        self.last = DecomposedLinear(self.feature_net.outlen, output)
        self.weight_keys = []
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def set_sw(self,glob_weights):
        self.feature_net.conv1.sw = Parameter(glob_weights[0])
        self.feature_net.layer1[0].conv1.sw = Parameter(glob_weights[1])
        self.feature_net.layer1[0].conv2.sw = Parameter(glob_weights[2])
        self.feature_net.layer1[1].conv1.sw = Parameter(glob_weights[3])
        self.feature_net.layer1[1].conv2.sw = Parameter(glob_weights[4])
        self.feature_net.layer2[0].conv1.sw = Parameter(glob_weights[5])
        self.feature_net.layer2[0].conv2.sw = Parameter(glob_weights[6])
        self.feature_net.layer2[1].conv1.sw = Parameter(glob_weights[7])
        self.feature_net.layer2[1].conv2.sw = Parameter(glob_weights[8])
        self.feature_net.layer3[0].conv1.sw = Parameter(glob_weights[9])
        self.feature_net.layer3[0].conv2.sw = Parameter(glob_weights[10])
        self.feature_net.layer3[1].conv1.sw = Parameter(glob_weights[11])
        self.feature_net.layer3[1].conv2.sw = Parameter(glob_weights[12])
        self.feature_net.layer4[0].conv1.sw = Parameter(glob_weights[13])
        self.feature_net.layer4[0].conv2.sw = Parameter(glob_weights[14])
        self.feature_net.layer4[1].conv1.sw = Parameter(glob_weights[15])
        self.feature_net.layer4[1].conv2.sw = Parameter(glob_weights[16])
        self.last.sw = glob_weights[17]
    def set_knowledge(self,t,from_kbs):
        self.feature_net.conv1.set_atten(t,from_kbs[0].size(-1))
        self.feature_net.conv1.set_knlwledge(from_kbs[0])
        self.feature_net.layer1[0].conv1.set_atten(t, from_kbs[1].size(-1))
        self.feature_net.layer1[0].conv1.set_knlwledge(from_kbs[1])
        self.feature_net.layer1[0].conv2.set_atten(t, from_kbs[2].size(-1))
        self.feature_net.layer1[0].conv2.set_knlwledge(from_kbs[2])
        self.feature_net.layer1[1].conv1.set_atten(t, from_kbs[3].size(-1))
        self.feature_net.layer1[1].conv1.set_knlwledge(from_kbs[3])
        self.feature_net.layer1[1].conv2.set_atten(t, from_kbs[4].size(-1))
        self.feature_net.layer1[1].conv2.set_knlwledge(from_kbs[4])

        self.feature_net.layer2[0].conv1.set_atten(t, from_kbs[5].size(-1))
        self.feature_net.layer2[0].conv1.set_knlwledge(from_kbs[5])
        self.feature_net.layer2[0].conv2.set_atten(t, from_kbs[6].size(-1))
        self.feature_net.layer2[0].conv2.set_knlwledge(from_kbs[6])
        self.feature_net.layer2[1].conv1.set_atten(t, from_kbs[7].size(-1))
        self.feature_net.layer2[1].conv1.set_knlwledge(from_kbs[7])
        self.feature_net.layer2[1].conv2.set_atten(t, from_kbs[8].size(-1))
        self.feature_net.layer2[1].conv2.set_knlwledge(from_kbs[8])

        self.feature_net.layer3[0].conv1.set_atten(t, from_kbs[9].size(-1))
        self.feature_net.layer3[0].conv1.set_knlwledge(from_kbs[9])
        self.feature_net.layer3[0].conv2.set_atten(t, from_kbs[10].size(-1))
        self.feature_net.layer3[0].conv2.set_knlwledge(from_kbs[10])
        self.feature_net.layer3[1].conv1.set_atten(t, from_kbs[11].size(-1))
        self.feature_net.layer3[1].conv1.set_knlwledge(from_kbs[11])
        self.feature_net.layer3[1].conv2.set_atten(t, from_kbs[12].size(-1))
        self.feature_net.layer3[1].conv2.set_knlwledge(from_kbs[12])

        self.feature_net.layer4[0].conv1.set_atten(t, from_kbs[13].size(-1))
        self.feature_net.layer4[0].conv1.set_knlwledge(from_kbs[13])
        self.feature_net.layer4[0].conv2.set_atten(t, from_kbs[14].size(-1))
        self.feature_net.layer4[0].conv2.set_knlwledge(from_kbs[14])
        self.feature_net.layer4[1].conv1.set_atten(t, from_kbs[15].size(-1))
        self.feature_net.layer4[1].conv1.set_knlwledge(from_kbs[15])
        self.feature_net.layer4[1].conv2.set_atten(t, from_kbs[16].size(-1))
        self.feature_net.layer4[1].conv2.set_knlwledge(from_kbs[16])

        self.last.set_atten(t, from_kbs[17].size(-1))
        self.last.set_knlwledge(from_kbs[17])
    def get_weights(self):
        weights = []

        w = self.feature_net.conv1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        w = self.feature_net.layer1[0].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer1[0].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer1[1].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer1[1].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False

        w = self.feature_net.layer2[0].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer2[0].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer2[1].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer2[1].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False

        w = self.feature_net.layer3[0].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer3[0].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer3[1].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer3[1].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False

        w = self.feature_net.layer4[0].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer4[0].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer4[1].conv1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.feature_net.layer4[1].conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False

        w = self.last.get_weight().detach()
        weights.append(w)
        w.requires_grad = False

        return weights
    def forward(self,x,t,pre=False,is_con=False):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t  * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

class Classification(nn.Module):
    def __init__(self, output=100, nc_per_task=10):
        super().__init__()

        self.last = nn.Linear(1024, output)
        self.nc_per_task=nc_per_task
        self.n_outputs = output

    def forward(self, h, t, pre=False, is_cifar=True):
        output = self.last(h)
        if is_cifar:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
class SupConMLP(nn.Module):
    def __init__(self, inputsize, output=100, nc_per_task=10):
        super().__init__()
        self.encoder = Cifar100Net(inputsize)
        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )
    def forward(self, x, return_feat=False):
        encoded = self.encoder(x)
        feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat