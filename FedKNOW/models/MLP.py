import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math

class MLP(nn.Module):

    def __init__(self, num_inputs=54, num_classes=7):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_classes = num_classes

        self.layer_size = 500

        #creating the various nn layers
        self.layer1 = nn.Linear(self.num_inputs, self.layer_size)
        self.layer2 = nn.Linear(self.layer_size, self.layer_size)
        self.layer3 = nn.Linear(self.layer_size, self.layer_size)
        self.layer4 = nn.Linear(self.layer_size, self.num_classes)
        self.selu1 = nn.ReLU()
        self.selu2 = nn.ReLU()
        self.selu3 = nn.ReLU()


    def forward(self, input, t):
        # print(f'input: {input}')
        # print(f' input size: {input.size()}')
        # print(f'tensor contents{[print(x.dtype) for x in input[0]]}')
        input = input.type(torch.FloatTensor)
        out = self.layer1(input)
        # print(f'output first layer: {out1.size()}')
        out = self.layer2(self.selu1(out))
        # print(f'output second layer: {out2.size()}')
        out = self.layer3(self.selu2(out))
        # print(f'output third layer: {out3.size()}')
        out = self.layer4(self.selu3(out))

        #make sure to apply a softmax layer inbetween.
        # out = nn.functional.softmax(out, dim=1)
        # print(f'Class predictions: {class_predictions}')
        return out



class MLP_Weit(nn.Module):
    def __init__(self, num_inputs=54, num_classes=7):
        super(MLP_Weit, self).__init__()


        self.num_inputs = num_inputs
        self.num_classes = num_classes

        self.layer_size = 500

        #here we create the various layers

        self.layer1 = DecomposedLinear(num_inputs, self.layer_size)
        self.layer2 = DecomposedLinear(self.layer_size, self.layer_size)
        self.layer3 = DecomposedLinear(self.layer_size, self.layer_size)
        self.layer4 = DecomposedLinear(self.layer_size, num_classes)

        self.selu1 = nn.ReLU()
        self.selu2 = nn.ReLU()
        self.selu3 = nn.ReLU()


        self.weight_keys = []
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)


    def forward(self, input, t):
        # print(f' input size: {input.size()}')
        # print(f'tensor contents{[print(x.dtype) for x in input[0]]}')
        out = self.layer1(input)
        # print(f'output first layer: {out1.size()}')
        out = self.layer2(self.selu1(out))
        # print(f'output second layer: {out2.size()}')
        out = self.layer3(self.selu2(out))
        # print(f'output third layer: {out3.size()}')
        out = self.layer4(self.selu3(out))

        #make sure to apply a softmax layer inbetween.
        # out = nn.functional.softmax(out, dim=1)
        # print(f'Class predictions: {class_predictions}')
        return out


    def set_sw(self, glob_weights):
        self.layer1.sw = Parameter(glob_weights[0])
        self.layer2.sw = Parameter(glob_weights[1])
        self.layer3.sw = Parameter(glob_weights[2])
        self.layer4.sw = Parameter(glob_weights[3])


    def set_knowledge(self, t, from_kbs):

        self.layer1.set_atten(t, from_kbs[0].size(-1))
        self.layer1.set_knlwledge(from_kbs[0])

        self.layer2.set_atten(t, from_kbs[1].size(-1))
        self.layer2.set_knlwledge(from_kbs[1])

        self.layer3.set_atten(t, from_kbs[2].size(-1))
        self.layer3.set_knlwledge(from_kbs[2])

        self.layer4.set_atten(t, from_kbs[3].size(-1))
        self.layer4.set_knlwledge(from_kbs[3])

    def get_weights(self):
        weights = []

        w = self.layer1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        w = self.layer2.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        w = self.layer3.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        w = self.layer4.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        return weights

class DecomposedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,bias: bool = True) -> None:
        super(DecomposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sw = Parameter(torch.FloatTensor(out_features, in_features))
        self.mask = Parameter(torch.FloatTensor(out_features))
        self.aw = Parameter(torch.FloatTensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.sw, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.sw)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.mask, -bound, bound)
    def set_atten(self,t,dim):
        device = torch.device('cpu')
        if t==0:
            self.atten = Parameter(torch.zeros(dim).to(device))
            self.atten.requires_grad=False
        else:
            self.atten = Parameter(torch.rand(dim).to(device))
    def set_knlwledge(self,from_kb):
        self.from_kb = from_kb

    def get_weight(self):
        m = nn.Sigmoid()
        sw = self.sw.transpose(0, -1)
        # newmask = m(self.mask)
        # print(sw*newmask)
        device = torch.device( 'cpu')
        weight = (sw * m(self.mask)).transpose(0, -1) + self.aw + torch.sum(self.atten * self.from_kb.to(device), dim=-1)
        if(device.type != 'cpu'):
            weight = weight.type(torch.cuda.FloatTensor)
        else:
            weight = weight.type(torch.FloatTensor)
        return weight
    def forward(self, input):
        weight = self.get_weight()

        # weight = weight.type(torch.float64)
        # self.bias = self.bias.type(torch.float64)
        # input = input.type(torch.float64)
        # print(f'dtype input: {input.dtype}')
        # print(f'dtype weights: {weight.dtype}')
        # print(f'dtype bias: {self.bias.dtype}')
        return F.linear(input, weight, self.bias)