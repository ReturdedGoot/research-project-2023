import torch
from torch import nn
from FedKNOW.models.layer import DecomposedConv,DecomposedLinear
from FedKNOW.models.Nets import compute_conv_output_size


class LeNetWEIT(nn.Module):
    def __init__(self, inputsize,n_ouputs=100,nc_per_task=5, window = "sliding"):
        super().__init__()
        self.nc_per_task = nc_per_task
        self.n_outputs = n_ouputs
        self.window = window
        ncha, size, _ = inputsize
        self.conv1 = DecomposedConv(in_channels=ncha, out_channels = 20, kernel_size=5, padding=2,stride=1)
        s = compute_conv_output_size(size, 5, stride=1,padding=2)  # 32

        self.pool1 = torch.nn.MaxPool2d(3, stride=2, padding=1) #16
        s = compute_conv_output_size(s, 3, stride=2,padding=1)

        self.conv2 = DecomposedConv(in_channels=20, out_channels=50, kernel_size=5, padding=2,stride=1) #16
        s = compute_conv_output_size(s, 5, stride=1,padding=2)

        self.pool2 = torch.nn.MaxPool2d(3, stride=2, padding=1) #8
        s = compute_conv_output_size(s, 3, padding=1,stride=2)  # 32
        self.fc1 = DecomposedLinear(s*s*50, 800)
        self.fc2 = DecomposedLinear(800, 500)
        self.last = torch.nn.Linear(500, self.n_outputs)

        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.layer_keys = [['conv1'],['conv2'],['fc1'],['fc2'],['last']]
    def set_sw(self,glob_weights):
        self.conv1.sw = Parameter(glob_weights[0])
        self.conv2.sw = Parameter(glob_weights[1])
        self.fc1.sw = Parameter(glob_weights[2])
        self.fc2.sw = Parameter(glob_weights[3])
    def set_knowledge(self,t,from_kbs):
        self.conv1.set_atten(t,from_kbs[0].size(-1))
        self.conv1.set_knlwledge(from_kbs[0])
        self.conv2.set_atten(t, from_kbs[1].size(-1))
        self.conv2.set_knlwledge(from_kbs[1])
        self.fc1.set_atten(t, from_kbs[2].size(-1))
        self.fc1.set_knlwledge(from_kbs[2])
        self.fc2.set_atten(t, from_kbs[3].size(-1))
        self.fc2.set_knlwledge(from_kbs[3])
    def get_weights(self):
        weights = []
        w = self.conv1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)
        w = self.conv2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.fc1.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        w = self.fc2.get_weight().detach()
        weights.append(w)
        w.requires_grad = False
        return weights
    def forward(self, x, t,avg_act=False):
        if x.size(1) !=3:
            bsz = x.size(0)
            x = x.view(bsz,3,32,32)
        act1 = self.relu(self.conv1(x))

        h = self.drop1(self.pool1(act1))

        act2 = self.relu(self.conv2(h))

        h = self.drop1(self.pool2(act2))

        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        act8 = self.relu(self.fc2(h))
        output = self.last(act8)
        # make sure we predict classes within the current task
        offset1 = 0 if self.window != "sliding" else int(t * self.nc_per_task)
        # offset1 = int(t * self.nc_per_task)
        offset2 = self.n_outputs if self.window == "full" else int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output
