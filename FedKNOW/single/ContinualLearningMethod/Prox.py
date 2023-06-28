import sys, time, os
import numpy as np
import torch
from copy import deepcopy

from torch.utils.data import DataLoader
from tqdm import tqdm
from FedKNOW.utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog

from FedKNOW.models.Update import DatasetConverter
from FedKNOW.utils.labelmapper import labelMapper

sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_decay = args.lr_decay
        self.optim_type = args.optim
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.old_task=-1
        self.lamb = args.lamb
        self.num_classes = args.num_classes // args.task
        return
    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        if "SGD" in self.optim_type:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        return optimizer
    def train(self, t):
        if t!=self.old_task:
            self.old_task=t
        lr = self.lr
        self.optimizer = self._get_optimizer()
        # Loop epochs
        self.globmodel = deepcopy(self.model)
        for e in range(self.nepochs):
            self.train_epoch(t, self.globmodel)
            train_loss, train_acc = self.eval(t)
            if e % self.nepochs == self.nepochs -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops

        return train_loss, train_acc

    def train_epoch(self, t, globmodel):
        self.model.train()
        for images,targets in self.tr_dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
                # targets = (targets - self.num_classes * t).cuda()
                targets.apply_(lambda x: int(labelMapper(x) + 5*t))
                targets = targets.cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, 5)
            self.optimizer.zero_grad()
            self.model.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            outputs = self.model.forward(images,t)
            # outputs = self.model.forward(images, t)
            _, pred = outputs.max(1)
            loss = self.ce(outputs, targets)
            proximal_term = 0.0
            for w, w_t in zip(self.model.parameters(), globmodel.parameters()):
                    proximal_term += (w - w_t).norm(2)

            loss += self.lamb*0.5*proximal_term

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    def eval(self, t,train=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        if train:
            dataloaders = self.tr_dataloader

        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets.apply_(lambda x: int(labelMapper(x) + 5*t))
                    # targets = (targets - self.num_classes*t).cuda()
                    targets = targets.cuda()
                # Forward
                offset1, offset2 = compute_offsets(t,5)
                output = self.model.forward(images,t)
                # output = self.model.forward(images,t)
                loss = self.ce(output, targets)
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss += self.lamb*0.5*proximal_term
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def evalCustom(self, t):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        dataloaders = self.tr_dataloader
        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                if torch.cuda.is_available():
                    images = images.cuda()
                    # targets = (targets - self.num_classes*t).cuda()
                    targets.apply_(lambda x: int(labelMapper(x) + 5*t))
                    targets = targets.cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 5)
                output = self.model.forward(images,t)
                # output = self.model.forward(images,t)
                loss = self.ce(output, targets)
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss += self.lamb*0.5*proximal_term
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round

    print('*' * 100)
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task)
    print('-' * 100)
    return appr.model.state_dict(),loss,0

def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].cuda()
        ytest = (testdatas[u][1] - u * 10).cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
                                                                              100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t])
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    print('Save at ' + args.output)
    if r == args.round - 1:
        writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc

def LongLifeTestCustom(args, appr, t, testdatas, writer): #this returns the test accuracy for a client on all the tasks learned until now
    acc = np.zeros((1, t+1), dtype=np.float32)
    lss = np.zeros((1, t+1), dtype=np.float32)
    for u in range(t + 1):
        taskdata = testdatas[u]
        DatasetObj = DatasetConverter(taskdata, "cifar100")
        tr_dataloader = DataLoader(DatasetObj,batch_size=args.local_bs, shuffle=True)
        appr.tr_dataloader = tr_dataloader
        test_loss, test_acc = appr.evalCustom(u)
        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss, 100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t+1])
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    # print('Save at ' + args.output)
    # if r == args.round - 1:
    writer.add_scalar('task_finish_and_agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc