import sys, time, os
import numpy as np
import torch
from copy import deepcopy

from torch.utils.data import DataLoader
from tqdm import tqdm
from FedKNOW.utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
import math
from FedKNOW.models.Update import DatasetConverter
from FedKNOW.utils.labelmapper import labelMapper

sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn

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
def fisher_matrix_diag(t,dataloader, model):
    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    offset1, offset2 = compute_offsets(t, 10)
    all_num = 0
    for images,target in dataloader:
        # target.apply_(lambda x: int(labelMapper(x) + 5*t))          #TODO add if statement

        # if torch.cuda.is_available():
        #     images = images.cuda()
        #     target = target.cuda()

        all_num += images.shape[0]
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, t)
        loss = criterion(outputs, target)
        loss.backward()
        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += images.shape[0] * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n, _ in model.named_parameters():
            fisher[n] = fisher[n] / all_num
    return fisher
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
def MultiClassCrossEntropy(logits, labels, t,T=2):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100, lr_decay = 0, optim = 'Adam', lamb = 1, local_epoch = 6
                 , num_classes = 2, task = 10,
                 args=None):
        self.model = model
        self.model_old = model
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_decay = lr_decay
        self.optim_type = optim
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.eval_matrix =[]

        self.eval_old_task = -1
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = lamb
        self.e_rep = local_epoch
        self.num_classes = num_classes // task
        self.old_task=-1
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())

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
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task=t
        lr = self.lr
        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            self.train_epoch(t)


            train_loss, train_acc = self.eval(t)
            if e % self.e_rep == self.e_rep -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops
        fisher_old = {}
        if t>0:
            for n, _ in self.model.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = fisher_matrix_diag(t,self.tr_dataloader, self.model)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (
                        t + 1)  # Checked: it is better than the other option

        return train_loss, train_acc

    def train_epoch(self,t):
        self.model.train()
        print(f'Number of batches: {len(self.tr_dataloader) }')
        total_loss = 0
        total_acc = 0
        total_num = 0
        for count, (images, targets) in enumerate(self.tr_dataloader):
            # if torch.cuda.is_available():
            #     images = images.cuda()
            # targets = (targets - self.num_classes * t)
            #     targets.apply_(lambda x: int(labelMapper(x) + 5*t))
            #     targets = targets.cuda()
            # Forward current model
            # print(f'images size: {images.size()}')


            offset1, offset2 = compute_offsets(t, 5)
            # self.optimizer.zero_grad()
            # self.model.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            outputs = self.model.forward(images,t)
            # outputs = self.model.forward(images, t)
            _, pred = outputs.max(1)
            # print(f"outputs: {outputs}")
            loss = self.criterion(t, outputs, targets)


            hits = torch.eq(pred, targets).int()

            # Log
            total_loss += loss.data.cpu().numpy() * len(images)
            total_acc += hits.sum().data.cpu().numpy()
            total_num += len(images)

            # print(f"Accuracy: {total_acc}, total num: {total_num}")


            # print(f'pred: {pred}')
            # print(f'target: {targets}')
            # print(f'hits : {hits}')


            # index = 1
            # # print(f'input: {images[index]}')
            # # print(f'output: {outputs[index]}')
            # print(f'label: {targets[index]}')
            # print(f'predicted: {pred[index]}')
            # print(f'loss: {loss}')

            # raise Exception
            # Backward
            print(f"Batch number: {count}, loss: {loss:4f}")

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
                # if torch.cuda.is_available():
                #     images = images.cuda()
                #     targets.apply_(lambda x: int(labelMapper(x) + 5*t))
                #     # targets = (targets - self.num_classes*t).cuda()
                #     targets = targets.cuda()
                # Forward
                offset1, offset2 = compute_offsets(t,5)
                output = self.model.forward(images,t)
                loss = self.criterion(t, output, targets)


                _, pred = output.max(1)
                hits = torch.eq(pred, targets).int()

                # print(f'hits: {hits}')

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg

    def evalCustom(self, t):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        dataloaders = self.tr_dataloader
        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                # if torch.cuda.is_available():
                #     images = images.cuda()
                #     # targets = (targets - self.num_classes*t).cuda()
                #     targets.apply_(lambda x: int(labelMapper(x) + 5*t))
                #     targets = targets.cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 5)
                output = self.model.forward(images,t)
                loss = self.criterion(t, output, targets)
                _, pred = output.max(1)

                hits = torch.eq(pred, targets).int()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

                # print(f"Accuracy: {total_acc}, total num: {total_num}")


        return total_loss / total_num, total_acc / total_num




def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
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
        xtest = testdatas[u][0]
        ytest = (testdatas[u][1] - u * 10)
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
                                                                              100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t])
    # print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    # print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    # print('Save at ' + args.output)
    if r == args.round - 1:
        writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc

def LongLifeTestCustom(args, appr, t, testdatas, writer):
    acc = np.zeros((1, t+1), dtype=np.float32)
    lss = np.zeros((1, t+1), dtype=np.float32)
    for u in range(t + 1):
        taskdata = testdatas[u]
        # DatasetObj = DatasetConverter(taskdata, "cifar100")
        # print(f'testing data length: {len(taskdata)}')
        tr_dataloader = DataLoader(taskdata,batch_size=args.local_bs, shuffle=True)
        appr.tr_dataloader = tr_dataloader
        test_loss, test_acc = appr.evalCustom(u)

        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss, 100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t+1])
    # print(f'accuracy: {acc}')

    #here we need to add the backward transfer part
    #TODO

    if appr.eval_old_task != t:
        #here we only want to append if it is a new task and not an old task.
        appr.eval_matrix.append(acc[0]) #acc has an extra dimension we do not need, so we just index it instead.
        appr.eval_old_task = t
    else:
        # print(f"OLD TASK VALUE: {appr.eval_old_task}")
        # print(f"T VALUE: {t}")
        #if we did encounter an old task, we just replace that instance with the latest round information.
        appr.eval_matrix[t] = acc[0]

    # print(f' appr Matrix: {appr.eval_matrix}')
    # now we make sure to evaulate the backward transfer
    bwt = np.zeros((1, t), dtype=np.float32)
    for u in range(t):
        #compare this with the latest test accuracy we just obtained.
        diff = acc[0, t] - appr.eval_matrix[u][u]
        bwt[0, u] = diff

    task_bwt = np.mean(bwt)
    isnan = False
    if math.isnan(task_bwt):
        task_bwt = 0
        isnan = True

    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    print(f'Backward Transfer: {task_bwt:5.1f}, is Nan: {isnan}')
    writer.add_scalar('task_finish_and_agg', mean_acc, t + 1)
    return mean_lss, mean_acc, task_bwt