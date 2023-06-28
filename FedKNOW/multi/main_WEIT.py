import copy
import blosc
import numpy as np
import torch
from FedKNOW.utils.options import args_parser
from FedKNOW.utils.train_utils import get_data, get_model
from FedKNOW.models.Update import DatasetSplit
from FedKNOW.multi.ContinualLearningMethod.WEIT import Appr,LongLifeTest,LongLifeTrain
from torch.utils.data import DataLoader
import time
from FedKNOW.models.Packnet import PackNet
import flwr as fl
import pickle
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from collections import OrderedDict
import datetime
import time
import sys
import gzip

from_kb = []

class FPKDClient(fl.client.NumPyClient):
    def __init__(self,appr,args):
        self.appr= appr
        self.args = args
        self.curTask = 0
    def get_parameters(self):
        net = appr.get_sw()
        return [val.detach().cpu().numpy() for val in net]

    def set_parameters(self, parameters):
        # print(len(parameters))
        # print("the last element on that list is",parameters.pop())
        net = appr.set_sw(parameters)
        return net
        # print(parameters,"olalalalala")
        # exit(0)
        # params_dict = zip(net.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        start = time.time()
        global from_kb
        train_round = config['round']
        if(config['kb'] != ""):
            from_kb = list(map(lambda x: torch.from_numpy(x), pickle.loads(gzip.decompress(config['kb']))))
        begintime = datetime.datetime.now()
        print('cur round{} begin training ,time is {}'.format(train_round,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        self.set_parameters(parameters)
        w_local, aws,loss, indd, acc = LongLifeTrain(self.args,appr,train_round-1,from_kb,args.client_id)
        kb_str = ""
        if (train_round-1) % args.round == args.round -1:
            from_kb_l = []
            ind = args.client_id
            for aw in aws:
                from_kb_l.append(aw.cpu().detach().numpy())
                #shape = np.concatenate([aw.shape, [int(round(args.num_users * args.frac))]], axis=0)
            kb_str = pickle.dumps(from_kb_l)
            kb_str = gzip.compress(kb_str)
                #from_kb_l = np.zeros(shape)
                #if len(shape) == 5:
                    #from_kb_l[:, :, :, :, ind] = aw.cpu().detach().numpy()
                #else:
                    #from_kb_l[:, :, ind] = aw.cpu().detach().numpy()
                #from_kb_l = torch.from_numpy(from_kb_l)
                #from_kb.append(from_kb_l)
        params = self.get_parameters() #No need to compress this is automatically done
        params_copy = weights_to_parameters(params) #to compress it using gzip
        #new_params = parameters_to_weights(params)
        endtime =time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('cur round {} end training ,time is {}'.format(train_round, endtime))
        end = time.time()
        clientExecTime = end - start
        paramSize = sum([len(x) for x in params_copy.tensors])
        kbSize = sys.getsizeof(kb_str)
        return params, indd, {'kb':kb_str,'clientExecTime':clientExecTime, 'parameter_size':paramSize, 'kb_size':kbSize, 'train_acc': acc}
        #return self.get_parameters(), indd, {}

    def evaluate(self, parameters, config):
        print('eval:')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test_round = config['round']
        self.set_parameters(parameters)
        loss, accuracy,totalnum = LongLifeTest(args, appr, test_round-1)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return float(loss), totalnum, {"accuracy": float(accuracy)}


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    net_glob = get_model(args)
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    print(total_num_layers)
    print(net_keys)

    # generate list of local models for each user

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    start = time.time()
    task=-1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # appr = Appr(copy.deepcopy(net_glob),PackNet(args.task,local_ep=args.local_ep,local_rep_ep=args.local_rep_ep,device=args.device),copy.deepcopy(net_glob), None,lr=args.lr, nepochs=args.local_ep, args=args)
    appr = Appr(copy.deepcopy(net_glob).to(device), None,lr=args.lr, nepochs=args.local_ep, args=args)
    for name,para in net_glob.named_parameters():
        if 'aw' in name:
            shape = np.concatenate([para.shape, [int(round(args.num_users * args.frac))]], axis=0)
            from_kb_l = np.zeros(shape)
            from_kb_l = torch.from_numpy(from_kb_l)
            from_kb.append(from_kb_l)
    for i in range(args.task):
        tr_dataloaders = DataLoader(DatasetSplit(dataset_train[i], dict_users_train[args.client_id]),
                                    batch_size=args.local_bs, shuffle=True, num_workers=0)
        te_dataloader = DataLoader(DatasetSplit(dataset_test[i], dict_users_test[args.client_id]), batch_size=args.local_test_bs, shuffle=False,num_workers=0)
        appr.traindataloaders.append(tr_dataloaders)
        appr.testdataloaders.append(te_dataloader)
    client = FPKDClient(appr,args)
    fl.client.start_numpy_client(args.ip, client=client)
