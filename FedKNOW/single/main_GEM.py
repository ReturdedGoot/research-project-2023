import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from FedKNOW.single.ContinualLearningMethod.GEM import LongLifeTestCustom
from FedKNOW.utils.options import args_parser
from FedKNOW.utils.train_utils import get_data, get_model, read_data
from FedKNOW.models.Update import DatasetSplit, DatasetConverter
from FedKNOW.models.test import test_img_local_all
from FedKNOW.single.ContinualLearningMethod.GEM import Appr,LongLifeTrain
from FedKNOW.models.Nets import RepTail
from pathlib import Path
from torch.utils.data import DataLoader
import time

import gc

filename = None
folder = './experimental_data/'

gc.enable()
def writefile(string):
        with open(filename, 'a') as file:
            file.write(string+'\n')



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cpu')

    if(args.seed != 1): #set seed if its not the default value
        np.random.seed(args.seed)

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or 'MiniImageNet' in args.dataset or 'FC100' in args.dataset or 'CORe50' in args.dataset or 'TinyImageNet' or 'Twitter' in args.dataset or 'Emotions' in args.dataset or 'Music' in args.dataset or 'Forest' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        # for idx in dict_users_train.keys():
        #     np.random.shuffle(dict_users_train[idx])
    else:
        print('Not this dataset!!')

    filename = folder+ args.exp_name + ".csv"
    print(args.alg)
    tb_writers = {}
    logdir = Path(f'./log/GEM_{time.time()}')
    logdir.mkdir(exist_ok=True, parents=True)
    logdir_base = str(logdir / f'{args.dataset}_round{args.round}_frac{args.frac}_model_{args.model}')

    # To store data to write to file later
    round_data_file = str(logdir / 'round_data.csv')
    round_data =[ ['step', 'test_acc', 'test_loss']]
    client_data_file = str(logdir / 'client_data.csv')
    client_data = [['client_id', 'step', 'test_acc', 'test_loss']]
    args_config_file = str(logdir / 'cli_args.json')

    tb_writers['server'] = SummaryWriter(f'{logdir_base}_server')

    # build model
    # net_glob = get_model(args)
    net_glob = get_model(args)
    net_glob.train()
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]
    w_glob_keys = []

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        tb_writers[user] = SummaryWriter(f'{logdir_base}_client{user}')
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    num_inputs = 3*32*32
    if 'Twitter' in args.dataset:
        num_inputs = 64
    elif 'Emotions' in args.dataset:
        num_inputs = 66
    elif 'Forest' in args.dataset:
        num_inputs = 54
    apprs = [Appr(net_glob.to(args.device), num_inputs, args.num_classes,args.task, args) for i in range(args.num_users)]
    print(args.round)

    args.epochs = args.task * args.round
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
        w_glob = {}
        fisher_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        for ind, idx in enumerate(idxs_users):
            glob_fisher = None
            start_in = time.time()

            # DatasetObj = DatasetConverter(dict_users_train[idx][task], "cifar100")
            tr_dataloaders = DataLoader(dict_users_train[idx][task],batch_size=args.local_bs, shuffle=True)
            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task],dict_users_train[idx][:args.m_ft]),batch_size=args.local_bs, shuffle=True)
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)
            appr = apprs[idx]
            appr.set_model(net_local.to(args.device))
            # appr.set_trData(tr_dataloaders)
            last = iter == args.epochs
            w_local,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * lens[idx]
                    else:
                        w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            times_in.append(time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)
        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        # if iter % args.round == args.round-1:
        if times == []:
            times.append(max(times_in))
        else:
            times.append(times[-1] + max(times_in))

        #TODO: Client test evaluation
        client_test_accuracies = []
        client_test_losses = []
        client_test_bwts = []

        for client in range(args.num_users):
            print("client", client,'test:')
            testdataset = dict_users_test[client]
        #     apprs[client].set_sw(w_glob) #set the model to the aggregated model
            client_avg_loss, client_avg_test_accuracy, client_backward_transfer = LongLifeTestCustom(args, apprs[client], task, testdataset, tb_writers[client])
            client_test_accuracies.append(client_avg_test_accuracy)
            client_test_losses.append(client_avg_loss)
            client_test_bwts.append(client_backward_transfer)

            client_data.append([client, iter, client_avg_test_accuracy, client_avg_loss])
        overall_avg_task_accuracies = sum(client_test_accuracies)/len(client_test_accuracies)
        overall_avg_task_losses = sum(client_test_losses)/len(client_test_losses)
        overall_backward_transfer = sum(client_test_bwts)/len(client_test_bwts)
        average_accuracy =100*overall_avg_task_accuracies
        average_loss_clients =overall_avg_task_losses
        average_client_bwt = overall_backward_transfer
        print("Avg accuracy across clients: {:.2f}%".format(100*overall_avg_task_accuracies))
        print("Avg loss across clients: {:.2f}".format(overall_avg_task_losses))
        print("Avg backward transfer clients: {:.2f}".format(overall_backward_transfer))


        writefile(f'{average_accuracy}, {average_loss_clients}, {average_client_bwt}')
        # writefile("Avg accuracy across clients: {:.2f}%".format(100*overall_avg_task_accuracies))
        print("Avg loss across clients: {:.2f}".format(overall_avg_task_losses))
        round_data.append([iter, overall_avg_task_accuracies, overall_avg_task_losses])
        tb_writers['server'].add_scalar('avg test loss', overall_avg_task_losses, iter)
        tb_writers['server'].add_scalar('avg test accuracy', overall_avg_task_accuracies, iter)
        gc.collect()
