import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from FedKNOW.utils.options import args_parser
from FedKNOW.utils.train_utils import get_data, get_model, read_data
from FedKNOW.models.Update import DatasetSplit, DatasetConverter
from FedKNOW.models.test import test_img_local_all, test_img_local_all_WEIT
from FedKNOW.single.ContinualLearningMethod.WEIT import Appr,LongLifeTrain, LongLifeTestCustom
from FedKNOW.models.Nets import WEITResNet
from torch.utils.data import DataLoader
import time
from pathlib import Path
import csv
import json
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import gc

filename = None
folder = './experimental_data/'

gc.enable()
def writefile(string):
        with open(filename, 'a') as file:
            file.write(string+'\n')



logging.basicConfig(
     level=logging.INFO
    #  format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    #  datefmt='%H:%M:%S'
 )

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    filename = folder+ args.exp_name + ".csv"

    if(args.seed != 1): #set seed if its not the default value
        np.random.seed(args.seed)

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset or 'Twitter' in args.dataset or 'Emotions' in args.dataset or 'Music' in args.dataset or 'Forest' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args) #gets the partitioned data

    logging.info(f'Agorithm used: {args.alg}')
    tb_writers = {}
    exp_name = args.exp_name
    if exp_name != '':
        exp_name += '_'
    logdir = Path(f'./log/WEIT_{exp_name}{int(time.time())}')
    logdir.mkdir(exist_ok=True, parents=True)
    logdir_base = str(logdir / f'{args.dataset}_round{args.round}_frac{args.frac}_model_{args.model}')

    # To store data to write to file later
    round_data_file = str(logdir / 'round_data.csv')
    round_data =[ ['step', 'test_acc', 'test_loss']]
    client_data_file = str(logdir / 'client_data.csv')
    client_data = [['client_id', 'step', 'test_acc', 'test_loss']]
    args_config_file = str(logdir / 'cli_args.json')

    # Add tensorboard writer for the server node
    tb_writers['server'] = SummaryWriter(f'{logdir_base}_server')

    # build model
    net_glob = get_model(args).to(args.device)
    net_glob.train()
    total_num_layers = len(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()] #returns the keys to access the network layers by name
    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    w_glob_keys = []

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    with logging_redirect_tqdm():
        for user in range(args.num_users):
            w_local_dict = {}
            # Add a tensorboard writer for each client
            tb_writers[user] = SummaryWriter(f'{logdir_base}_client{user}')
            for key in net_glob.state_dict().keys():
                w_local_dict[key] = net_glob.state_dict()[key] #for each client create a local dictionary to access each of its model's layers by name
            w_locals[user] = w_local_dict #w_locals[client number][layername] gives that particular layer


        indd = None  # indices of embedding for sent140
        loss_train = []
        accs = []
        times = []
        accs10 = 0
        accs10_glob = 0
        start = time.time()
        task=-1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        apprs = [Appr(copy.deepcopy(net_glob).to(device), None, lr=args.lr, nepochs=args.local_ep, args=args, num_classes=1) for i in range(args.num_users)]
        logging.info(f"{args.epochs} rounds in total")
        from_kb =[]

        #initialize knowledge base
        for name,para in net_glob.named_parameters():
            if 'aw' in name:
                shape = np.concatenate([para.shape, [int(round(args.num_users * args.frac))]], axis=0)
                from_kb_l = np.zeros(shape)
                from_kb_l = torch.from_numpy(from_kb_l).to(args.device)
                from_kb.append(from_kb_l)
        w_glob=[]

        # Compute the total amount of ticks we are going to make
        # total = args.epochs * (num_users * fraction)
        m = max(int(args.frac * args.num_users), 1)

        total_ticks = int(args.epochs) * m

        #training
        pbar = tqdm(total=total_ticks)
        args.epochs = args.round * args.task #making sure epochs is the product of rounds and tasks.
        for iter in range(args.epochs):
            pbar.set_description(f'Round {iter}')
            if iter % (args.round) == 0:
                task+=1
            w_agg=w_glob
            w_glob = []
            loss_locals = []
            # m = max(int(args.frac * args.num_users), 1)
            if iter == args.epochs:
                m = args.num_users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False) #select fraction of users from total number of users at each epoch
            w_keys_epoch = w_glob_keys
            times_in = []
            total_len = 0
            tr_dataloaders= None
            if iter % args.round == 0:
                for i in range(args.num_users):
                    apprs[i].model.set_knowledge(task,from_kb)

            for ind, idx in enumerate(idxs_users): #time-share between all users selected for that epoch. Multiprocessing can be used here
                start_in = time.time()

                if not 'Twitter' and not 'Emotions'  in args.dataset:
                    DatasetObj = DatasetConverter(dict_users_train[idx][task], "cifar100")

                tr_dataloaders = DataLoader(dict_users_train[idx][task],batch_size=args.local_bs, shuffle=True)
                w_local = []
                appr = apprs[idx] #select the model parameters for client under consideration
                appr.set_sw(w_agg)
                appr.set_trData(tr_dataloaders) #client's data partition
                last = iter == args.epochs
                w_local, aws,loss, indd = LongLifeTrain(args,appr,iter,from_kb,idx, tb_writers[idx]) #trains the client for one round

                #update the knowledge base after the last round for the task
                if iter % args.round == args.round -1:
                    from_kb = []
                    for aw in aws:
                        shape = np.concatenate([aw.shape, [int(round(args.num_users * args.frac))]], axis=0)
                        from_kb_l = np.zeros(shape)
                        if len(shape) == 5:
                            from_kb_l[:, :, :, :, ind] = aw.cpu().detach().numpy()
                        else:
                            from_kb_l[:, :, ind] = aw.cpu().detach().numpy()
                        from_kb_l = torch.from_numpy(from_kb_l)
                        from_kb.append(from_kb_l)
                loss_locals.append(copy.deepcopy(loss)) #training loss
                total_len += lens[idx]
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for i in range(len(w_glob)):
                        w_glob[i] = w_glob[i] * lens[idx]
                else:
                    for i in range(len(w_glob)):
                        w_glob[i] += w_local[i]*lens[idx]
                times_in.append(time.time() - start_in)
                pbar.update()
            loss_avg = sum(loss_locals) / len(loss_locals) #training loss
            loss_train.append(loss_avg)

            # get weighted average for global weights
            for i in range(len(w_glob)):
                w_glob[i] = torch.div(w_glob[i], total_len)
            if iter % args.round == args.round-1:
                for i in range(args.num_users):
                    if len(apprs[i].pre_weight['aw']) < task+1:
                        logging.info("client " + str(i) + " not train")
                        if not "Twitter" in args.dataset:
                            DatasetObj = DatasetConverter(dict_users_train[i][task], "cifar100")

                        tr_dataloaders = DataLoader(dict_users_train[i][task],batch_size=args.local_bs, shuffle=True)
                        apprs[i].set_sw(w_agg)
                        apprs[i].set_trData(tr_dataloaders)
                        LongLifeTrain(args, apprs[i], iter, from_kb, i, tb_writers[i])
                        gc.collect()
                if times == []:
                    times.append(max(times_in))
                else:
                    times.append(times[-1] + max(times_in))
                pbar.update()

            #TODO: Test accuracy evaluation
            client_test_accuracies = []
            client_test_losses = []
            client_test_bwts = []

            for client in range(args.num_users):
                logging.debug(f'client {client} test:')
                testdataset = dict_users_test[client]
                apprs[client].set_sw(w_glob) #set the model to the aggregated model
                client_avg_loss, client_avg_test_accuracy, client_backward_transfer = LongLifeTestCustom(args, apprs[client], task, testdataset, tb_writers[client], client)
                client_test_accuracies.append(client_avg_test_accuracy)
                client_test_losses.append(client_avg_loss)
                client_test_bwts.append(client_backward_transfer)
                client_data.append([client, iter, client_avg_test_accuracy, client_avg_loss])
            overall_avg_task_accuracies = sum(client_test_accuracies)/len(client_test_accuracies)
            overall_avg_task_losses = sum(client_test_losses)/len(client_test_losses)
            overall_backward_transfer = sum(client_test_bwts)/len(client_test_bwts)
            logging.info(f'[Round {iter}] Avg clients stats, accuracy: {100*overall_avg_task_accuracies:.2f}, loss: {overall_avg_task_losses:.2f}')

            #TODO: write to a file.

            average_accuracy =100*overall_avg_task_accuracies
            average_loss_clients =overall_avg_task_losses
            average_client_bwt = overall_backward_transfer
            writefile(f'{average_accuracy}, {average_loss_clients}, {average_client_bwt}')



            # logging.info("Avg accuracy across clients: {:.2f}%".format(100*overall_avg_task_accuracies))
            # logging.info("Avg loss across clients: {:.2f}".format(overall_avg_task_losses))
            round_data.append([iter, overall_avg_task_accuracies, overall_avg_task_losses])
            tb_writers['server'].add_scalar('avg test loss', overall_avg_task_losses, iter)
            tb_writers['server'].add_scalar('avg test accuracy', overall_avg_task_accuracies, iter)
            gc.collect()
                # for client in range(args.num_users):
                #     testdataset = DatasetConverter(dataset_test)
                    # client_avg_test_accuracy = LongLifeTestCustom(args, apprs[client], task, )
                # acc_test, loss_test = test_img_local_all_WEIT(apprs, args, dataset_test, dict_users_test,task,
                #                                          w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                #                                          dataset_train=dataset_train, dict_users_train=dict_users_train,
                #                                          return_all=False,write=write,device=args.device)

                # accs.append(acc_test)
                # # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
                # if iter != args.epochs:
                #     logging.info('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                #         iter, loss_avg, loss_test, acc_test))
                # else:
                #     # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                #     logging.info('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                #         loss_avg, loss_test, acc_test))
                # if iter >= args.epochs - 10 and iter != args.epochs:
                #     accs10 += acc_test / 10
                #
                # if iter >= args.epochs - 10 and iter != args.epochs:
                #     accs10_glob += acc_test / 10


        # logging.info('Average accuracy final 10 rounds: {}'.format(accs10))
        # if args.alg == 'fedavg' or args.alg == 'prox':
        #     logging.info('Average global accuracy final 10 rounds: {}'.format(accs10_glob))

        # Convert torch.device to string to make it serializable for JSON
        args.device = str(args.device)
        # Write all cli arguments to file for later analysis
        with open(args_config_file, 'w') as f:
            args_dict = vars(args)
            json.dump(args_dict, f, indent=2)

        # Write all the collected data to the csv files
        with open(client_data_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerows(client_data)

        with open(round_data_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerows(round_data)

        end = time.time()
        # logging.info(end - start)
        # logging.info(times)
        # logging.info(accs)
        # base_dir = './single/save/WEIT/accs_WEIT_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
        #             args.shard_per_user) + '_iterFinal' + '_frac_'+str(args.frac)+ '_model_'+args.model+'.csv'
        # user_save_path = base_dir
        # accs = np.array(accs)
        # accs = pd.DataFrame(accs, columns=['accs'])
        # accs.to_csv(base_dir, index=False)



        logging.info("done")
