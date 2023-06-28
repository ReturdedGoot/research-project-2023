#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import numpy as np
import torch

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[],dataname='cifar100'):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        if dataname == 'miniimagenet' or dataname == 'FC100' or dataname == 'tinyimagenet':
            label = torch.tensor(dataset.data[i]['label']).item()
        elif dataname == 'Corn50':
            label = torch.tensor(dataset.data['label'][i]).item()
        else:
            label = torch.tensor(dataset.data[i][1]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i) #add the element to the idxs dict for the corresponding fine label
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 20):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    testb = False
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    # for key, value in dict_users.items():
    #     x = np.unique(torch.tensor(dataset.targets)[value])
    #     test.append(value)
    # test = np.concatenate(test)
    return dict_users, rand_set_all

def specnoniid(dataset, num_users, num_task, partitioning, tasks):
    dict_users = {i: [] for i in range(num_users)}
    if partitioning != "unique":
        for task in tasks:
            task_dataset = dataset[task].data
            chunk_size = len(task_dataset)//num_users
            # remainder = len(task_dataset)%num_users #discard the remainder
            client_id = 0
            for i in range(0, len(task_dataset), chunk_size):
                dict_users[client_id].append(task_dataset[i:i + chunk_size])
                client_id = (client_id + 1)%num_users
            # if(remainder != 0):
            #     dict_users[0].append(task_dataset[-remainder:])
        if(partitioning == "balanced"):
            for i in range(num_users):
                dict_users[i] = rotate(dict_users[i], i)
    else: #if its unique partitioning
        id = 0
        task_id = 0
        while(len(dict_users[num_users-1]) < num_task):
            dict_users[id].append(dataset[task_id].data)
            id = (id + 1)%num_users
            task_id += 1


    return dict_users, None

def rotate(l, n):
    return l[-n:] + l[0:-n]

