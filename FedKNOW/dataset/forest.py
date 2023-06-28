import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split




class ForestDataset(Dataset):

    def __init__ (self, csv_file, num_classes = 7, task_num=1):
        dataset = pd.read_csv(csv_file) #Temporary file can be removed at a later date

        self.samples = dataset.drop(['Cover_Type'], axis = 1).to_numpy() #make sure to drop these
        self.labels = dataset['Cover_Type'].to_numpy()
        self.labels = [x-1 for x in self.labels]


        self.num_classes = num_classes
        self.max_classes = 7
        self.task_num = task_num

        #We do not need to perform any changes to the data. Can be used directly in an MLP setting.

    def __len__(self):
        return len(self.labels)

        #make sure to drop the first 2 columns

    def __getitem__(self, index):
        #get row of data
        sample = self.samples[index]
        label = self.labels[index]

        sample = sample.astype(np.float32)
        # label = label.astype(np.float)

        return sample, label

    def train_test(self, train_size, test_size):
        """
        Generates a train and test dataset of the given sizes.
        the values of "train_size + test_size < 1"
        Classes have an even distribution in both sets.

        input: train_size -> int
        input: test_size -> int

        """

        train_indices, test_indices, _, _ = train_test_split(range(len(self)), self.labels, stratify=self.labels, train_size=train_size, test_size=test_size)

        #make it into the subsets of the original dataset.
        training_data = torch.utils.data.Subset(self, train_indices)
        testing_data = torch.utils.data.Subset(self, test_indices)
        return training_data, testing_data


    def data_split_task_incremental(self, train_size, test_size):
        #We first split the dataset into subsets of indicies.
        num_classes = self.num_classes
        dataset_indices = []
        for i in range(self.max_classes):
            dataset_indices.append([])
        #should follow the same structure as the dictionary

        #we iterate through the dataset.
        for i in range(len(self.labels)):
            #for each index we check the label and add its index to the corresponding dataset
            dataset_indices[self.labels[i]].append(i)

        string = f"class split : ({len(dataset_indices)}, ("
        for arr in dataset_indices:
            string+= f"{len(arr)}, "
        string+= "))"
        print(string)
        # print(f'Dataset: {dataset_indices}')


        div = num_classes // self.task_num
        remainder = num_classes % self.task_num
        #now we make sure that each task gets atleast 1 class.
        tasks = []
        for i in range(self.task_num):
            tasks.append([])

        #for i in tasks
        class_index = 0
        is_first = True

        for i in range(self.task_num):
            #for each class we add it to the corresponding list
            count = div
            if is_first:
                is_first = False
                count+= remainder
            while count > 0:
                # tasks[i].append(class_index) #Test method needs to be changed
                tasks[i].extend(dataset_indices[class_index])
                class_index+=1
                count -= 1

        # print(f'Tasks structure: {tasks}')
        string = f"task split : ({len(tasks)}, ("
        for arr in tasks:
            string+= f"{len(arr)}, "
        string+= "))"
        print(string)

        #now we need to make sure to randomly split each subtask evenly for each task
        training_data = []
        testing_data = []
        all_labels = np.array(self.labels)
        for task in tasks:
            #get corresponding labels
            labels = all_labels[task]
            #now we make the train and task split
            train_indices, test_indices, _, _ = train_test_split(task, labels, train_size=train_size, test_size=test_size, stratify=labels)

            train_set = torch.utils.data.Subset(self, train_indices)
            test_set = torch.utils.data.Subset(self, test_indices)
            training_data.append(train_set)
            testing_data.append(test_set)

        print(f'Training Data lengths: {[len(x) for x in training_data]}')
        print(f'Testing Data lengths: {[len(x) for x in testing_data]}')

        return training_data, testing_data




    def data_split(self, dataset):
        #here we want to split the dataset evenly for each tasks.
        #split the dataset into multiple datasets
        length = len(dataset)//self.task_num
        lengths = [length for i in range(self.task_num)]
        if len(dataset) != sum(lengths):
            #add some more length to the lengths
            difference = len(dataset) - sum(lengths)
            lengths[0] = lengths[0]+difference
        # print(f'Dataset length: {len(self.tr_dataset)}, sum lengths: {sum(lengths)}')
        datasets = torch.utils.data.random_split(dataset, lengths )


        #do the same for the test
        # dataloaders = [DataLoader(dataset, self.batch_size) for dataset in datasets]
        # self.tr_dataloaders = dataloaders
        # self.task_lengths = lengths
        # print(f' dataloaders: {len(dataloaders)}')
        return datasets


    def getTaskDataset(self, train_size, test_size, num_users):
        num_classes = self.num_classes

        #here we compute an array of datasets to be used for each task. or do we?
        #we compute the dataset splits

        if self.task_num <= num_classes:
            #then we can safely split the training data and testing data in
            #the task incremental learning format.
            train_datasets, test_datasets = self.data_split_task_incremental(train_size, test_size)
        else:
            train_dataset, test_dataset = self.train_test(train_size, test_size)
            #now we run it through the data splitter
            train_datasets = self.data_split(train_dataset)
            test_datasets = self.data_split(test_dataset)



        #now we 2 dictionaries, 1 for the test set and 1 for the train set.
        #note we essentially use the same tasks for each user. instead.
        dict_users_train = {}
        dict_users_test = {}
        for i in range(num_users):
            dict_users_train[i] = train_datasets
            dict_users_test[i] = test_datasets
        return dict_users_train, dict_users_test