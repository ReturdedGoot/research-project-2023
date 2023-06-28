import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

#Creating a custom dataset class

class EmotionSentiment(Dataset):
    """ Twitter sentiment dataset. 0 means negative and 4 means positive."""

    def __init__(self, csv_file, valid_size=0.01, task_num=1,  transform = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable): optional transform to be applied on the sampels


        """
        print(f"Loading data from: {csv_file}")
        self.emotion_dataset = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
        #save dataframe
        # print('Writing to pickel')
        # self.twitter_dataset.to_pickle('twitter_data.pkl')
        self.class_to_index ={'sadness':0, 'joy':1, 'fear':2, 'anger': 3, 'surprise': 4,  'love': 5}

        labels = self.emotion_dataset.iloc[:, 1]
        #turning all 4's into 1's for classfication and sentiment analysis later.
        self.labels = [self.class_to_index[x] for x in labels]
        self.text = self.emotion_dataset.iloc[:, 0]
        self.transform = transform
        self.valid_size = valid_size
        self.total_size = len(self.text)
        self.task_num = task_num

        self.sentences = []
        self.targets = []
        print("Succesfully loaded data")
        self.word_to_index, self.class_to_index = self.generate_words()
        self.convert_to_indices()

        #we need to ensure that the dataset is representative of what we want.

        #here we only select a portion of the data
        # num_train = len(text)
        # #select portion of indices randomly
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # split = int(np.floor(valid_size * num_train))
        # # we select the split.
        # self.text = self.text[:split]
        # self.labels = self.labels[:split]

    def train_test(self, train_size, test_size):
        """
        Generates a train and test dataset of the given sizes.
        the values of "train_size + test_size < 1"
        Classes have an even distribution in both sets.

        input: train_size -> int
        input: test_size -> int

        """

        train_indices, test_indices, _, _ = train_test_split(range(len(self)+1), self.labels, stratify=self.labels, train_size=train_size, test_size=test_size)

        #make it into the subsets of the original dataset.
        training_data = torch.utils.data.Subset(self, train_indices)
        testing_data = torch.utils.data.Subset(self, test_indices)
        return training_data, testing_data



    def data_split_task_incremental(self, train_size, test_size):
        #We first split the dataset into subsets of indicies.
        num_classes = len(self.class_to_index)

        dataset_indices = []
        for i in range(num_classes):
            dataset_indices.append([])
        #should follow the same structure as the dictionary

        #we iterate through the dataset.
        for i in range(len(self.labels)):
            #for each index we check the label and add its index to the corresponding dataset
            dataset_indices[self.labels[i]].append(i)

        # string = f"class split : ({len(dataset_indices)}, ("
        # for arr in dataset_indices:
        #     string+= f"{len(arr)}, "
        # string+= "))"
        # print(string)
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
        # string = f"task split : ({len(tasks)}, ("
        # for arr in tasks:
        #     string+= f"{len(arr)}, "
        # string+= "))"
        # print(string)

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


        #Printing datasets size



        return training_data, testing_data

    def __len__(self):
        return len(self.text)-1

    def generate_words(self):
        """ This function generates the words to index dictionary and the class to index dictionary"""
        word_to_index = {}
        print(f"Generating vocabulary indices")

        #find all words in dataset
        for i in range(0, len(self)+1):
            text = self.text[i]
            # print(batch)
            # print(batch['text'].split())
            # try:
            sentence = text.split()
            # except:
            #     print("batch")
            for word in  sentence:
                if word not in word_to_index:
                    try:
                        word_to_index[word] = len(word_to_index)
                    except:
                        print(f"failed at current word: {word}, word_to_index size {len(word_to_index)}")
        print(f"Finished creating vocabulary")
        self.word_to_index = word_to_index
        return word_to_index, self.class_to_index


    def convert_to_indices(self):
        #we convert all sentences into indicies and attempt to pad the sentences with 0's
        print(f'Converting sentences to index arrays')
        for sentence in self.text:
            # try:
            word_indices = torch.tensor([self.word_to_index[word] for word in sentence.split() if word != ' '], dtype=torch.long)
            # except:
                # print(f'Failed on this sentence: {sentence}')
            #now we add this to the list of all

            self.sentences.append(word_indices)

        print(f'converting labels to index arrays')
        #after converting all sentences to indices we want to convert all labels into integers.
        for label in self.labels:
            # labelTensor = torch.tensor(self.class_to_index[label], dtype=torch.long)
            self.targets.append(label)

        #we ensure all sentences are of equal length by padding with 0's
        #convert everything to a tensor?
        self.sentences = nn.utils.rnn.pad_sequence(self.sentences)

        #reshape the sentences tensor to be of the ccorrect size.
        self.sentences = self.sentences.reshape(self.sentences.size()[1], -1)
        self.targets = torch.tensor(self.targets, dtype=torch.long )
        print(f'succesfully completed conversion')
        print(f"Shape of sentences : {self.sentences.size()}, Shape of targets: {self.targets.size()}")


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
        num_classes = len(self.class_to_index)

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


    def convert_sentence(self, sentence):

        word_indices = None
        """Converts a sentence into an array of words with each word being an index"""

        try:
            word_indices = [self.word_to_index[word] for word in sentence.split() if word != ' ']
        except:
            print(f"sentence: {sentence}")
        return word_indices

    # def prepare_sequence(self, sentence, to_index):
    #     indexes = [to_index[word] for word in sentence.split() if word != ' ' ]
    #     return torch.tensor(indexes, dtype = torch.long)
    #     def __len__(self):
    #         return len(self.text)-1


    # def prepare_sequence_batch(self, sentences, to_index):
    #     #we can essentially use the previous one.
    #     indexes = [prepare_sequence(x, to_index) for x in sentences]
    #     #make rnn
    #     #we have to pad the indexes with 0's to ensure they are the same size for a Tensor
    #     #Note a tensor does not accept variable lenght elements. So we must make sureeach element has the same dimensions.
    #     result = nn.utils.rnn.pad_sequence(indexes)
    #     return result


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        #maybe we can modify the get item.
        text = self.sentences[index]
        label = self.targets[index]

        # sample
        # sentence = self.convert_sentence(text)
        # target = self.class_to_index[label]

        # sample = {"text" : text, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return text, label
