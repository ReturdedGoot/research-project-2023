import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math

class NLP_LSTM(nn.Module):
    def __init__(self, embedding_dim =128, hidden_dim =128, vocab_size=1350279, num_classes=2):
        super(NLP_LSTM, self).__init__()

        #storing the hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes


        #creating the layers of the NN network:
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  #word embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)                  #LSTM layer
        self.hidden2output = nn.Linear(hidden_dim, num_classes)


    def forward(self, input_sentence, t):
        #note the input sentence is not a string but a list containing indexes to the words.
        embeddings = self.word_embeddings(input_sentence)
        # embeddings = embeddings.view(len(input_sentence), 1 , -1)
        # print(embeddings)
        # print(f"embeddings output shape: {embeddings.size()}")
        lstm_output, _ = self.lstm(embeddings)
        # lstm_output = lstm_output.view(len(input_sentence), -1)
        #we need to now transp
        lstm_output = torch.transpose(lstm_output, 0, 1)
        # print(f"lstm output shape: {lstm_output.size()}")
        # print(f'lstm last output shape: {lstm_output[-1].size()}')
        class_vector = self.hidden2output(lstm_output[-1])
        # print(f'linear layer shape: {class_vector.size()}')
        class_predictions = nn.functional.log_softmax(class_vector, dim=1) #converts vector into probabilities
        # print(f'class predictions shape: {class_predictions.size()}')

        #create an array of the exact size to demonstrate output
        # print(class_predictions.size())
        # result = np.zeros((class_predictions.size()[0], class_predictions.size()[1]))
        # print(f'result: {result.shape}')
        # result = torch.from_numpy(result)
        return class_predictions


    def prepare_words(self):
        self.word_to_index = {}




class NLP_LSTM_Weit(nn.Module):
    def __init__(self, embedding_dim =128, hidden_dim =128, vocab_size=15212, num_classes=2):
        super(NLP_LSTM_Weit, self).__init__()

        #storing the hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes


        #creating the layers of the NN network:

        self.word_embeddings = DecomposedEmbedding(vocab_size, embedding_dim)
        self.lstm = DecomposedLSTM(embedding_dim, hidden_dim, 1)
        self.hidden2output = DecomposedLinear(hidden_dim, num_classes)

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  #word embedding layer
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)                  #LSTM layer
        # self.hidden2output = nn.Linear(hidden_dim, num_classes)
        self.weight_keys = []
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)


    def forward(self, input_sentence, t):
        #note the input sentence is not a string but a list containing indexes to the words.
        # print('beginnign over  here')
        embeddings = self.word_embeddings(input_sentence)
        # embeddings = embeddings.view(len(input_sentence), 1 , -1)
        # print(embeddings)
        # print(f"embeddings output shape: {embeddings.size()}")
        # print(f'WE GOT HER E')
        lstm_output, _ = self.lstm(embeddings)
        # lstm_output = lstm_output.view(len(input_sentence), -1)
        #we need to now transp
        lstm_output = torch.transpose(lstm_output, 0, 1)
        # print(f"lstm output shape: {lstm_output.size()}")
        # print(f'lstm last output shape: {lstm_output[-1].size()}')
        # print(f'AMAZING')

        class_vector = self.hidden2output(lstm_output[-1])
        # print(f'linear layer shape: {class_vector.size()}')
        class_predictions = nn.functional.log_softmax(class_vector, dim=1) #converts vector into probabilities
        # print(f'class predictions shape: {class_predictions.size()}')

        return class_predictions

    def set_sw(self, glob_weights):
        self.word_embeddings.sw = Parameter(glob_weights[0])

        self.lstm.sw1 =  Parameter(glob_weights[1])
        self.lstm.sw2 = Parameter(glob_weights[2])

        self.hidden2output.sw = Parameter(glob_weights[3])




    def set_knowledge(self, t, from_kbs):
        #set the first knowledge and attention
        self.word_embeddings.set_atten(t, from_kbs[0].size(-1))
        self.word_embeddings.set_knlwledge(from_kbs[0])

        self.lstm.set_atten(t, from_kbs[1].size(-1), from_kbs[2].size(-1))
        self.lstm.set_knlwledge(from_kbs[1], from_kbs[2])

        self.hidden2output.set_atten(t, from_kbs[3].size(-1))
        self.hidden2output.set_knlwledge(from_kbs[3])


    def get_weights(self):
        weights = []

        #for each layer we get the weights
        w = self.word_embeddings.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        w1, w2 = self.lstm.get_weight()
        w1 = w1.detach()
        w2 = w2.detach()
        w1.requires_grad = False
        w2.requires_grad = False
        weights.append(w1)
        weights.append(w2)

        w = self.hidden2output.get_weight().detach()
        w.requires_grad = False
        weights.append(w)

        return weights

class DecomposedEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:

        super(DecomposedEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sw = Parameter(torch.Tensor(in_features, out_features))
        self.mask = Parameter(torch.Tensor(in_features))
        self.aw = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
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
        device = torch.device('cpu')




        # print(f'size of from_kb1 {self.from_kb.size()}')
        # print(f'sw1 : {sw.size()}')
        # print(f'mask1: {self.mask.size()}')

        # print(f'tranpose section: {(sw * m(self.mask)).transpose(0, -1).size()}')
        # print(f'aw1: {self.aw.size()}')
        # print(f'attent1: {torch.sum(self.atten * self.from_kb.to(device), dim=-1).size()}')




        weight = (sw * m(self.mask)).transpose(0, -1) + self.aw + torch.sum(self.atten * self.from_kb.to(device), dim=-1)
        if(device.type != 'cpu'):
            weight = weight.type(torch.cuda.FloatTensor)
        else:
            weight = weight.type(torch.FloatTensor)
        return weight

    def forward(self, input):
        weight = self.get_weight()
        #now we use the functional embeddign layer

        emb = nn.Embedding(1350279, 128)
        # print(f'weight shape: {emb.state_dict()["weight"].size()}')
        # print(f'Weights : {weight.size()}')
        # print(f'input: {input.size()}')
        return F.embedding(input, weight)




class DecomposedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,bias: bool = True) -> None:
        super(DecomposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sw = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.Tensor(out_features))
        self.aw = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
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
        return F.linear(input, weight, self.bias)

class DecomposedLSTM(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_layers: int, bias: bool = True) -> None:

        super(DecomposedLSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #For the input layer
        self.sw1 = Parameter(torch.Tensor(4*out_features, in_features))
        self.mask1 = Parameter(torch.Tensor(4*out_features))
        self.aw1 = Parameter(torch.Tensor(4*out_features, in_features))

        #For the hidden layers
        self.sw2 = Parameter(torch.Tensor(4*out_features, out_features))
        self.mask2 = Parameter(torch.Tensor(4*out_features))
        self.aw2 = Parameter(torch.Tensor(4* out_features, out_features))

        self.lstm = nn.LSTM(in_features, out_features, batch_first = True)

        if bias:
            self.bias1 = Parameter(torch.Tensor(4*out_features))
            self.bias2 = Parameter(torch.Tensor(4*out_features))

        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.sw1, a=math.sqrt(5))
        init.kaiming_uniform_(self.sw2, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw1, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw2, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in1, _ = init._calculate_fan_in_and_fan_out(self.sw1)
            fan_in2, _ = init._calculate_fan_in_and_fan_out(self.sw2)


            bound1 = 1 / math.sqrt(fan_in1)
            bound2 = 1 / math.sqrt(fan_in2)
            init.uniform_(self.bias1, -bound1, bound1)
            init.uniform_(self.mask1, -bound1, bound1)
            init.uniform_(self.bias2, -bound2, bound2)
            init.uniform_(self.mask2, -bound2, bound2)


    def set_atten(self,t,dim1, dim2):
        device = torch.device('cpu')
        if t==0:
            self.atten1 = Parameter(torch.zeros(dim1).to(device))
            self.atten1.requires_grad=False
            self.atten2 = Parameter(torch.zeros(dim2).to(device))
            self.atten2.requires_grad=False
        else:
            self.atten1 = Parameter(torch.rand(dim1).to(device))
            self.atten2 = Parameter(torch.rand(dim2).to(device))


    def set_knlwledge(self,from_kb1, from_kb2):
        self.from_kb1 = from_kb1
        self.from_kb2 = from_kb2


    def get_weight(self):
        m = nn.Sigmoid()
        sw1 = self.sw1.transpose(0, -1)
        sw2 = self.sw2.transpose(0, -1)

        # newmask = m(self.mask)
        # print(sw*newmask)
        device = torch.device('cpu')

        # print(f'size of from_kb1 {self.from_kb1.size()}')
        # print(f'sw1 : {sw1.size()}')
        # print(f'mask1: {self.mask1.size()}')

        # print(f'tranpose section: {(sw1 * m(self.mask1)).transpose(0, -1).size()}')
        # print(f'aw1: {self.aw1.size()}')
        # print(f'attent1: {torch.sum(self.atten1 * self.from_kb1.to(device), dim=-1).size()}')

        weight1 = (sw1 * m(self.mask1)).transpose(0, -1) + self.aw1 + torch.sum(self.atten1 * self.from_kb1.to(device), dim=-1)
        weight2 = (sw2 * m(self.mask2)).transpose(0, -1) + self.aw2 + torch.sum(self.atten2 * self.from_kb2.to(device), dim=-1)

        if(device.type != 'cpu'):
            weight1 = weight1.type(torch.cuda.FloatTensor)
            weight2 = weight2.type(torch.cuda.FloatTensor)
        else:
            weight1 = weight1.type(torch.FloatTensor)
            weight2 = weight2.type(torch.FloatTensor)
        return weight1, weight2

    # def _lstm_forward(self, input, weight):


    def forward(self, input):
        weight1, weight2 = self.get_weight()
        #we use the two weights and add them to the corresponding layer state_dict
        #load the current state_dict, modify it and then put it back into the layer
        state = self.lstm.state_dict()

        #modfication
        state["weight_ih_l0"] = weight1
        state['weight_hh_l0'] = weight2
        state['bias_ih_l0'] = self.bias1
        state['bias_hh_l0'] = self.bias2

        #load the state dict into the
        self.lstm.load_state_dict(state)

        #now we use the functional embeddign layer
        return self.lstm(input)



#Making the actual NLP LSTM but with the decomposed layers.
