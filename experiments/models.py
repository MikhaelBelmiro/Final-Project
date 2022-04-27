import torch

from torch import nn
from torch.autograd import Variable 

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super().__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(2*hidden_size, 784)
        self.fc = nn.Linear(784, num_classes)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        out_1, (h1, c1) = self.lstm_1(x) #lstm with input, hidden, and internal state
        out_2, (h2, c2) = self.lstm_2(out_1)
        h2 = h2.view(-1, 2*self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(h2)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

class CLTFPModel(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        first_cnn_kernel, 
        second_cnn_kernel, 
        third_cnn_kernel,
        num_nodes,
        lstm_hidden_channels
        ):
        super().__init__()
        # lstm_hidden_layer
        self.first_cnn = nn.Conv1d(in_channels, out_channels, first_cnn_kernel)
        self.second_cnn = nn.Conv1d(out_channels, out_channels, second_cnn_kernel)
        self.third_cnn = nn.Conv1d(out_channels, out_channels, third_cnn_kernel)
        self.lstm = nn.LSTM(num_nodes, lstm_hidden_channels, batch_first=True)
        self.first_linear = nn.Linear(508, 497)
        self.second_linear = nn.Linear(30, 1)

    def forward(self, x):
        out_cnn = nn.ReLU()(self.first_cnn(x))
        out_cnn = nn.ReLU()(self.second_cnn(out_cnn))
        out_cnn = nn.ReLU()(self.third_cnn(out_cnn))

        out_lstm, (_, _) = self.lstm(x)

        out = torch.cat((out_cnn, out_lstm), 2)
        out = self.first_linear(out).permute(0, 2, 1)
        out = self.second_linear(out)
        return out
