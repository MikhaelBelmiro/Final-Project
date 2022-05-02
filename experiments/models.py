import torch

from torch import nn
from torch.autograd import Variable 

class LSTMModel(nn.Module):
    def __init__(
        self, 
        in_timesteps, 
        out_timesteps, 
        num_nodes, 
        hidden_size, 
        num_layers
        ):
        super().__init__()
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.lstm = nn.LSTM(input_size=num_nodes, hidden_size=hidden_size,
                            num_layers=out_timesteps, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 784)
        self.fc = nn.Linear(784, num_nodes)
    
    def forward(self,x):
        _, (h, _) = self.lstm(x)
        h = h.permute(1, 0, 2)
        out = nn.ReLU()(h)
        out = self.fc_1(out) #first Dense
        out = nn.ReLU()(out) #relu
        out = self.fc(out) #Final Output
        return out

class CLTFPModel(nn.Module):
    def __init__(
        self,
        in_timesteps,
        out_timesteps,
        out_cnn_channels, 
        first_cnn_kernel, 
        second_cnn_kernel, 
        third_cnn_kernel,
        num_nodes,
        lstm_hidden_channels
        ):
        super().__init__()
        self.first_cnn = nn.Conv1d(in_timesteps, out_cnn_channels, first_cnn_kernel)
        self.second_cnn = nn.Conv1d(out_cnn_channels, out_cnn_channels, second_cnn_kernel)
        self.third_cnn = nn.Conv1d(out_cnn_channels, in_timesteps, third_cnn_kernel)
        self.lstm = nn.LSTM(num_nodes, lstm_hidden_channels, batch_first=True)
        self.first_linear = nn.Linear(num_nodes-(first_cnn_kernel-1)-(second_cnn_kernel-1)-(third_cnn_kernel-1)+lstm_hidden_channels, num_nodes)
        self.second_linear = nn.Linear(in_timesteps, out_timesteps)

    def forward(self, x):
        x = x.squeeze()

        out_cnn = nn.ReLU()(self.first_cnn(x))
        out_cnn = nn.ReLU()(self.second_cnn(out_cnn))
        out_cnn = nn.ReLU()(self.third_cnn(out_cnn))

        out_lstm, (_, _) = self.lstm(x)

        print(out_cnn.shape)
        print(out_lstm.shape)

        out = torch.cat((out_cnn, out_lstm), 2)
        out = self.first_linear(out).permute(0, 2, 1)
        out = self.second_linear(out)
        return out.permute(0, 2, 1)
