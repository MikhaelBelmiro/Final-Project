import torch
from torch import nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Kt):
        '''
        :param in_channels: number of input channels. For
        first layer of TemporalLayer should be num_features

        :param out_channels: number of output channels. Typically
        out_channels < in_channels, however it is arbitrary

        :param Kt: temporal filter size. Arbitrary value.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Kt = Kt

        self.conv = nn.Conv2d(in_channels=self.in_channels, 
                              out_channels=2*self.out_channels,
                              kernel_size=(self.Kt, 1))

    def forward(self, X):
        '''
        :param X: input data of shape (batch_size, num_timesteps, 
        num_nodes, node_features)
        '''
        X = X.permute(0, 3, 1, 2) #(batch_size, node_features, num_timesteps, num_nodes)
        temp = self.conv(X)
        print('Temp shape', temp.shape)
        P = temp[:, :self.out_channels, :, :]
        print('P shape', P.shape)
        Q = temp[:, self.out_channels:, :, :]
        print('Q shape', Q.shape)
        out = P*nn.Sigmoid()(Q)
        out = out.permute(0, 2, 3, 1)
        return out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        :param in_channels: number of input channels. For
        first layer of TemporalLayer should be num_features

        :param out_channels: number of output channels. Typically
        out_channels < in_channels, however it is arbitrary

        :param Kt: temporal filter size. Arbitrary value.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Theta = nn.Parameters(torch.FloatTensor(self.in_channels, 
                                                     self.out_channels))
        stdv = 1. / math.sqrt(self.Theta1.shape[1])                                             
        self.Theta.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        '''
        :param X: input of shape (batch_size, num_timesteps, 
        num_nodes, node_features)
        :param A_hat: Normalized adj_matrix (D^(-1/2) W D^(-1/2)).
        The size should be (num_nodes, num_nodes)
        '''
        temp = torch.einsum('ij,kljm->klim', [A_hat, X])
        out = nn.ReLU(torch.einsum('ijkl,lm->ijkm', [temp, self.Theta]))
        return out

class STConvBlock(nn.Module):
    def __init__(self, A_hat, temporal_in_channels, spatial_in_channels,
                 spatial_out_channels, temporal_out_channels, Kt):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Kt_1, self.Kt_2 = Kt
        self.A_hat = A_hat

        self.first_temporal = TemporalBlock(temporal_in_channels, spatial_in_channels, self.Kt_1)
        self.spatial = SpatialBlock(spatial_in_channels, spatial_out_channels)
        self.second_temporal = TemporalBlock(spatial_out_channels, temporal_out_channels, self.Kt_1)

    def forward(X):
        t1_out = self.first_temporal(X)
        spat_out = nn.ReLU(self.spatial(t1_out, self.A_hat))
        t2_out = self.second_temporal(spat_out)
        out = nn.BatchNorm2d(t2_out.shape[1])
        return t2_out

class STGCN(nn.Module):
    def __init__(self, adj_matrix, num_features, in_timesteps, out_timesteps):
        self.adj_matrix = adj_matrix
        self.A_hat = normalize_matrix(self.adj_matrix)

        self.block1 = STConvBlock(self.A_hat, num_features, 64, 16, 64, 3)
        self.block2 = STConvBlock(self.A_hat, 64, 64, 16, 64, 3)
        self.last_temporal = TemporalBlock(64, 64, 3)
        self.fc = nn.Linear(1, out_timesteps)

    def forward(X):
        out1 = self.block1(X)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        out = self.fc(out3)
        return out
