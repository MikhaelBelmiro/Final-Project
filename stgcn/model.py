from torch import nn
from torch_geometric_temporal import STConv, TemporalConv

class STGCN(nn.Module):
    def __init__(self, in_timesteps, out_timesteps, num_nodes, in_channels, hidden_channels, out_channels, kernel_size, K):
        super().__init__()
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.first_stconv = STConv(num_nodes, 1, hidden_channels, out_channels, kernel_size, K)
        self.second_stconv = STConv(num_nodes, out_channels, hidden_channels, out_channels, kernel_size, K)
        self.final_conv = TemporalConv(out_channels, out_channels, self.in_timesteps-4*(kernel_size-1))
        self.linear =  nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        out_1 = self.first_stconv(x, edge_index, edge_weight)
        out_2 = self.second_stconv(out_1, edge_index, edge_weight)
        out_3 = self.final_conv(out_2).squeeze()
        out = self.linear(out_3)
        return out