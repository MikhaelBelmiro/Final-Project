from torch import nn
from torch_geometric_temporal import STConv

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, kernel_size, K):
        super().__init__()
        self.first_stconv = STConv(num_nodes, 1, hidden_channels, out_channels, kernel_size, K)
        self.second_stconv = STConv(num_nodes, out_channels, hidden_channels, out_channels, kernel_size, K)
        self.final_conv = None
        self.linear =  nn.Linear(2,2)

    def forward(self, x, edge_index, edge_weight):
        out_1 = self.first_stconv(x, edge_index, edge_weight)
        print(out_1.shape)
        out_2 = self.second_stconv(out_1, edge_index, edge_weight)
        print(out_2.shape)
        out_3 = self.final_conv(out_2)
        out = self.linear(out_3)
        return out