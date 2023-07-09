import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Linear(hidden_size, 2*residual_channels)
            #nn.Conv1d(1, 2 * residual_channels, 1, padding=2, padding_mode="circular")
        self.output_projection = nn.Linear(residual_channels, 2*residual_channels) #nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-2)
        #print(diffusion_step)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = torch.permute(y,(0,1,3,2))
        y = y.reshape(x.shape[0]*x.shape[1], x.shape[3], -1)
        y = torch.reshape(self.dilated_conv(y).permute(0,2,1), (x.shape[0], x.shape[1], x.shape[2], -1)) + conditioner

        gate, filter = torch.chunk(y, 2, dim=3)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=3)
        return (x + residual) / math.sqrt(2.0), skip

class GraphBatchInputProjection(nn.Module):
    def __init__(self, feat_dim, residual_channels, bias = True, K = 2, normalization = 'sym'):
        super().__init__()
        self.input_gcn = ChebConv(
            in_channels=feat_dim,
            out_channels=residual_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

    def forward(self, x, edge_index, edge_weight, lambda_max = None):
        batches = x.shape[0]
        num_nodes = x.shape[2]
        l = x.shape[1]
        y = []
        for i in range(l):
            y.append(self.input_gcn(x[:,i,:].reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max).reshape(batches, num_nodes, -1))
        y = torch.stack(y, dim = 1)
        return y

class CondUpsampler(nn.Module):
    def __init__(self, cond_length, cond_feat):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, cond_feat // 2)
        self.linear2 = nn.Linear(cond_feat // 2, cond_feat)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class GraphEpsilonTheta(nn.Module):
    def __init__(
        self,
        num_nodes,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
        K=2,
        normalization  = "sym",
        bias = True
    ):
        super().__init__()
        self.input_projection = GraphBatchInputProjection(1, residual_channels, bias =bias, K = K, normalization=normalization )
        #nn.Conv1d(
        #    1, residual_channels, 1, padding=2, padding_mode="circular"
        #)
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            cond_feat=residual_hidden, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Linear(residual_channels, residual_channels) #nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Linear(residual_channels, 1) #nn.Conv1d(residual_channels, 1, 3)

        for lin in self.input_projection.input_gcn.lins:
            nn.init.kaiming_normal_(lin.weight)
        nn.init.zeros_(self.input_projection.input_gcn.bias)
        #nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond, edge_index, edge_weight):
        x = self.input_projection(inputs, edge_index, edge_weight)    # inputs: the noised data according to q functions, cond is parameter of dimensions eg 100
        x = F.leaky_relu(x, 0.4)             # need a graph layer:  in size: (batch x length) x 1 x nodes

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)  # do a graph layer.   In size: batch x Length x nodes x features
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x).squeeze(dim=3)
        return x
