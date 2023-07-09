import torch
from torch_geometric.nn import ChebConv


class GConvGRU(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        K: int,
        dropout_rate: float,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self.edge_index = None
        self.edge_weight = None
        self.dropout_rate = dropout_rate

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[2], self.hidden_size).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        #if self.edge_index is None:
        #    all_edge_index = []
        #    for i in range(batches):
        #        all_edge_index.append(edge_index + i * num_nodes)
        #    self.edge_index = torch.cat(all_edge_index, dim = 1)
        #if (self.edge_weight is None) and (edge_weight is not None):
        #    self.edge_weight = torch.tile(edge_weight, (batches,))

        Z = self.conv_x_z(X.reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H.reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        Z = torch.sigmoid(Z).reshape(batches, num_nodes, -1)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        #if self.edge_index is None:
        #    all_edge_index = []
        #    for i in range(batches):
        #        all_edge_index.append(edge_index + i * num_nodes)
        #    self.edge_index = torch.cat(all_edge_index, dim = 1)
        #if (self.edge_weight is None) and (edge_weight is not None):
        #    self.edge_weight = torch.tile(edge_weight, (batches,))

        R = self.conv_x_r(X.reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        R = R + self.conv_h_r(H.reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        R = torch.sigmoid(R).reshape(batches, num_nodes, -1)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        #if self.edge_index is None:
        #    all_edge_index = []
        #    for i in range(batches):
        #        all_edge_index.append(edge_index + i * num_nodes)
        #    self.edge_index = torch.cat(all_edge_index, dim = 1)
        #if (self.edge_weight is None) and (edge_weight is not None):
        #    self.edge_weight = torch.tile(edge_weight, (batches,))

        H_tilde = self.conv_x_h(X.reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h((H * R).reshape(batches * num_nodes, -1), edge_index.squeeze(), edge_weight.squeeze(), lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde).reshape(batches, num_nodes, -1)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features:  [Batch x SequenceLen x nodes x in_channels].
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden initial state matrix for all nodes. [Batch x out_channels]
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **States** *(PyTorch Float Tensor)* - All hidden state matrix for all nodes along the sequence.
        """
        L = X.shape[1]
        H = self._set_hidden_state(X, H)
        States = []
        for i in range(L):
            Z = self._calculate_update_gate(X[:,i,:,:], edge_index, edge_weight, H, lambda_max)
            R = self._calculate_reset_gate(X[:,i,:,:], edge_index, edge_weight, H, lambda_max)
            H_tilde = self._calculate_candidate_state(X[:,i,:,:], edge_index, edge_weight, H, R, lambda_max)
            H = self._calculate_hidden_state(Z, H, H_tilde)
            States.append(H)
        return torch.stack(States, dim=1), H