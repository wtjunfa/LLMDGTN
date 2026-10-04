import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F


class EGCN(torch.nn.Module):
    def __init__(self, activation, device='cuda', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})
        emb_dim = 64
        # Define feature dimensions (2 layers)
        feats = [emb_dim,
                 emb_dim * 2,
                 emb_dim]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({
                'in_feats': feats[i - 1],   # Input feature dimension
                'out_feats': feats[i],     # Output feature dimension
                'activation': activation   # Activation function
            })

            grcu_i = GRCU(GRCU_args)  # EvolveGCN layer
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list):
        # Get features from the last layer
        node_feats = Nodes_list[-1]

        # Forward pass through all GRCU layers
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)

        out = Nodes_list
        if self.skipfeats:
            # Use node_feats.to_dense() if input is 2-hot encoded
            out = torch.cat((out, node_feats), dim=1)
        return out


class GRCU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        # RNN cell for weight matrix evolution
        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        # GCN initial weights [in_feats, out_feats]
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list):
        GCN_weights = self.GCN_init_weights  # Initialize GCN weights
        out_seq = []
        # Iterate over adjacency matrices at each time step
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]  # Node features at current time step

            # Evolve GCN weights using GRU
            GCN_weights = self.evolve_weights(GCN_weights)
            # GCN propagation
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # GRU update gate Zt
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        # GRU reset gate Rt
        self.reset = mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid())

        # Candidate hidden state Ht_hat
        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

    def forward(self, prev_Q):
        # prev_Q: previous GCN weight matrix
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        # Update hidden state
        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # Weight matrices for GRU gate
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(
            self.W.matmul(x) +
            self.U.matmul(hidden) +
            self.bias
        )
        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        # Top-k selection
        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # Transpose the output
        return out.t()