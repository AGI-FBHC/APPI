import numpy as np
from layers import *
from config import Config

import warnings
warnings.filterwarnings("ignore")

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttPreSite(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, num_layer, K, dropout=.1,
                 excitation_rate=1., layernorm=True, bias=True, activate='tanh', p=0.6):
        super(AttPreSite, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.num_layers = num_layer
        self.K = K
        self.use_lnorm = layernorm
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.decoder = nn.Linear(hidden, out_dim)

        L = num_layer
        for i in range(L):
            in_c = in_dim if i == 0 else hidden
            out_c = hidden if i < L - 1 else hidden
            self.layers.append(
                AttPreSiteLayer(
                    in_dim=in_c, out_dim=out_c, K=K, dropout=dropout, excitation_rate=excitation_rate, bias=bias,
                    activate=activate, type='no'
                )
            )
            if i < L - 1 and layernorm:
                self.layernorms.append(nn.LayerNorm(out_c, elementwise_affine=True))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,
                                                                    min_lr=1e-6)

    def forward(self, n_feat, graph=None, adj_matrix=None, edge_index=None):
        if edge_index is not None:
            edge_index = edge_index.squeeze(0)
        n_feat = n_feat.view([n_feat.shape[0] * n_feat.shape[1], n_feat.shape[2]])
        for i in range(self.num_layers - 1):
            n_feat = self.layers[i](n_feat, Config.alpha, Config.LAMBDA, adj_matrix=adj_matrix, graph=graph, edge_index=edge_index)
            if self.use_lnorm:
                n_feat = self.layernorms[i](n_feat)
            n_feat = F.relu6(n_feat)
        n_feat = self.layers[-1](n_feat, Config.alpha, Config.LAMBDA, adj_matrix=adj_matrix, graph=graph, edge_index=edge_index)
        layer_inner = F.dropout(n_feat, self.dropout, training=self.training)
        layer_inner = self.decoder(layer_inner)
        return layer_inner