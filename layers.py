import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from torch_geometric.nn import GCNConv, GATConv


class AGATLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2, use_efeats=True):
        super(AGATLayer, self).__init__()
        self.use_efeats = use_efeats
        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
        if self.use_efeats:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
            self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
            self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        else:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.use_efeats:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

    def edge_attention(self, edges):
        if self.use_efeats:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
            a = self.attn_fc(z2)
        else:
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)

        if self.use_efeats:
            ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
            return {'e': F.leaky_relu(a), 'ez': ez}
        else:
            return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.use_efeats:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        if self.use_efeats:
            h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        z = self.fc(h)
        g.ndata['z'] = z
        if self.use_efeats:
            ex = self.fc_edge_for_att_calc(e)
            g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')


class BaseModule(nn.Module):
    def __init__(self, in_features, out_features, type=None, edge_feature=0):
        super(BaseModule, self).__init__()
        self.agat = AGATLayer(in_features, out_features)
        self.AGAT = GATConv(in_features, out_features, num_heads=1)
        self.egret = EGRETLayer(in_features, out_features)
        self.esgret = ESGRETLayer(in_features, out_features)
        self.edge_agg = EdgeAgg(in_features)
        self.trans = nn.Linear(in_features * 2, out_features, bias=False)
        if type == 'gcn':
            self.gcn = GraphConv(in_features, in_features, norm='both', weight=False, bias=False)
        elif type == 'gat':
            self.gat = GATConv(in_features, out_features, num_heads=1)
        elif type == 'edge_index':
            self.edge_mlp = nn.Sequential(
                nn.Linear(out_features * 2 + edge_feature, out_features),
                nn.SiLU(),
                nn.Linear(out_features, out_features),
                nn.SiLU())
            self.node_mlp = nn.Sequential(
                nn.Linear(out_features + out_features, out_features),
                nn.SiLU(),
                nn.Linear(out_features, out_features))
            self.att_mlp = nn.Sequential(
                nn.Linear(out_features, 1),
                nn.Sigmoid())
        self.in_features = 2 * in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self, input, h0, lamda, alpha, l, type='no', adj_matrix=None, graph=None, efeats=None, edge_index=None):
        theta = min(1, math.log(lamda / l + 1))
        if type == 'no':
            hi = torch.sparse.mm(adj_matrix, input)
        elif type == 'gcn':
            hi = self.gcn(graph, input)
        elif type == 'edge_index':
            row, col = edge_index
            edge_agg = torch.cat([input[row], input[col]], dim=1)
            out = self.edge_mlp(edge_agg)
            att_val = self.att_mlp(out)
            out = out * att_val
            agg = self.unsorted_segment_sum(out, row, input.shape[0])
            agg = torch.cat([agg, input], dim=1)
            hi = self.node_mlp(agg)
        elif type == 'gat':
            hi = self.gat(graph, input)
            hi = torch.squeeze(hi)
        support = torch.cat([hi, h0], 1)
        r = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        output = output + input
        return output


class AttPreSiteLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K, dropout, excitation_rate=1.,
                 type="no", bias=True, activate='tanh'):
        super(AttPreSiteLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.dropout = dropout
        self.type = type
        self.act_fn = nn.ReLU()
        self.gc = nn.ModuleList()
        self.st = nn.Linear(in_dim, out_dim)
        for _ in range(K):
            self.gc.append(BaseModule(out_dim, out_dim, type=type))
        self.seblock = SEAggLayer(out_dim, K, excitation_rate=excitation_rate, bias=bias, activate=activate)

    def forward(self, n_feat, alpha, lamda, adj_matrix=None, graph=None, edge_index=None):
        n_feat = F.dropout(n_feat, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.st(n_feat))
        aggr_results = [layer_inner]
        for i in range(self.K):
            layer_inner = aggr_results[-1]
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.gc[i](input=layer_inner, h0=aggr_results[0], lamda=lamda, alpha=alpha, l=i + 1,
                                     graph=graph, adj_matrix=adj_matrix, type=self.type, edge_index=edge_index)
            layer_inner = self.act_fn(layer_inner)
            aggr_results.append(layer_inner)
        out = self.seblock(aggr_results)
        return out


class SEAggLayer(nn.Module):
    def __init__(self, hidden, K, excitation_rate=1.0, bias=True, activate='tanh'):
        super(SEAggLayer, self).__init__()
        e_chs = int((K + 1) * excitation_rate - 1) + 1
        self.e_weight1 = Parameter(torch.Tensor(K + 1, e_chs))
        self.e_weight2 = Parameter(torch.Tensor(e_chs, K + 1))
        self.e_att = Parameter(torch.Tensor(hidden, 1))
        self.bias = bias
        self.K = K
        self.activate = activate
        if bias is True:
            self.e_bias1 = Parameter(torch.Tensor(e_chs))
            self.e_bias2 = Parameter(torch.Tensor(K + 1))
        else:
            self.register_parameter("e_bias1", None)
            self.register_parameter("e_bias2", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.e_weight1)
        init.xavier_uniform_(self.e_weight2)
        init.xavier_uniform_(self.e_att)
        if self.bias is True:
            init.zeros_(self.e_bias1)
            init.zeros_(self.e_bias2)

    def forward(self, aggr_results):
        stack_result = torch.stack(aggr_results, dim=-1)  # N x in_dim x (K+1)
        squeeze_result = (stack_result * self.e_att).sum(dim=-2).squeeze()
        squeeze_result = F.normalize(squeeze_result, dim=-1)
        excitation_result = torch.matmul(squeeze_result, self.e_weight1)
        if self.e_bias1 is not None:
            excitation_result = excitation_result + self.e_bias1
        excitation_result = F.relu6(excitation_result)
        excitation_result = torch.matmul(excitation_result, self.e_weight2)
        if self.e_bias2 is not None:
            excitation_result = excitation_result + self.e_bias2
        if self.activate == 'tanh':
            excitation_result = torch.tanh(excitation_result).view(-1, 1, self.K + 1)
        else:
            excitation_result = torch.softmax(excitation_result, dim=1).view(-1, 1, self.K + 1)
        out = (stack_result * excitation_result).sum(dim=-1)
        return out


class EGRETLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2):
        super(EGRETLayer, self).__init__()
        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
        self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
        self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        a = self.attn_fc(z2)
        ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
        return {'e': F.leaky_relu(a), 'ez': ez}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        z = self.fc(h)
        g.ndata['z'] = z
        ex = self.fc_edge_for_att_calc(e)
        g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class EdgeAgg(nn.Module):
    def __init__(self, nfeats_out_dim, edge_dim=2):
        super(EdgeAgg, self).__init__()
        self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
        self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
        self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        a = self.attn_fc(z2)
        ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
        return {'e': F.leaky_relu(a), 'ez': ez}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'ez': edges.data['ez']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        # print(h.shape)
        g.ndata['z'] = h
        ex = self.fc_edge_for_att_calc(e)
        g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class ESGRETLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2, use_efeats=True):
        super(ESGRETLayer, self).__init__()
        self.use_efeats = use_efeats
        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
        self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
        self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        self.attn_edge = nn.Parameter(torch.FloatTensor(size=(1, 1, nfeats_out_dim)))
        self.fc_edge = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

    def edge_attention(self, edges):
        h_edge = edges.data['ex']
        feat_edge = self.fc_edge(h_edge).view(-1, 1, self.nfeats_out_dim)
        ee = (feat_edge * self.attn_edge).sum(dim=-1)

        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        a = self.attn_fc(z2)

        a += ee

        ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
        return {'e': F.leaky_relu(a), 'ez': ez}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        z = self.fc(h)
        g.ndata['z'] = z
        ex = self.fc_edge_for_att_calc(e)
        g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')