import itertools
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle
import dgl
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import Config
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

def init():
    SEED = 2020
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(SEED)
    pass

def embedding(sequence_name):
    pssm_feature = np.load(Config.feature_path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Config.feature_path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_pssm_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "pssm/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_hmm_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "hmm/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Config.feature_path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)


def get_rsa_feature(sequence_name):
    rsa_feature = np.load(Config.feature_path + "rsa/" + sequence_name + '.npy')
    return rsa_feature.astype(np.float32)


def get_graph(sequence_name):
    norm = np.load(Config.graph_path + Config.center + 'normalize/' + sequence_name + '.npy')
    weight = np.load(Config.graph_path + Config.center + 'weight/' + sequence_name + '.npy')
    adj = np.load(Config.graph_path + Config.center + 'adj/' + sequence_name + '.npy')
    weight_no_norm = np.load(Config.graph_path + Config.center + 'weight_no_norm/' + sequence_name + '.npy')
    ffk_norm = np.load(Config.graph_path + Config.center + 'ffk/' + sequence_name + '.npy')
    return weight.astype(np.float32), norm.astype(np.float32), adj.astype(np.float32), weight_no_norm.astype(np.float32), ffk_norm.astype(np.float32)


def get_weight_graph(sequence_name):
    # test AlphaFold3
    if Config.AlphaFold3_pred:
        weight = np.load(
            Config.graph_path + Config.center + 'AlphaFold3_weight/' + sequence_name + '.npy')
    else:
        weight = np.load(
            Config.graph_path + Config.center + 'PPIS/weight_' + (str)(Config.MAP_CUTOFF) + '/' + sequence_name + '.npy')
    return weight.astype(np.float32)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

# weight graph
def edge_weight(dist):
    dist = torch.Tensor(dist)
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    graph = softmax(1./(torch.log(torch.log(dist+2))))
    graph[matrix>Config.MAP_CUTOFF] = 0
    return graph


def cal_edges(sequence_name, radius=Config.MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Config.feature_path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int64)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return radius_index_list, norm_matrix


def cal_Ligand_adj(sequence_name):
    dist_matrix = np.load("./Graph/SC/DNA/weight_14/" + sequence_name + ".npy")
    return dist_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix


def attpresite_graph_collate(samples):
    sequence_name, sequence, label, node_features, adj_matrix = map(list, zip(*samples))
    label = torch.Tensor(label)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix[0])
    return sequence_name, sequence, label, node_features, adj_matrix


class AttPreSiteProDataset(Dataset):
    def __init__(self, dataframe, radius=Config.MAP_CUTOFF, dist=Config.DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        rsa_features = get_rsa_feature(sequence_name)
        res_atom_features = get_res_atom_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features, res_atom_features, rsa_features], axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        adj_matrix = get_weight_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        return sequence_name, sequence, label, node_features, adj_matrix

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features=None):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        if edge_features is not None:
            G.edata['ex'] = torch.tensor(edge_features)


class AttPreSiteLigandDataset(Dataset):
    def __init__(self, dataframe, radius=Config.MAP_CUTOFF, dist=Config.DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl', feature_path=''):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.features = pickle.load(open(feature_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        if sequence_name[-1].islower():
            if len(sequence_name.split('_')[1]) == 1:
                sequence_name += sequence_name[-1]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)
        node_features = self.features[sequence_name]
        node_features = torch.from_numpy(node_features)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        adj_matrix = cal_Ligand_adj(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        return sequence_name, sequence, label, node_features, adj_matrix

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features=None):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        if edge_features is not None:
            G.edata['ex'] = torch.tensor(edge_features)