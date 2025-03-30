import os
import numpy as np
import pickle
import torch
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def edge_weight(dist):
    dist = torch.Tensor(dist)
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    graph = softmax(1./(torch.log(torch.log(dist+2))))
    graph[matrix>14] = 0
    return graph

def cal_Ligand_adj():
    with open('./Dataset/DNA_Test_129.pkl', 'rb') as f:
        data = pickle.load(f)
        for protein in tqdm(data):
            if protein[-1].islower():
                if len(protein.split('_')[1]) == 1:
                    protein += protein[-1]
            dist_matrix = np.load("./Feature/distance_map_SC/DNA/" + protein + ".npy")
            adj = normalize(edge_weight(dist_matrix).detach().numpy())
            save_path = './Graph/SC/DNA/weight_14/' + protein
            np.save(save_path, adj)
    pass
