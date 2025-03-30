import math
import freesasa
import os
import numpy as np
from tqdm import tqdm

path = "./Feature/rsa/"
pdb_path = './Dataset/pdb/'


def get_RSA(protein, chains):
    RSA_dict = {}
    structure = freesasa.Structure(pdb_path + protein + '.pdb')
    result = freesasa.calc(structure, freesasa.Parameters({'algorithm': freesasa.LeeRichards, 'n-slices': 100, 'probe-radius': 1.4}))
    residueAreas = result.residueAreas()
    RSA = []
    for c in chains:
        for r in residueAreas[c.upper()].keys():
            RSA_AA = []
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeTotal))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativePolar))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeApolar))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeMainChain))
            if math.isnan(residueAreas[c.upper()][r].relativeSideChain):
                RSA_AA.append(0)
            else:
                RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeSideChain))
            RSA.append(RSA_AA)
    RSA_dict[protein] = RSA
    return RSA_dict


def process_rsa():
    global pdb_path
    pdb_path = './Dataset/pdb/'
    for file in tqdm(os.listdir(pdb_path)):
        protein_name = file.split('.')[0]
        chain = protein_name.split('_')[1] # or protein_name[-1]
        rsa = get_RSA(protein_name, chain)
        data = rsa[protein_name]
        aa = np.array(data)
        maxv = aa.max()
        minv = aa.min()
        for x in aa:
            for i in range(len(x)):
                v = x[i]
                xv = (v - minv) / (maxv - minv)
                x[i] = xv
        save_path = path + protein_name
        np.save(save_path, aa)


if __name__ == '__main__':
    process_rsa()
