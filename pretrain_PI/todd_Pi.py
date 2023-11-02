import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import gudhi as gd
import networkx as nx
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from torch_geometric.utils import remove_self_loops
import sys
from torch_geometric.data import download_url
from loader import MoleculeDataset
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, Silhouette, PersistenceEntropy
sys.path.append(".")
from torch_geometric.utils import to_networkx
from gtda.diagrams import BettiCurve
# import learnable_filter.loaddatas_LP as lds
import torch
import sys
#from spectral import SpectralClustering
from tqdm import tqdm
import random
import pickle
#from new_PD import perturb_filter_function, Union_find
from accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD
import time


def hks_signature(subgraph, time):
    A = nx.adjacency_matrix(subgraph)
    L = csgraph.laplacian(A, normed=True)
    egvals, egvectors = eigh(L.toarray())
    return np.square(egvectors).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)


def compute_persistence_image(g, filt):
    # extract subgraph
    subgraph = g
    mapping = nx.get_edge_attributes(g,'edge_attr')
    if filt == 'bond':
        bond_filt = [0] * subgraph.number_of_nodes()
        for edge in g.edges():
            bond_filt[edge[0]] = max(bond_filt[edge[0]], mapping[edge][0])
            bond_filt[edge[1]] = max(bond_filt[edge[1]], mapping[edge][0])
            
    if filt == 'atom':
        filtration_val = np.array([[g.nodes[i]['x'][0],i] for i in g.nodes()])
        filtration_set = sorted(list(set([g.nodes[i]['x'][0] for i in g.nodes()])))
    elif filt == 'charge':
        temp=[]
        temp_value=[]
        for i in g.nodes():
            try:
                temp.append([round(g.nodes[i]['x'][2]*10),i])
                temp_value.append(round(g.nodes[i]['x'][2]*10))
            except:
                temp.append([0,i])
                temp_value.append(0)
        filtration_val = np.array(temp)
        filtration_set = sorted(list(set(temp_value)))
    elif filt == 'bond':
        filtration_val = np.array([[fv,i] for i,fv in enumerate(bond_filt)])
        filtration_set = sorted(list(set(bond_filt)))
    elif filt == 'hks':
        filtration_val = hks_signature(g, time=0.1)
        filtration_val = np.array([[fv,i] for i,fv in enumerate(filtration_val)])
        filtration_set = sorted(list(set(hks_signature(g, time=0.1))))
    list_features=[]
    Dist_matrix = nx.floyd_warshall_numpy(g, weight='None')
    # print(Dist_matrix)
    # print(filtration_val,filtration_set)
    dict = {idx:filtration_val[:,0]<=i for idx,i in enumerate(filtration_set)}
    for idx, i in enumerate(filtration_set):
        # subgraph = g.subgraph(np.array(range(len(dict[idx])))[dict[idx]])
        VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0,1], max_edge_length=15)
        vectorizer = BettiCurve(n_bins=6, n_jobs=-1)
        dist_matrix=Dist_matrix[dict[idx]][:,dict[idx]]
        diagrams = VR.fit_transform(dist_matrix.reshape(1, *dist_matrix.shape))
        # print(diagrams)
        features = vectorizer.fit_transform(diagrams)
        # print(features)
        list_features.append(features)
    list_features=np.hstack(list_features)
    # print(list_features)
    return list_features


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity

def call(data, name='zinc', filt = 'degree', hks_time = 0.1, mode = 'PI', num_models = 5):

    dict_store = []
    PD = []
    result = []
    pool = multiprocessing.Pool(processes=10)
    # for original computation
    for tt in tqdm(range(len(data))):
        g = to_networkx(data[tt],to_undirected=True,node_attrs=["x"],edge_attrs=["edge_attr"])
        result.append(pool.apply_async(compute_persistence_image, args=(g,filt)))
        # result.append(compute_persistence_image(g,filt))
    pool.close()
    pool.join()
    dict_store = [r.get() for r in result]
    # print(dict_store)
    if filt != 'hks':
        save_name = './PI/' + name + '_' + filt + '_VR.pkl'
    else:
        save_name = './' + name + '_' + filt + str(hks_time) + '_VR.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    return 0

import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='implementation of PI')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent')
    args = parser.parse_args()
    dataset = MoleculeDataset("../finetune/dataset/" + args.dataset, dataset=args.dataset)

    for filt in ['bond', 'atom']:
        print(filt)
        if args.dataset == 'zinc_standard_agent':
            data = dataset
            print(call(name='zinc', data = data, filt = filt, hks_time = 0.1, num_models = 10))
        else:
            data = dataset
            print(call(name=args.dataset, data = data, filt = filt, hks_time=0.1, num_models=10))



# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
# from torch.utils import data
# from torch_geometric.data import Data



# # allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list' : list(range(1, 119)),
#     'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list' : [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list' : [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds' : [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs' : [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }
# def mol_to_graph_data_obj_simple(mol):
#     """
#     Converts rdkit mol object to graph Data object required by the pytorch
#     geometric package. NB: Uses simplified atom and bond features, and represent
#     as indices
#     :param mol: rdkit mol object
#     :return: graph data object with the attributes: x, edge_index, edge_attr
#     """
#     # atoms
#     num_atom_features = 2   # atom type,  chirality tag
#     atom_features_list = []
#     mol_ = Chem.AddHs(mol)
#     AllChem.ComputeGasteigerCharges(mol_)
#     charges = [atom.GetProp('_GasteigerCharge') for atom in mol_.GetAtoms()]
#     idx = 0
#     for atom in mol.GetAtoms():
#         atom_feature = [allowable_features['possible_atomic_num_list'].index(
#             atom.GetAtomicNum())] + [allowable_features[
#             'possible_chirality_list'].index(atom.GetChiralTag())] + [float(charges[idx])]
#         idx+=1
#         atom_features_list.append(atom_feature)
#     x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
#     # bonds
#     num_bond_features = 2   # bond type, bond direction
#     if len(mol.GetBonds()) > 0: # mol has bonds
#         edges_list = []
#         edge_features_list = []
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             edge_feature = [allowable_features['possible_bonds'].index(
#                 bond.GetBondType())] + [allowable_features[
#                                             'possible_bond_dirs'].index(
#                 bond.GetBondDir())]
#             edges_list.append((i, j))
#             edge_features_list.append(edge_feature)
#             edges_list.append((j, i))
#             edge_features_list.append(edge_feature)

#         # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
#         edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

#         # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#         edge_attr = torch.tensor(np.array(edge_features_list),
#                                  dtype=torch.long)
#     else:   # mol has no bonds
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#         edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#     return data

# if __name__ == "__main__":
    
#     # with open("data.pkl", "rb") as file:
#     #     dataset = pickle.load(file)
    
#     parser = argparse.ArgumentParser(description='implementation of PI')
#     parser.add_argument('--dataset', type=str, default='zinc_standard_agent')
#     args = parser.parse_args()
    
#     input_path = '../finetune/dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz'
#     input_df = pd.read_csv(input_path, sep=',', compression='gzip',
#                             dtype='str')
#     smiles_list = list(input_df['smiles'])
#     zinc_id_list = list(input_df['zinc_id'])
#     data_list = []
#     for i in range(len(smiles_list)):
#         print(i)
#         s = smiles_list[i]
#         # each example contains a single species
#         rdkit_mol = AllChem.MolFromSmiles(s)
#         if rdkit_mol != None:  # ignore invalid mol objects
#             # # convert aromatic bonds to double bonds
#             # Chem.SanitizeMol(rdkit_mol,
#             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#             data = mol_to_graph_data_obj_simple(rdkit_mol)
            
#             # manually add mol id
#             id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
#             data.id = torch.tensor(
#                 [id])  # id here is zinc id value, stripped of
#             # leading zeros
#             data_list.append(data)
#             # break
#     dataset = data_list
#     # with open("data.pkl", "wb") as file:
#     #     pickle.dump(dataset, file)

#     for filt in ['charge','bond','atom']:
#         print(filt)
#         if filt == 'hks':
#             for hks_time in [0.1]:
#                 data = dataset
#                 print(call(name=args.dataset, data = data, filt = filt, hks_time = hks_time, num_models = 10))
#         else:
#             data = dataset
#             print(call(name=args.dataset, data = data, filt = filt, hks_time=0.1, num_models=10))
