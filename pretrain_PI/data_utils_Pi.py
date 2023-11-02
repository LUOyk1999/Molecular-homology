import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gudhi as gd
import networkx as nx
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from torch_geometric.utils import remove_self_loops
import sys
from torch_geometric.data import download_url
from loader import MoleculeDataset
sys.path.append(".")
from torch_geometric.utils import to_networkx
from gtda.diagrams import BettiCurve, PersistenceLandscape, PersistenceImage
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

def apply_graph_extended_persistence(num_vertices, xs, ys, filtration_val):
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0,2])
    dgmRel1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0,2])
    dgmExt0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0,2])
    dgmExt1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0,2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def original_extended_persistence(subgraph, filtration_val):
    simplex_filter = perturb_filter_function(subgraph, filtration_val)
    dgmOrd0, dgmExt0, dgmRel1, Pos_edges, Neg_edges = Union_find(simplex_filter)
    dgmExt1 = Accelerate_PD(Pos_edges, Neg_edges, simplex_filter)
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def new_extended_persistence(subgraph, filtration_val):
    simplex_filter = perturb_filter_function(subgraph, filtration_val)
    dgmOrd0 ,dgmExt0, dgmRel1, dgmExt1 = Union_find(simplex_filter)

    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def hks_signature(subgraph, time):
    A = nx.adjacency_matrix(subgraph)
    L = csgraph.laplacian(A, normed=True)
    egvals, egvectors = eigh(L.toarray())
    return np.square(egvectors).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)


from mendeleev.fetch import fetch_table

def compute_persistence_image(g, filt = 'degree', hks_time = 0.1, mode = 'PI'):
    # extract subgraph
    
    subgraph = g
    mapping = nx.get_edge_attributes(g,'edge_attr')
    if filt == 'bond':
        bond_filt = [100] * subgraph.number_of_nodes()
        for edge in g.edges():
            bond_filt[edge[0]] = min(bond_filt[edge[0]], mapping[edge][0])
            bond_filt[edge[1]] = min(bond_filt[edge[1]], mapping[edge][0])

    # prepare computation of extended persistence
    if len(subgraph.edges()) == 0:
        return np.zeros(50)
    
    if len(subgraph.edges()) > 0:
        edge_index = torch.Tensor([[e[0], e[1]] for e in subgraph.edges()]).transpose(0, 1).long()
    else:
        edge_index = torch.Tensor([[0], [0]]).long()
        
    ptable = fetch_table('elements')
    cols = ['atomic_radius','electron_affinity']
    atom = [subgraph.nodes[i]['x'][0] for i in subgraph.nodes()]
    selected_attribute = ptable[cols]
    atomic_radius = []
    electron_affinity = []
    for i in atom:
        atomic_radius.append(selected_attribute.iloc[i,0])
        electron_affinity.append(selected_attribute.iloc[i,1])
    # compute filter function
    if filt == 'hks':
        filtration_val = hks_signature(subgraph, time=hks_time)
        filtration_val /= (max(filtration_val) + 1e-10)
    elif filt == 'centrality':
        filtration_val = [nx.degree_centrality(subgraph)[i] for i in subgraph.nodes()]
        max_val = max(filtration_val)
        filtration_val = [fv / (max_val + 1e-10) for fv in filtration_val]
    elif filt == 'clustering':
        filtration_val = [nx.clustering(subgraph)[i] for i in subgraph.nodes()]
        max_val = max(filtration_val)
        filtration_val = [fv / (max_val + 1e-10) for fv in filtration_val]
    elif filt == 'degree':
        filtration_val = [subgraph.degree()[i] for i in subgraph.nodes()]
        filtration_val = [fv / (max(filtration_val) + 1e-10) for fv in filtration_val]
    elif filt == 'atom':
        filtration_val = [subgraph.nodes[i]['x'][0] for i in subgraph.nodes()]
        filtration_val = [fv / (max(filtration_val) + 1e-10) for fv in filtration_val]
    elif filt == 'bond':
        filtration_val = bond_filt
        filtration_val = [fv / (max(filtration_val) + 1e-10) for fv in filtration_val]
    elif filt == 'radius':
        filtration_val = atomic_radius
        filtration_val = [fv / (max(filtration_val) + 1e-10) for fv in filtration_val]
    elif filt == 'electron':
        filtration_val = electron_affinity
        filtration_val = [(fv - min(filtration_val)) / (max(filtration_val) + 1e-10) for fv in filtration_val]
    # generate edge number
    cnt_edge = 0
    for edge in subgraph.edges():
        u1, v1 = edge[0], edge[1]
        if 'num' not in subgraph[u1][v1]:
            subgraph[u1][v1]['num'] = cnt_edge
            subgraph[v1][u1]['num'] = cnt_edge
            cnt_edge += 1

    if mode == 'PI':
        t = time.time()
        '''
        # the gudhi EPD computation algorithm
        num_vertices = len(subgraph.nodes())
        edge_list = np.array([i for i in subgraph.edges()])
        xs = edge_list[:, 0]
        ys = edge_list[:, 1]
        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(num_vertices, xs, ys, filtration_val)
        '''
        #dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = new_extended_persistence(subgraph, filtration_val)
        #t = time.time()
        # the fast EPD computation algorithm
        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = original_extended_persistence(subgraph, filtration_val)
        t1 = time.time()
        PD_time = t1 - t
        vectorizer = PersistenceImage(n_bins=5, n_jobs=-1)
        # print(dgmOrd0)
        # pers_imager = pimg.PersistenceImager(resolution=5)
        if len(dgmOrd0) != 0:
            flat_matrix = dgmOrd0.flatten()
            unique_sorted_values = np.unique(np.sort(flat_matrix))
            mapping = {value: index + 1 for index, value in enumerate(unique_sorted_values)}
            dgmOrd0 = np.vectorize(mapping.get)(dgmOrd0)
        if len(dgmExt1) != 0:
            flat_matrix = dgmExt1.flatten()
            unique_sorted_values = np.unique(np.sort(flat_matrix))
            mapping = {value: index + 1 for index, value in enumerate(unique_sorted_values)}
            dgmExt1 = np.vectorize(mapping.get)(dgmExt1)
        # PI0 = pers_imager.transform(dgmOrd0).reshape(-1) if len(dgmOrd0) > 0 else np.zeros(25)
        # PI1 = pers_imager.transform(dgmExt1).reshape(-1) if len(dgmExt1) > 0 else np.zeros(25)
        if len(dgmOrd0) == 0:
            ones = np.ones((dgmExt1.shape[0], 1))
            dgmExt1 = np.hstack((dgmExt1, ones))
            vectorizer = PersistenceImage(n_bins=5, n_jobs=-1)
            features = vectorizer.fit_transform(dgmExt1[np.newaxis, :]).flatten()
            features = np.pad(features, (0, 25), mode='constant', constant_values=0)
            # pers_img = PI1
        elif len(dgmExt1) == 0:
            zeros = np.zeros((dgmOrd0.shape[0], 1))
            dgmOrd0 = np.hstack((dgmOrd0, zeros))
            vectorizer = PersistenceImage(n_bins=5, n_jobs=-1)
            features = vectorizer.fit_transform(dgmOrd0[np.newaxis, :]).flatten()
            features = np.pad(features, (0, 25), mode='constant', constant_values=0)
            # pers_img = PI0
        else:
            zeros = np.zeros((dgmOrd0.shape[0], 1))
            dgmOrd0 = np.hstack((dgmOrd0, zeros))
            ones = np.ones((dgmExt1.shape[0], 1))
            dgmExt1 = np.hstack((dgmExt1, ones))
            # print(np.concatenate((dgmOrd0, dgmExt1))[np.newaxis, :])
            features = vectorizer.fit_transform(np.concatenate((dgmOrd0, dgmExt1))[np.newaxis, :]).flatten()
            # pers_img = pers_imager.transform(np.concatenate((dgmOrd0, dgmExt1))).reshape(-1)
        
        PI_time = time.time() - t1
        #return np.concatenate((dgmOrd0, dgmExt0)), np.array(dgmExt1), pers_img.reshape(-1), filtration_val, edge_index, Loop_features, Loop_edge_indices
        
        return features


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity

def call(data, name='zinc', filt = 'degree', hks_time = 0.1, mode = 'PI'):

    total_time_PD = 0
    total_time_PI = 0
    dict_store = []
    PD = []
    result = []
    pool = multiprocessing.Pool(processes=10)
    # for original computation
    for tt in tqdm(range(len(data))):
        g = to_networkx(data[tt],to_undirected=True,node_attrs=["x"],edge_attrs=["edge_attr"])
        result.append(pool.apply_async(compute_persistence_image, args=(g,filt)))
        
    pool.close()
    pool.join()
    dict_store = [r.get() for r in result]
    # print(dict_store)
    if (name=='zinc_standard_agent'):
        name_copy = 'zinc'
    else:
        name_copy = name
    if filt != 'hks':
        save_name = './PI/' + name_copy + '_' + filt + '_Pi.pkl'
    else:
        save_name = './PI/' + name_copy + '_' + filt + str(hks_time) + '_Pi.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    return total_time_PD, total_time_PI

        
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='implementation of PI')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent')
    args = parser.parse_args()
    dataset = MoleculeDataset("../finetune/dataset/" + args.dataset, dataset=args.dataset)
    for filt in ['degree', 'hks', 'atom']:
    # for filt in ['electron', 'radius']:
        print(filt)
        if filt == 'hks':
            for hks_time in [0.1]:
                data = dataset
                print(call(name=args.dataset, data = data, filt = filt, hks_time = hks_time))
        else:
            data = dataset
            print(call(name=args.dataset, data = data, filt = filt, hks_time=0.1))
