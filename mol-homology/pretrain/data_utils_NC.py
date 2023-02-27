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
from loader import MoleculeDataset_init
sys.path.append(".")
from torch_geometric.utils import to_networkx
import sg2dgm.PersistenceImager as pimg
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
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, Silhouette, PersistenceEntropy
from gtda.plotting import plot_diagram
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

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



def compute_persistence_image(g, filt = 'hks', hks_time = 0.1, mode = 'PI', num_models = 5, max_loop_len = 10, cycle_the = 2):
    # extract subgraph
    
    subgraph = g
    
    # prepare computation of extended persistence
    if len(subgraph.edges()) == 0:
        return None, None
    #num_vertices = len(subgraph.nodes())
    #edge_list = np.array([i for i in subgraph.edges()])
    #xs = edge_list[:, 0]
    #ys = edge_list[:, 1]
    if len(subgraph.edges()) > 0:
        edge_index = torch.Tensor([[e[0], e[1]] for e in subgraph.edges()]).transpose(0, 1).long()
    else:
        edge_index = torch.Tensor([[0], [0]]).long()
    # compute filter function
    if filt == 'hks':
        filtration_val = hks_signature(subgraph, time=hks_time)
        filtration_val /= (max(filtration_val) + + 1e-10)
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

    # generate edge number
    cnt_edge = 0
    for edge in subgraph.edges():
        u1, v1 = edge[0], edge[1]
        if 'num' not in subgraph[u1][v1]:
            subgraph[u1][v1]['num'] = cnt_edge
            subgraph[v1][u1]['num'] = cnt_edge
            cnt_edge += 1

    # G = nx.Graph()  

    # G.add_edges_from([(1, 3),(1,4),(1,6),(4,6),(3,4),(2,3),(2,5)])
    # VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0,1], max_edge_length=4)
    # vectorizer = BettiCurve(n_bins=13, n_jobs=-1)
    # dist_matrix = nx.floyd_warshall_numpy(G, weight='None')
    # diagrams = VR.fit_transform(dist_matrix.reshape(1, *dist_matrix.shape))
    # print(diagrams[0],subgraph.edges())
    # fig = plot_diagram(diagrams[0])
    # fig.write_image("a.png")
    # nx.draw(G)
    # plt.savefig("b.png")
    # plt.clf() 
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
        pers_imager = pimg.PersistenceImager(resolution=5)
        PI0 = pers_imager.transform(dgmOrd0).reshape(-1) if len(dgmOrd0) > 0 else np.zeros(25)
        PI1 = pers_imager.transform(dgmExt1).reshape(-1) if len(dgmExt1) > 0 else np.zeros(25)
        if len(dgmOrd0) == 0:
            pers_img = PI1
        elif len(dgmExt1) == 0:
            pers_img = PI0
        else:
            pers_img = pers_imager.transform(np.concatenate((dgmOrd0, dgmExt1))).reshape(-1)
        PI_time = time.time() - t1
        #return np.concatenate((dgmOrd0, dgmExt0)), np.array(dgmExt1), pers_img.reshape(-1), filtration_val, edge_index, Loop_features, Loop_edge_indices
        return np.array(dgmOrd0), np.array(dgmExt1), pers_img, filtration_val, edge_index, PI0, PI1, PD_time, PI_time

    elif mode == 'filtration':
        #return filtration_val, edge_index, Loop_features, Loop_edge_indices
        return filtration_val, edge_index

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def call(data, name='zinc', filt = 'degree', hks_time = 10, mode = 'PI', num_models = 5):

    total_time_PD = 0
    total_time_PI = 0
    dict_store = {}
    
    # for original computation
    for tt in tqdm(range(len(data))):
        g = to_networkx(data[tt],to_undirected=True,node_attrs=["x"])
        
        dict_store[tt] = \
            compute_persistence_image(g, filt = filt, hks_time = hks_time, mode = mode, num_models = num_models)
    # original save name, for further evaluation
    if filt != 'hks':
        save_name = './' + name + '_' + filt + '_NC.pkl'
    else:
        save_name = './' + name + '_' + filt + str(hks_time) + '_NC.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)

    '''
    # for time evaluation for hop 1 / 3
    save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + '_hop3_test.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    '''
    return total_time_PD, total_time_PI

if __name__ == "__main__":

    dataset = MoleculeDataset_init('./data_init')
    # for filt in ['degree', 'hks', 'atom']:
    for filt in ['degree', 'atom', 'hks']:
    # for filt in ['centrality', 'clustering']:
        print(filt)
        if filt == 'hks':
            for hks_time in [0.1, 10]:
                data = dataset
                print(call(data = data, filt = filt, hks_time = hks_time, num_models = 10))
        else:
            data = dataset
            print(call(data = data, filt = filt, hks_time=10, num_models=10))
