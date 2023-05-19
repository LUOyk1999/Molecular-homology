import json
import re

import pandas as pd

# from matplotlib import pyplot as plt

from polymers.utils.mol import encode_smiles, FP
from rogi import RoughnessIndex
from rdkit import Chem
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')
from torch_geometric.datasets import MoleculeNet
from molhomology.finetune.loader import MoleculeDataset

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# xs = [(numy-2d-array, string)] ie '[(data , data name)]'
def pic_sings(xs, colors, alphas,
              xlabel='Singular Value Rank Index', ylabel='Singular Value', title='', path=''):

    # font = {'family': 'normal',
    #         'weight': 'bold',}
    #         # 'size': 22}
    #
    # matplotlib.rc('font', **font)

    plt.figure(figsize=(6, 5))
    fig, ax = plt.subplots()

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    for i, yl in enumerate(xs):
        y, l = yl
        ax.plot(np.array(list(range(len(y)))), y, c=colors[i], alpha=alphas[i], label=l, linewidth=3)

    plt.legend()
    # plt.show()
    if path:
        fig.savefig(path)


# TODO direct clr has attribution 4.0 license so check if other repo has same code...
def singular(latents):
    latents = torch.from_numpy(np.stack(latents, axis=0))

    z = torch.nn.functional.normalize(latents, dim=1)

    # calculate covariance
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)
    embedding_spectrum = d
    embedding_spectrum = np.log(embedding_spectrum)
    return embedding_spectrum


if __name__ == "__main__":
    DPATH = '/Users/veronika/Desktop/git/polymers/polymers/analyze/embeddings'

    ms = ['simgrace', 'graphlog'] #'graphcl',
    ns = [ 'SimGRACE', 'GraphLOG'] # 'GraphCL',
    ms = ['graphcl', 'JOAO', 'simgrace', 'graphlog']
    ns = ['GraphCL', 'JOAO', 'SimGRACE', 'GraphLOG']
    # ms = [ 'JOAO']
    # ns = ['JOAO']
    ds = ['Clintox', 'Bace', 'BBBP', 'Sider']
    cs = ['b', 'b', 'm', 'm','r', 'r', 'g', 'g']
    alphas = [1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5]

    for d in ds:
        dl = d.lower()
        dgs = []
        for i, m in enumerate(ms):
            p = f'{DPATH}/{d}_{m}_TDL.pkl'
            with open(p, "rb") as f:
                dm = pickle.load(f)

            sv = singular(dm)
            dd = pd.DataFrame({'x': list(range(len(sv))), 'y': sv}).to_csv(f"./pics/{dl}_{m.lower()}_tdl.txt", index=False)
            # np.savetxt(f"./pics/{dl}_{m.lower()}_tdl.txt", dd)
            dgs += [(singular(dm), ns[i]+"+TDL")]
            p = f'{DPATH}/{d}_{m}.pkl'
            with open(p, "rb") as f:
                dm = pickle.load(f)
            sv = singular(dm)
            dd = pd.DataFrame({'x': list(range(len(sv))), 'y': sv}).to_csv(f"./pics/{dl}_{m.lower()}.txt", index=False)

            dgs += [(singular(dm), ns[i])]

        pic_sings(dgs, cs, alphas, title=d, path=f"./pics/{d}.png")


