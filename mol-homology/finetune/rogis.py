import json
import re
import numpy as np
import pandas as pd
import pickle
import torch
from rdkit import Chem
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')
from torch_geometric.datasets import MoleculeNet
from loader import MoleculeDataset

# needs https://github.com/coleygroup/rogi
from rogi import RoughnessIndex


def compute_rogis(x, y, idx):
    ris = []
    for j, yj in enumerate(y):
        ri = RoughnessIndex(Y=yj[idx[j]], X=x[idx[j]], metric="euclidean")
        ris += [ri.compute_index()]

    # weight by actual data length (w/o nans)
    # ll = [len(s) for s in x]
    # rr = [ris[i] * l / sum(ll) for i, l in enumerate(ll)]
    # print(m, sum(rr), sum(ris) / len(ris))  # AVG
    return ris, sum(ris) / len(ris)  # sum(rr),


CAP_START='results:'
def df_to_str(df, models_sorted, cols, name):
    df = df.set_index("model")  # otherwise other index is printed
    df = df.reindex(models_sorted) if models_sorted else df.sort_values('model')
    df.index.name = None
    df.columns = cols

    #         .format(subset=stdcols, formatter='\stdcol{:.4f}') \
    #         # .highlight_min(axis=0, props='bfseries: ;', subset=[m for m in metric_cols if m_stem(m) not in MAX_METRICS]) \
    #         # .highlight_max(axis=0, props='bfseries: ;', subset=[m for m in metric_cols if m_stem(m) in MAX_METRICS]) \
    # if c in metric_cols else "@{\hskip\clspace}c"
    s = df.style \
        .format(precision=4) \
        .to_latex(position="t", position_float='centering',
                  column_format="l" + "".join(["c" for c in cols]),
                  hrules=True, label=f"tab:{name}",
                  caption=f"{CAP_START}{name}.")
    s = re.sub(r'tmpstd[0-9]*', '', s)  # remove dummy col names
    return s.replace('_', '\_')


if __name__ == "__main__":

    DATA = '/Users/veronika/Desktop/git/polymers/polymers/analyze/embeddings/'

    ms = ['graphcl', 'JOAO', 'simgrace', 'graphlog']
    ns = ['GraphCL', 'JOAO', 'SimGRACE', 'GraphLOG']
    ds = ['Clintox','Bace', 'BBBP', 'Sider']  #

    results = np.ndarray((len(ms)*2, len(ds)))

    for id, d in enumerate(ds):
        name = d.lower()
        root = './dataset/'
        dataset = MoleculeDataset(f"{root}/{name}", dataset=name)

        print(name, len(dataset), dataset.data.y.shape[-1])
        yo = dataset.data.y.reshape(len(dataset), -1)

        #         is_valid = y**2 > 0
        #         #Loss matrix
        #         loss_mat = criterion(pred.double(), (y+1)/2)
        idx, y = [], []
        for i in range(yo.shape[-1]):
         idx += [[j for j, yy in enumerate(yo[:, i]) if yy**2 > 0]]
         y += [np.array([yy.item() for yy in yo[:, i]])]
        for im, m in enumerate(ms):
            p = f'/{DATA}/{d}_{m}.pkl'
            with open(p, "rb") as f:
                dm = pickle.load(f)

            x = np.stack(dm)
            ris, avgg = compute_rogis(x, y, idx)
            print(m, ris, avgg)

            results[im*2, id] = avgg

            p = f'{DATA}/{d}_{m}_TDL.pkl'
            with open(p, "rb") as f:
                dm = pickle.load(f)

            x = np.stack(dm)
            ris, avgg = compute_rogis(x, y, idx)
            print(m+"+TDL", ris, avgg)

            results[im * 2 + 1, id] = avgg

    cs = []
    for n in ns:
        cs +=[n, n+"+TDL"]

    df = pd.DataFrame(results)
    df['model'] = cs
    s = df_to_str(df, cs, ds, 'dummytitle')

    print(results)
    print(s)