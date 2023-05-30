import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loader import MoleculeDataset
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from rdkit import DataStructs
parser = argparse.ArgumentParser(description='implementation of PI')
parser.add_argument('--dataset', type=str, default='bace')
args = parser.parse_args()
smiles_list = pd.read_csv('../finetune/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()[:10001]

fp_size = 1024
radius = 2

dict_save=None
with open('./PI/'+args.dataset+'_atom_Pi.pkl', 'rb') as f:
    dict_save1 = np.stack(pickle.load(f))
with open('./PI/'+args.dataset+'_degree_Pi.pkl', 'rb') as f:
    dict_save2 = np.stack(pickle.load(f))
with open('./PI/'+args.dataset+'_hks0.1_Pi.pkl', 'rb') as f:
    dict_save3 = np.stack(pickle.load(f))
dict_save = (dict_save1,dict_save2,dict_save3)
PI = np.concatenate(dict_save, 1)[:10001,:]
print(PI.shape)

ecfp_array = np.zeros((len(smiles_list), fp_size), dtype=int)
ecfp_ = []
for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
        ecfp_array[i, :] = np.array(ecfp, dtype=int)
        ecfp_.append(ecfp)
print(ecfp_array.shape)
ecfp_array = ecfp_array[:-1]
print(ecfp_array.shape)
def dice_similarity(a, b):
    intersection = np.sum(np.minimum(a, b))
    return 2 * intersection / (np.sum(a) + np.sum(b))

def get_highest_dice_similarity_pairs(ecfp_array, top_pairs_count):
    
    similarity_matrix = pairwise_distances(ecfp_array, metric=dice_similarity)
    similarity_matrix_ = similarity_matrix
    np.fill_diagonal(similarity_matrix, -np.inf)

    top_pairs_indices = np.array(range(top_pairs_count))
    top_pairs_indices = np.column_stack(np.unravel_index(top_pairs_indices, similarity_matrix.shape))
    
    matched_pairs = []
    unmatched_elements = set(range(similarity_matrix.shape[0]))
    count = 0
    for i, j in sorted(top_pairs_indices, key=lambda x: similarity_matrix[x[0], x[1]], reverse=True):
        if i in unmatched_elements and j in unmatched_elements:
            matched_pairs.append((i, j))
            unmatched_elements.discard(i)
            unmatched_elements.discard(j)
            count+=1
            if(count == int(len(ecfp_array)*0.1)):
                break

    unmatched_elements = list(unmatched_elements)
    np.random.shuffle(unmatched_elements)
    print(unmatched_elements)
    remaining_pairs = [(unmatched_elements[i], unmatched_elements[i + 1]) for i in range(0, len(unmatched_elements), 2)]

    return similarity_matrix_, matched_pairs, remaining_pairs

similarity_matrix, matched_pairs, remaining_pairs = get_highest_dice_similarity_pairs(ecfp_array, len(ecfp_array)*len(ecfp_array))

print("top:", matched_pairs)
print("remain:", remaining_pairs)

def cosine_similarity(v1, v2):
    if (np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0):
        return 0.0
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    x_normalized = (x - x_mean) / x_std
    y_normalized = (y - y_mean) / y_std
    pearson_correlation = np.sum(x_normalized * y_normalized) / len(x)
    return pearson_correlation

matched_=[]
un_matched_=[]

matched_fp=[]
un_matched_fp=[]

for idx in matched_pairs:
    matched_.append(cosine_similarity(PI[idx[0]],PI[idx[1]]))
    matched_fp.append(similarity_matrix[idx[0],idx[1]])
for idx in remaining_pairs:
    un_matched_.append(cosine_similarity(PI[idx[0]],PI[idx[1]]))
    un_matched_fp.append(similarity_matrix[idx[0],idx[1]])
print(matched_,un_matched_)
print(np.mean(matched_),np.mean(un_matched_),np.mean(matched_fp),np.mean(un_matched_fp))

import numpy as np
import matplotlib.pyplot as plt

# plt.title('Bace', fontsize=14)
plt.xlabel('Cosine Similarity', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.yscale('log')
plt.hist(matched_, density=True, bins=100, label='ecfp_matched')
plt.hist(un_matched_, density=True, bins=100, alpha=0.5, label='ecfp_un_matched')

plt.savefig('Pi_'+args.dataset+'.png')