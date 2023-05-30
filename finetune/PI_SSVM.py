import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, scaffold_split_subset
import pandas as pd

import os
import shutil

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def eval(y_true, y_scores):
   
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--dataset', type=str, default = 'clintox', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--threshold', type=int, default = 3)
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])
    
    train_Y = np.concatenate([item.y.numpy() for item in train_dataset]).reshape((-1,num_tasks))
    valid_Y = np.concatenate([item.y.numpy() for item in valid_dataset]).reshape((-1,num_tasks))
    test_Y = np.concatenate([item.y.numpy() for item in test_dataset]).reshape((-1,num_tasks))
    
    train_X = np.concatenate([item.PI.numpy() for item in train_dataset]).reshape((-1,150))
    valid_X = np.concatenate([item.PI.numpy() for item in valid_dataset]).reshape((-1,150))
    test_X = np.concatenate([item.PI.numpy() for item in test_dataset]).reshape((-1,150))

    best=0
    for iC in [0.01, 0.1, 1, 10]:
        for ig in [0.01, 0.1, 1, 10]:
            svm_clf = SVC(kernel="rbf", C=iC, gamma=ig, probability=True)
            from sklearn.multioutput import MultiOutputClassifier
            multi_clf = MultiOutputClassifier(svm_clf, n_jobs=-1)
            

            multi_clf.fit(train_X, ((train_Y + 1)/2).astype(int))

            y_pred_val = multi_clf.predict_proba(valid_X)
            y_pred = []
            for i in range(len(y_pred_val)):
                y_pred.append(y_pred_val[i][:,1])
            y_pred = np.concatenate(y_pred).reshape((num_tasks,-1)).T
            print(eval(valid_Y,y_pred))
            if(best<eval(valid_Y,y_pred)):
                best = eval(valid_Y,y_pred)
                y_pred_test = multi_clf.predict_proba(test_X)
                y_pred = []
                for i in range(len(y_pred_test)):
                    y_pred.append(y_pred_test[i][:,1])
                y_pred = np.concatenate(y_pred).reshape((num_tasks,-1)).T
                # print(y_pred.shape,test_Y.shape)
                best_test = eval(test_Y,y_pred)
    print("best:",best, best_test)
    
    print_to_file("SVM.txt", str(args.split_seed)+','+str(best_test))
if __name__ == "__main__":
    main()
