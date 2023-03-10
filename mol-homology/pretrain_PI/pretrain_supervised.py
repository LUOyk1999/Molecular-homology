import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from Topo_model import Topo_Model
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

# criterion = nn.BCEWithLogitsLoss(reduction = "none")
Loss = torch.nn.MSELoss()

def train(args, model, device, loader, optimizer):
    model.train()
    cnt_sample = 0
    Total_loss_0 = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x0 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss_PI = Loss(x0, batch.PI)
        loss = loss_PI
        Total_loss_0 += loss_PI.cpu().detach()
        cnt_sample += args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return Total_loss_0/cnt_sample

import pickle
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = 'output', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #Bunch of classification tasks
    if args.dataset == "zinc_standard_agent":
        num_tasks = 1310
    else:
        raise ValueError("Invalid dataset name.")

    dict_save=None
    with open('./PI/zinc_atom_PI.pkl', 'rb') as f:
        dict_save1 = pickle.load(f)
    with open('./PI/zinc_degree_PI.pkl', 'rb') as f:
        dict_save2 = pickle.load(f)
    with open('./PI/zinc_hks0.1_PI.pkl', 'rb') as f:
        dict_save3 = pickle.load(f)
    with open('./PI/zinc_hks10_PI.pkl', 'rb') as f:
        dict_save4 = pickle.load(f)
    dict_save = (dict_save1,dict_save2,dict_save3,dict_save4)
    dataset = MoleculeDataset("../finetune/dataset/" + args.dataset, dataset=args.dataset, topo_features=dict_save)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = Topo_Model(hidden_dim = args.emb_dim, dropout = args.dropout_ratio)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        # evaluate influence of numbers
       
        loss = train(args, model, device, loader, optimizer)
        print(loss)

        if not args.output_model_file == "":
            torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")
            torch.save(model.state_dict(), args.output_model_file + "_total.pth")

if __name__ == "__main__":
    main()
