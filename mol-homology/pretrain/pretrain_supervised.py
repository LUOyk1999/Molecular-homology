import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from Topo_model import GNN, Topo_Model
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

# criterion = nn.BCEWithLogitsLoss(reduction = "none")
Loss = torch.nn.MSELoss()

def train(args, model, device, loader, optimizer):
    model.train()
    Total_loss_0 = 0
    Total_loss_PI = 0
    Total_loss_xy0 = 0; Total_loss_xd0 = 0; Total_loss_yd0 = 0
    cnt_sample = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x0, x, loss_0, loss_xy0, loss_xd0, loss_yd0, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.PD, kernel='wasserstein', grad_PI = False)
        
        #PI = F.normalize(PI, dim = -1)
        # loss_PI = Loss(x, batch.PI)
        loss = loss_0
        # loss = loss_0 + loss_PI
        # Total_loss_PI += loss_PI.cpu().detach()
        Total_loss_0 += loss_0.cpu().detach()
        Total_loss_xy0 += loss_xy0.cpu().detach()
        Total_loss_xd0 += loss_xd0.cpu().detach()
        Total_loss_yd0 += loss_yd0.cpu().detach()
        cnt_sample += args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return Total_loss_0 / cnt_sample, Total_loss_PI / cnt_sample, \
           Total_loss_xy0 / cnt_sample, Total_loss_xd0 / cnt_sample, Total_loss_yd0 / cnt_sample

# def eval(args, model, device, loader, normalized_weight):
#     model.eval()
#     y_true = []
#     y_scores = []

#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)

#         with torch.no_grad():
#             pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

#         y_true.append(batch.y.view(pred.shape).cpu())
#         y_scores.append(pred.cpu())

#     y_true = torch.cat(y_true, dim = 0).numpy()
#     y_scores = torch.cat(y_scores, dim = 0).numpy()

#     roc_list = []
#     weight = []
#     for i in range(y_true.shape[1]):
#         #AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
#             is_valid = y_true[:,i]**2 > 0
#             roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
#             weight.append(normalized_weight[i])

#     if len(roc_list) < y_true.shape[1]:
#         print("Some target is missing!")
#         print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

#     weight = np.array(weight)
#     roc_list = np.array(roc_list)

#     return weight.dot(roc_list)

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
    parser.add_argument('--dataset', type=str, default = './data/zinc/all.txt', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = './saved_model/homology_output', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    num_tasks = 1310

    #set up dataset
    filt='degree'
    type='NC'
    if filt != 'hks':
        save_name = './' + 'zinc' + '_' + filt + '_' + type + '.pkl'
    else:
        save_name = './' + 'zinc' + '_' + filt + '0.1'+ '_' +  type + '.pkl'
    dict_save=None
    with open(save_name, 'rb') as f:
        dict_save = pickle.load(f)
    dataset = MoleculeDataset('./data', topo_features=dict_save)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = Topo_Model(hidden_dim = args.emb_dim, type = 'GIN', num_models=1, dropout = args.dropout_ratio)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        # evaluate influence of numbers
       
        train_loss_0, train_loss_PI, train_xy0, train_xd0, train_yd0 = train(args, model, device, loader, optimizer)
        print(train_loss_0, train_loss_PI)

        if not args.output_model_file == "":
            print(args.output_model_file)
            torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
