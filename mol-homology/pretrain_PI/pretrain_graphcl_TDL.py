import argparse

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from Topo_model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd


from copy import deepcopy


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

def loss_topo(x1, x2, PI):
    T = 0.1
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    
    distance = torch.cdist(PI, PI)
    distance.fill_diagonal_(0)
    
    indices = torch.argsort(distance, dim=1)
    rank = torch.zeros_like(distance)
    rank = rank.scatter_(1, indices, torch.arange(distance.shape[1], device=distance.device).unsqueeze(0).expand(distance.size(0), -1).float())
    
    rk = rank.repeat(batch_size, 1, 1)
    arange = torch.arange(batch_size, device=rank.device).unsqueeze(1).unsqueeze(2)
    mask = (rk >= arange)
    # mask[0].fill_diagonal_(False)
    denominator = torch.sum(mask*sim_matrix.unsqueeze(0), dim=-1).T
    numerator = torch.gather(sim_matrix, dim=1, index=indices)
    denominator = denominator[:,1:]
    numerator = numerator[:,1:]
    # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = numerator / denominator
    loss = - torch.log(loss).mean()
    # print(loss)
    return loss

class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.projection_head_topo = nn.Sequential(nn.Linear(100, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.projection_head_topo_phi = nn.Sequential(nn.Linear(400, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
    
    def forward_topo(self, x):
        x = self.projection_head_topo(x)
        return x
    
    def forward_topo_phi(self, PI, x):
        x = torch.cat((PI, x), dim=-1)
        x = self.projection_head_topo_phi(x)
        return x
    
    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset0 = deepcopy(dataset1)
    dataset0.aug, dataset1.aug_ratio = 'none', 0
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader0 = DataLoader(dataset0, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_l2 = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader0, loader1, loader2), desc="Iteration")):
        batch0, batch1, batch2 = batch
        batch0 = batch0.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()
        
        PI = batch0.PI.reshape(batch0.id.shape[0],-1)
        PI = PI[:,:100]
        x0 = model.forward_cl(batch0.x, batch0.edge_index, batch0.edge_attr, batch0.batch)
        # x0 = model.forward_topo_phi(PI, x0)
        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        
        l1 = model.loss_cl(x1, x2)
        # l2 = model.loss_cl(x1, x0)
        l2 = loss_topo(x0, x0, PI) 
        loss = l1 + l2
        loss.backward()
        optimizer.step()

        train_loss_accum += float(l1.detach().cpu().item())
        
        train_l2 += float(l2.detach().cpu().item())

    return train_l2/(step+1), train_loss_accum/(step+1)

import pickle
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
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
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'dropN')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'dropN')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
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
    dataset = MoleculeDataset_aug("../finetune/dataset/" + args.dataset, dataset=args.dataset, topo_features=dict_save)
    print(dataset)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
    
        train_l2, train_loss = train(args, model, device, dataset, optimizer)

        print(train_l2)
        print(train_loss)

        if epoch % 20 == 0:
            torch.save(gnn.state_dict(), "./graphcl_TDL_dropN" + str(epoch) + ".pth")

if __name__ == "__main__":
    main()
