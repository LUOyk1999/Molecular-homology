import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from Topo_model import GNN

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

class Topo_Model(torch.nn.Module):
    def __init__(self, gnn, hidden_dim = 300):
        super(Topo_Model, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(hidden_dim, 300), nn.ReLU(inplace=True), nn.Linear(300, 100))
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))
    def forward(self, x0, edge_index0, edge_attr, batch):
        # this process rely on the ground truth PD, therefore only used for training

        x = self.gnn(x0, edge_index0, edge_attr)
        x = self.projection_head(x)
        x = self.pool(x,batch)
        x = x.reshape(-1)
    
        return x

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

def train(args, model, device, loader, optimizer):
    model.train()
    cnt_sample = 0
    Total_loss_0 = 0
    distance = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x0 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # print(x0.shape,batch.PI.shape)
        # PI = F.normalize(batch.PI, dim = -1)
        x0 = x0.reshape(batch.id.shape[0],-1)
        PI = batch.PI.reshape(batch.id.shape[0],-1)
        PI = PI[:,:100]
        distance += torch.mean(torch.sum((x0-PI)*(x0-PI),dim=1).sqrt()).cpu().detach()
        
        loss = loss_topo(x0, x0, PI)
        
        Total_loss_0 += loss.cpu().detach()
        cnt_sample += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return Total_loss_0/cnt_sample, distance/cnt_sample

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
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = 'output_TDL_', help='filename to output the pre-trained model')
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
    dataset = MoleculeDataset("../finetune/dataset/" + args.dataset, dataset=args.dataset, topo_features=dict_save)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = Topo_Model(gnn)
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
        if epoch % 20 == 0:
            if not args.output_model_file == "":
                torch.save(model.gnn.state_dict(), args.output_model_file + str(epoch) + ".pth")


if __name__ == "__main__":
    main()
