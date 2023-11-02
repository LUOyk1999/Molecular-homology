import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
import pickle
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

class Topo_Model(torch.nn.Module):
    def __init__(self, gnn, hidden_dim = 300, filt=-1):
        super(Topo_Model, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        if filt==-1:
            self.projection_head = nn.Sequential(nn.Linear(hidden_dim, 300), nn.ReLU(inplace=True), nn.Linear(300, 150))
        else:
            self.projection_head = nn.Sequential(nn.Linear(hidden_dim, 100), nn.ReLU(inplace=True), nn.Linear(100, 50))
    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))
    def forward(self, x0, edge_index0, edge_attr, batch):
        # this process rely on the ground truth PD, therefore only used for training

        x = self.gnn(x0, edge_index0, edge_attr)
        x = self.projection_head(x)
        x = self.pool(x,batch)
    
        return x

def pearson_correlation(x, y):

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    numerator = torch.sum((x - x_mean) * (y - y_mean))

    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2))

    correlation = numerator / denominator
    return correlation

def eval(args, model, device, loader, PI):
    model.eval()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x0 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).cpu().detach()
        distance = torch.mean(torch.sum((x0-PI)*(x0-PI),dim=1).sqrt()).cpu().detach()
        pearson = pearson_correlation(x0,PI)
        return distance, pearson

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
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'bace', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './models/tae_whole', help='filename to read the model (if there is any)')
    
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
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
    dataset = MoleculeDataset("../finetune/dataset/" + args.dataset, dataset=args.dataset)
    dict_save=None
    count=0
    with open('./PI/'+args.dataset+'_atom_Pi.pkl', 'rb') as f:
        dict_save1 = pickle.load(f)
    with open('./PI/'+args.dataset+'_degree_Pi.pkl', 'rb') as f:
        dict_save2 = pickle.load(f)
    with open('./PI/'+args.dataset+'_hks0.1_Pi.pkl', 'rb') as f:
        dict_save3 = pickle.load(f)
    smiles_list = pd.read_csv('../finetune/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_idx = []
    input_path = "../finetune/dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                            dtype='str')
    smiles_list_zinc = list(input_df['smiles'])
    zinc_id_list = list(input_df['zinc_id'])
    print(len(smiles_list))
    count = 0
    for i in range(len(smiles_list)):
        s = smiles_list[i]
        if s not in smiles_list_zinc:
            count+=1
            print('count:',count)
            train_idx.append(i)
            if(count>=1000):
                break
            
    print(train_idx)
    dataset = dataset[torch.tensor(train_idx)]
    PI = [torch.FloatTensor(dict_save1),torch.FloatTensor(dict_save2),torch.FloatTensor(dict_save3)]
    PI = torch.cat(PI,dim=1)
    PI = PI[torch.tensor(train_idx)]
    
    print(PI.shape)
    print(count)
    print(dataset)
    train_loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers = args.num_workers)
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = Topo_Model(gnn)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)
    loss = eval(args, model, device, train_loader, PI)
    print(loss)
if __name__ == "__main__":
    main()
