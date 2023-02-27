import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, GCNConv, SAGEConv
#from Knowledge_Distillation.gat_conv import GATConv
#from torch_geometric.nn import GATConv
from scipy.spatial.distance import cityblock
import wasserstein as gdw
#from kmeans_pytorch_my import kmeans
from visualize_PD import draw_PD
from pimg import PersistenceImager
from sg2dgm.PersistenceImager import PersistenceImager as PersistenceImager_nograd
import time
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_batch, to_dense_adj
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        return self.propagate(edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class Topo_Model(torch.nn.Module):
    def __init__(self, hidden_dim = 32, out_dim = 25, num_models = 3, dropout = 0.2, type = 'GIN'):
        super(Topo_Model, self).__init__()

        self.gnn = GNN(num_layer=5, emb_dim=hidden_dim, JK='last', drop_ratio=dropout, gnn_type = 'gin')

        self.lin5 = Linear(2 * hidden_dim, hidden_dim)
        #self.lin5 = Linear(2 * hidden_dim, 2)
        self.lin6 = Linear(hidden_dim, 2)
        self.num_models = num_models
        self.lin1 = Linear(2, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.lin3 = Linear(hidden_dim, hidden_dim)
        self.lin4 = Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.pers_imager = PersistenceImager(resolution=5)
        self.pers_imager_nograd = PersistenceImager_nograd(resolution=5)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))
    def forward(self, x0, edge_index0, edge_attr, batch, PD, kernel = 'sliced', M = 50, p = 1, pair_diagonal = False, draw_fig = False, fig_name = '', compute_loss = True, grad_PI = True):
        # this process rely on the ground truth PD, therefore only used for training

        t1 = time.time()
        #print(x0.device)
        #print(edge_index0.device)
        #print(next(self.DIM0_Model.parameters()).is_cuda)#False
        # print(x0.shape,edge_index0.shape,edge_attr.shape)
        x = self.gnn(x0, edge_index0, edge_attr)
        x_in = x[edge_index0[0]]
        x_out = x[edge_index0[1]]
        x = self.lin5(torch.cat((x_in, x_out), dim = -1))
        x = F.prelu(x, weight = torch.tensor(0.1).cuda())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin6(x)
        x_init = x
        odd_rows = x[1::2, :]
        even_rows = x[::2, :]
        x = (odd_rows + even_rows)/2
        t2 = time.time()

        if compute_loss:
            if not pair_diagonal:
                loss0, _, loss_xy0, loss_xd0, loss_yd0 = self.compute_PD_loss(x, PD, kernel = kernel, M = M, p = p, num_models = 1)
            else:
                loss0, _, loss_xy0, loss_xd0, loss_yd0 = self.compute_PD_loss(x, PD, kernel=kernel, M=M, p=p, num_models=1, type = 'inference')
        else:
            loss0 = None; loss_xy0 = None; loss_xd0 = None; loss_yd0 = None


        if draw_fig:
            if not pair_diagonal:
                draw_PD(PD1=x.detach().cpu().numpy(), save_name='./train_' + fig_name + '.png', PD2=PD.cpu().numpy())
            else:
                draw_PD(PD1=x.detach().cpu().numpy(), save_name='./test_' + fig_name + '.png', PD2=PD.cpu().numpy())


        x0 = x
        
        if grad_PI:
            adj=to_dense_adj(edge_index=edge_index0, edge_attr=x_init, batch=batch)
            adj=adj+adj.transpose(-2, -3)
            adj[:, torch.tril(torch.ones(adj.shape[1], adj.shape[2]), diagonal=-1) == 1] = 0
            indices = torch.nonzero(torch.sum(adj != torch.Tensor([0, 0]).cuda(), dim=-1))
            sizes = degree(indices[:, 0], dtype=torch.long).tolist()
            result_PD=adj[indices[:, 0], indices[:, 1], indices[:, 2]].split(sizes, 0)
            x = []
            for idx in range(len(result_PD)):
                x_temp = result_PD[idx]
                x_temp = self.pers_imager.transform(x_temp).reshape(-1)
                x.append(x_temp)
        #     x = self.pers_imager.transform(x.detach().cpu(), use_cuda = True).reshape(-1).cuda()
        # else:
        #     x = torch.tensor(self.pers_imager_nograd.transform(np.array(x.detach().cpu())).reshape(-1)).cuda()
            x = torch.cat(x)

        t3 = time.time()
        '''
        x = self.lin1(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim = 0)
        x = self.lin4(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        '''
        #x = F.normalize(x, dim = -1)

        return x0, x, loss0, loss_xy0, loss_xd0, loss_yd0, t2 - t1, t3 - t2


    def compute_PD_loss(self, PD1, PD2, p = 1, M = 50, kernel = 'wasserstein', num_models = 1, type = 'train'):
        # M is for sliced
        # p denotes p-wassersein
        if kernel == 'sliced':
            diag_theta = torch.FloatTensor([np.cos(0.25 * np.pi), np.sin(0.25 * np.pi)]).cuda()
            l_theta1 = [torch.dot(diag_theta, x) for x in PD1]
            l_theta2 = [torch.dot(diag_theta, x) for x in PD2]
            PD_delta1 = [[torch.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta1]
            PD_delta2 = [[torch.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta2]
            loss = torch.FloatTensor([0]).cuda()
            theta = 0.5
            step = 1.0 / M
            for i in range(M):
                l_theta = torch.FloatTensor([np.cos(theta * np.pi), np.sin(theta * np.pi)]).cuda()
                V1 = [torch.dot(l_theta, x) for x in PD1] + [l_theta[0] * x[0] + l_theta[1] * x[1] for x in PD_delta2]
                V2 = [torch.dot(l_theta, x) for x in PD2] + [l_theta[0] * x[0] + l_theta[1] * x[1] for x in PD_delta1]
                loss += step * cityblock(sorted(V1), sorted(V2))
                theta += step
        elif kernel == 'wasserstein':
            loss = torch.FloatTensor([0]).cuda()
            loss_xy = torch.FloatTensor([0]).cuda(); loss_xd = torch.FloatTensor([0]).cuda(); loss_yd = torch.FloatTensor([0]).cuda();
            # first choice
            #loss += gdw.wasserstein_distance(PD1, PD2, order=p, enable_autodiff=True)[0]
            # second choice
            if type == 'train':
                temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance(PD1, PD2, order=p, enable_autodiff=True, num_models = num_models)
                loss += temp_loss
                loss_xy += wxy; loss_xd += wxd; loss_yd += wyd
            else:
                temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance_inference(PD1, PD2, order=p, enable_autodiff=True)
                loss += temp_loss
                loss_xy += wxy; loss_xd += wxd; loss_yd += wyd
        return loss, ind_tmp_test, loss_xy, loss_xd, loss_yd


