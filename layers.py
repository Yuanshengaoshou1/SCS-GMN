import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import dgl.function as fn
from dgl.data import MiniGCDataset
import dgl.function as fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn.pytorch import SumPooling
import numpy as np
from dgl.data.utils import save_graphs, get_download_dir, load_graphs
# from dgl.subgraph import DGLSubGraph
from torch.utils.data import Dataset, DataLoader
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear

#GCN
class gcn(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(gcn, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    def forward(self, adj_matrix, node_features):
        adj_matrix = adj_matrix.to(torch.float32)
        node_features = node_features.to(torch.float32)
        adj_norm = F.normalize(adj_matrix, p=1, dim=1)
        #T_node_features = torch.transpose(node_features,0,1)
        output = torch.mm(adj_matrix, node_features)
        output = torch.mm(output, self.weight)
        #output = torch.sigmoid(output)
        #activated_output = torch.nn.functional.leaky_relu(output)
        return output

#GAT
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_size, out_size,alpha,concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_size
        self.out_features = out_size
        self.concat = concat
        self.alpha = alpha

        self.gat_w = nn.Parameter(torch.zeros(size=(in_size, out_size)))
        nn.init.xavier_uniform_(self.gat_w.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, input, adj):
        input = input.float()
        h = torch.mm(input, self.gat_w) # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, alpha, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, alpha=alpha, concat=False)

    def forward(self, adj, x):
        #x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# class cross_sim(torch.nn.Module):
#     def __init__(self, D):
#         super(cross_sim, self).__init__()
#         self.D = D
#         self.setup_weights()
#         self.init_parameters()
#
#     def setup_weights(self):
#         """
#         Defining weights.
#         """
#         self.cross_sim_weight = torch.nn.Parameter(torch.Tensor(self.D, self.D))
#
#     def init_parameters(self):
#         """
#         Initializing weights.
#         """
#         torch.nn.init.xavier_uniform_(self.cross_sim_weight)
#
#     def forward(self, batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
#         T_batch_da_em = torch.transpose(batch_da_em, 0, 1)
#         cross = torch.matmul(batch_q_em, self.cross_sim_weight) # 9xD
#         cross = torch.matmul(cross, T_batch_da_em) # cross.shape -> 9x18
#         cross = torch.nn.functional.normalize(cross, p=2, dim=1)
#         return cross  # cross bx1x5x18

class cross_sim(torch.nn.Module):
    def __init__(self, D):
        super(cross_sim, self).__init__()
        self.D = D
        self.cross_sim_weight = torch.nn.Parameter(torch.randn(self.D, self.D))

    def forward(self, batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
        #learnable weight
        T_batch_da_em = torch.transpose(batch_da_em, 0, 1)

        #weighted_inner_product
        cross = torch.matmul(batch_q_em, self.cross_sim_weight) # 9xD
        cross = torch.matmul(cross, T_batch_da_em) # cross.shape -> 9x18
        #cross = torch.mm(batch_q_em,T_batch_da_em)
        cross = torch.nn.functional.normalize(cross, p=2, dim=1)

        #cosine_similarity
        # batch_q_em = batch_q_em/torch.norm(batch_q_em,dim=-1,keepdim=True)
        # batch_da_em = batch_da_em/torch.norm(batch_da_em,dim=-1,keepdim=True)
        # T_batch_da_em = torch.transpose(batch_da_em, 0, 1)
        # cross = torch.mm(batch_q_em,T_batch_da_em)
        return cross  # cross bx1x5x18

def att_layer(batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
    D = batch_q_em.size()[2]
    T_batch_da_em = torch.transpose(batch_da_em, 1, 2)
    att = torch.matmul(batch_q_em, T_batch_da_em)
    att = att / (D ** 0.5)
    att = torch.nn.functional.softmax(att, dim=2).unsqueeze(1)
    return att  # att bx1x5x18

class linear_layer(torch.nn.Module):
    def __init__(self,in_size,out_size):
        super(linear_layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.attention_weight = torch.nn.Parameter(torch.randn(self.in_size, self.out_size))

    def forward(self,target_embedding):

        output = torch.mm(target_embedding, self.attention_weight)
        fin_output = torch.sigmoid(output)

        return fin_output

class two_linear_layer(torch.nn.Module):
    def __init__(self,in_size,out_size):
        super(two_linear_layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.Linear1 = linear_layer(self.in_size,self.out_size)
        self.Linear2 = linear_layer(self.out_size,self.out_size)

    def forward(self,target_embedding):
        h1 = self.Linear1(target_embedding)
        h2 = self.Linear2(h1)

        return h2

class cosine_sim(torch.nn.Module):

    def __init__(self,out_size, structure_info_count):
        super(cosine_sim, self).__init__()
        self.out_size = out_size
        self.structure_info_count = structure_info_count
        self.cos_weight = torch.nn.Parameter(torch.randn(self.out_size, self.out_size))
        self.structure_weight = torch.nn.Parameter(torch.randn(self.structure_info_count, self.out_size))

    #def forward(self,query_embedding,target_embedding,candidate_nodes,candidate_adj):
    def forward(self, query_embedding, target_embedding, candidate_nodes, candidate_adj, target_node_features,
                query_node_features, structure_info_count):

        #common operation
        candidated_target_embedding = target_embedding[candidate_nodes]
        tran_target_embedding = torch.matmul(candidated_target_embedding,self.cos_weight)
        tran_query_embedding = torch.matmul(query_embedding, self.cos_weight)
        nor_query_embedding = F.normalize(tran_query_embedding)
        nor_target_embedding = F.normalize(tran_target_embedding)
        #nor_target_embedding_cat_structure = torch.cat([nor_target_embedding,candidated_target_structure_info],dim=1)
        #nor_query_embedding_cat_structure = torch.cat([nor_query_embedding,query_structure_info],dim=1)
        T_target_embedding = torch.transpose(nor_target_embedding,0,1)
        #T_target_embedding = torch.transpose(nor_target_embedding_cat_structure, 0, 1)
        cos_sim = torch.matmul(nor_query_embedding,T_target_embedding)
        #cos_sim = torch.matmul(nor_query_embedding_cat_structure, T_target_embedding)
        masked_cos_sim = torch.mul(cos_sim,candidate_adj)
        #print('len(self.all_sim) :',len(all_sim))
        return masked_cos_sim

class NTN(torch.nn.Module):
    def __init__(self, q_size, da_size, D, k):
        super(NTN, self).__init__()
        self.k = k
        self.D = D
        self.q_size = q_size
        self.da_size = da_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.w = torch.nn.Parameter(torch.Tensor(self.k, self.D, self.D))
        #self.V = torch.nn.Parameter(torch.Tensor(self.k, 2 * self.D))
        #self.b = torch.nn.Parameter(torch.Tensor(self.k, 1, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.w)
        #torch.nn.init.xavier_uniform_(self.V)
        #torch.nn.init.xavier_uniform_(self.b)

    def forward(self, batch_q_em, batch_da_em):
        batch_q_em_adddim = torch.unsqueeze(batch_q_em, 0)
        #print('result.shape', batch_q_em_adddim.shape) torch.Size([1, 60, 128])
        batch_da_em_adddim = torch.unsqueeze(batch_da_em, 0)
        #print('result.shape', batch_da_em_adddim.shape) torch.Size([1, 400, 128])
        T_batch_da_em_adddim = torch.transpose(batch_da_em_adddim, 1, 2)  # trans T_batch_da_em _adddim bx1xcx18   torch.tensor
        #print('result.shape', T_batch_da_em_adddim.shape) torch.Size([1, 128, 400])
        # first part
        first = torch.matmul(batch_q_em_adddim, self.w)
        first = torch.matmul(first, T_batch_da_em_adddim)
        #print('result.shape', first.shape)
        # first part
        # # second part
        # ed_batch_q_em = torch.unsqueeze(batch_q_em, 2)
        # ed_batch_q_em = ed_batch_q_em.repeat(1, 1, self.da_size, 1)
        # ed_batch_q_em = ed_batch_q_em.reshape(-1, self.q_size * self.da_size, self.D)
        #
        # ed_batch_da_em = torch.unsqueeze(batch_da_em, 1)
        # ed_batch_da_em = ed_batch_da_em.repeat(1, self.q_size, 1, 1)
        # ed_batch_da_em = ed_batch_da_em.reshape(-1, self.q_size * self.da_size, self.D)
        #
        # mid = torch.cat([ed_batch_q_em, ed_batch_da_em], 2)
        # mid = torch.transpose(mid, 1, 2)
        # mid = torch.matmul(self.V, mid)
        # mid = mid.reshape(-1, self.k, self.q_size, self.da_size)  # mid bxkx5x18
        # second part
        #end = first + mid + self.b

        sum_tensor = torch.sum(first,dim=0)
        #print('result.shape', sum_tensor.shape) # torch.Size([60, 400])
        result = torch.sigmoid(sum_tensor)
        #print('result.shape',result.shape)
        return result  # end bxkx5x18

class predict_layer(torch.nn.Module):

    def __init__(self,q_size, da_size, D):
        super(predict_layer, self).__init__()
        self.in_size = q_size
        self.out_size = da_size
        self.D = D
        self.predict_layer_weight = torch.nn.Parameter(torch.randn(self.D, self.D))
        self.predict_layer_bias = torch.nn.Parameter(torch.randn(self.in_size, self.out_size))

    def forward(self,q_embedding,da_embedding):

        T_da_embedding = torch.transpose(da_embedding,0,1)

        output = torch.matmul(q_embedding,self.predict_layer_weight)
        output = torch.matmul(output, T_da_embedding)

        sum_output = output + self.predict_layer_bias
        fin_output = torch.nn.functional.normalize(sum_output, p=2, dim=1)

        return fin_output

class decoder(torch.nn.Module):
    def __init__(self,graph_size):
        super(decoder, self).__init__()
        self.in_size = graph_size
        self.out_size = graph_size
        self.Linear1 = linear_layer(in_size=self.in_size,out_size=self.out_size)
        self.Linear2 = linear_layer(in_size=self.out_size,out_size=self.out_size)

    def forward(self,embedding):
        h1 = self.Linear1(embedding)
        h2 = self.Linear2(h1)

        return h2