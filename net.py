import random

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
from layers import gcn,predict_layer,NTN,two_linear_layer,GAT,decoder,cosine_sim

class scs_GMN(torch.nn.Module):
    def __init__(self, GCN_in_size, GCN_out_size, da_size, mask=False):
        super(scs_GMN, self).__init__()
        self.GCN_in_size = GCN_in_size
        self.GCN_out_size = GCN_out_size  # D
        self.da_size = da_size
        self.mask = mask
        self.target_all_nodes = [i for i in range(da_size)]
        self.inner_product_weight = torch.nn.Parameter(torch.randn(self.GCN_out_size, self.GCN_out_size))

        #1
        self.GCN1_da = gcn(in_features=self.GCN_in_size, out_features=self.GCN_out_size)
        self.GCN1_q = gcn(in_features=self.GCN_in_size, out_features=self.GCN_out_size)
        #self.GCN1 = gcn(in_features=self.GCN_in_size, out_features=self.GCN_out_size)
        #self.GAT1 = GAT(nfeat=self.GCN_in_size,nhid=self.GCN_out_size,nclass=self.GCN_out_size,alpha=0.1,nheads=2)
        #self.cross1 = NTN(q_size=self.q_size,da_size=self.da_size,D=self.GCN_out_size,k=NTN_k)
        self.cross1 = cosine_sim(out_size=self.GCN_out_size,structure_info_count=4)
        #self.two_linear1 = two_linear_layer(target_size=self.da_size,in_size=self.GCN_out_size + self.GCN_out_size, out_size=self.GCN_out_size)
        #self.two_linear1_da = two_linear_layer(in_size=self.GCN_out_size + self.GCN_out_size, out_size=self.GCN_out_size)
        #self.two_linear1_q = two_linear_layer(in_size=self.GCN_out_size + self.GCN_out_size,out_size=self.GCN_out_size)
        self.dropout_q = nn.Dropout(p=0.3)
        self.dropout_da = nn.Dropout(p=0.3)
        self.BN_q = nn.BatchNorm1d(256)
        self.BN_da = nn.BatchNorm1d(256)

        #2
        self.GCN2_da = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        self.GCN2_q = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        #self.GCN2 = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        #self.GAT2 = GAT(nfeat=self.GCN_out_size, nhid=self.GCN_out_size, nclass=self.GCN_out_size, alpha=0.1, nheads=2)
        #self.cross2 = NTN(q_size=self.q_size,da_size=self.da_size,D=self.GCN_out_size,k=NTN_k)
        self.cross2 = cosine_sim(out_size=self.GCN_out_size,structure_info_count=4)
        #self.two_linear2 = two_linear_layer(target_size=self.da_size,in_size=self.GCN_out_size + self.q_size, out_size=self.GCN_out_size)
        # self.two_linear2_da = two_linear_layer(in_size=self.GCN_out_size + self.GCN_out_size,out_size=self.GCN_out_size)
        # self.two_linear2_q = two_linear_layer(in_size=self.GCN_out_size + self.GCN_out_size,out_size=self.GCN_out_size)

        #3
        #self.GCN3_da = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        #self.GCN3_q = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        #self.GCN3 = gcn(in_features=self.GCN_out_size, out_features=self.GCN_out_size)
        #self.GAT3 = GAT(nfeat=self.GCN_out_size, nhid=self.GCN_out_size, nclass=self.GCN_out_size, alpha=0.5, nheads=1)
        #self.cross3 = NTN(q_size=self.q_size,da_size=self.da_size,D=self.GCN_out_size,k=NTN_k)
        #self.cross3 = cosine_sim()
        #self.two_linear3 = two_linear_layer(target_size=self.da_size,in_size=self.GCN_out_size + self.q_size, out_size=self.GCN_out_size)
        #self.two_linear3_da = two_linear_layer(target_size=self.da_size, in_size=self.GCN_out_size + self.q_size,
                                            #out_size=self.GCN_out_size)
        #self.two_linear3_q = two_linear_layer(target_size=self.q_size, in_size=self.GCN_out_size + self.da_size,
                                            #out_size=self.GCN_out_size)
        #output
        #self.decoder_q = decoder(graph_size = self.q_size)
        #self.decoder_t = decoder(graph_size = self.da_size)
        #self.predict_layer = predict_layer(q_size=self.q_size,da_size=self.da_size,D=self.GCN_out_size)

    def forward(self, target_adj, node_features_da,query_adj, node_features_q,candidate_set, candidate_adj,threshold):  # b_same bx5x8 #feat=graph.ndata['x']

        # 1-layer
        da1 = self.GCN1_da(target_adj, node_features_da) # 18xD
        da1 = torch.nn.functional.leaky_relu(da1)
        q1 = self.GCN1_q(query_adj, node_features_q) # 9xD
        q1 = torch.nn.functional.leaky_relu(q1)
        #c1 = self.cross1(query_embedding=q1, target_embedding=da1,candidate_nodes=candidate_set,candidate_adj=candidate_adj)  # c1 60x400
        c1 = self.cross1(query_embedding=q1, target_embedding=da1, candidate_nodes=candidate_set,
                         candidate_adj=candidate_adj
                         , target_node_features=node_features_da, query_node_features=node_features_q,
                         structure_info_count=4)
        #T_c1 = torch.transpose(c1,0,1) #T_c1 1000x60

        #q->G
        h1_da = torch.matmul(torch.transpose(q1,0,1), c1)
        nor_h1_da = torch.nn.functional.normalize(h1_da,p=2,dim=1)
        #att_da1 = da1 + torch.transpose(nor_h1_da,0,1)
        pre_da1 = da1[candidate_set]
        da1[candidate_set].data = da1[candidate_set].data + torch.transpose(nor_h1_da, 0, 1)
        #att_da1 = torch.mul(da1,torch.transpose(nor_h1_da,0,1))
        #input_da1 = torch.cat([da1,torch.transpose(nor_h1_da,0,1)],dim=1)
        #att_da1 = self.two_linear1_da(input_da1)
        att_da1 = torch.nn.functional.leaky_relu(da1)
        #print('att_da1 :',att_da1)
        att_da1 = self.dropout_da(att_da1)
        #h1_da = torch.cat([da1,T_c1],dim=1)
        #att_da1 = self.two_linear1(h1_da)

        #G->q
        h1_q = torch.matmul(c1,pre_da1)
        nor_h1_q = torch.nn.functional.normalize(h1_q,p=2,dim=0)
        att_q1 = q1 + nor_h1_q  # 不能用+=
        #att_q1 = torch.mul(q1,nor_h1_q)
        #input_q1 = torch.cat([q1, nor_h1_q], dim=1)
        #att_q1 = self.two_linear1_q(input_q1)
        att_q1 = torch.nn.functional.leaky_relu(att_q1)
        #print('att_q1 :', att_q1)
        att_q1 = self.dropout_q(att_q1)
        # h1_q = torch.cat([q1, c1], dim=1)
        # att_q1 = self.two_linear1_q(h1_q)


        # 2-layer
        da2 = self.GCN2_da(target_adj,att_da1)
        da2 = torch.nn.functional.leaky_relu(da2)
        da2_nonzero_index = torch.nonzero(da2 < 1)
        q2 = self.GCN2_q(query_adj,att_q1)
        q2 = torch.nn.functional.leaky_relu(q2)
        q2_nonzero_index = torch.nonzero(q2 < 1)

        c2 = self.cross2(query_embedding=q2, target_embedding=da2, candidate_nodes=candidate_set,
                         candidate_adj=candidate_adj
                         , target_node_features=node_features_da, query_node_features=node_features_q,
                         structure_info_count=4)


        #q->G
        h2_da = torch.matmul(torch.transpose(q2, 0, 1), c2)
        nor_h2_da = torch.nn.functional.normalize(h2_da,p=2,dim=1)
        pre_da2 = da2[candidate_set]
        da2[candidate_set].data = da2[candidate_set].data + torch.transpose(nor_h2_da,0,1)
        att_da2 = torch.nn.functional.leaky_relu(da2)


        #G->q
        h2_q = torch.matmul(c2, pre_da2)
        nor_h2_q = torch.nn.functional.normalize(h2_q,p=2,dim=0)
        att_q2 = q2 + nor_h2_q

        att_q2 = torch.nn.functional.leaky_relu(att_q2)

        #predict_layer
        #get query graph embedding
        att_query_graph_emb = torch.mean(att_q2,dim=0,keepdim=True) #query community embedding
        #inner-product
        #end = torch.matmul(att_q2,torch.transpose(att_da2,0,1))
        #end = torch.matmul(att_query_graph_emb, torch.transpose(att_da2, 0, 1))
        end = F.cosine_similarity(att_query_graph_emb,att_da2,dim=1)
        end = torch.unsqueeze(end,0)
        #print('end',end)
        #end = torch.nn.functional.normalize(end, p=2, dim=1)
        #end = torch.sigmoid(end)
        print('end',end)

        #Predict
        #end = self.predict_layer(att_q2,att_da2)
        # matching_matrix = torch.zeros((self.q_size, self.da_size))
        #max_val, max_idx = torch.max(end, dim=1)

        #return matching_matrix
        #print('end :',end)
        #return end,re_adj,da1,c1,h1_da,da2,c2,h2_da
        #return end, re_adj, da1, c1, nor_h1_da, att_da1, att_q1, da2, c2, nor_h2_da, att_da2, att_q2

        # # get predicted community number
        predict_nodes_index = torch.nonzero(end>threshold,as_tuple=True)[1] #tuple
        predict_nodes = predict_nodes_index.tolist()
        print('predict_nodes',len(predict_nodes))

        # candidate-based filter
        # if len(predict_nodes) != 0:
        #     overlap_to_candidate = list(set(predict_nodes))
        # else:
        #     overlap_to_candidate = list(set(candidate_set))
        #overlap_to_candidate = list(set(candidate_set) & set(predict_nodes))
        overlap_to_candidate = list(set(predict_nodes))

        # reconstruct adj
        for_rec_features = att_da2[overlap_to_candidate]
        # inner product for target graph
        re_sub_adj = torch.matmul(for_rec_features, torch.transpose(for_rec_features, 0, 1))
        # add learnable weight
        #weighted_adj = torch.matmul(for_rec_features, self.inner_product_weight)
        #re_sub_adj = torch.matmul(weighted_adj, torch.transpose(for_rec_features, 0, 1))

        nor_re_adj = torch.nn.functional.normalize(re_sub_adj, p=2, dim=1)
        #nor_re_adj = torch.sigmoid(re_sub_adj)
        #print('nor_re_adj',nor_re_adj)

        #nor_neg_adj = torch.sigmoid(re_neg_adj)
        # orignal target adj
        sub_adj = target_adj[overlap_to_candidate]
        ori_sub_adj = sub_adj[:, overlap_to_candidate]

        # target graph-based filter
        nor_re_adj = torch.mul(nor_re_adj,ori_sub_adj)

        # get target graph degree\node number(indirect) approximation
        if len(overlap_to_candidate) != 0:
            sum_re_adj = torch.sum(nor_re_adj,dim=1)
            pre_avg_degree = torch.mean(sum_re_adj)
        else:
            pre_avg_degree = 0
        if len(overlap_to_candidate) != 0:
            pre_avg_edges = torch.sum(nor_re_adj)
        else:
            pre_avg_edges = 0
        if len(overlap_to_candidate) != 0:
            pre_avg_nodes = torch.trace(nor_re_adj)
        else:
            pre_avg_nodes = 0
        # density
        if len(overlap_to_candidate) != 0:
            pre_density = 2 * torch.sum(nor_re_adj)/(torch.trace(nor_re_adj)*(torch.trace(nor_re_adj)-1)+0.0001)
        else:
            pre_density = 0


        return end, att_da2, att_q2, \
                   pre_avg_degree,pre_density,pre_avg_nodes
        #return end, da1, c1, h1_da, att_da1, att_q1, da2, c2, h2_da, att_da2, att_q2



