import torch
import numpy as np
import networkx as nx
from dgl.data.utils import load_graphs
import os

# pickle_protocol=4
#
# train_path = './dataset/for_train_self_build/'
# test_path = './dataset/for_test_self_build/'
# train_path_for_citeseer = './dataset/for_train_citeseer/0_3/'
# test_path_for_citeseer = './dataset/for_test_citeseer/0_3/'
# train_path_for_cora = './dataset/for_train_cora/'
# test_path_for_cora = './dataset/for_test_cora/'
# train_path_for_pubmed = './dataset/for_train_pubmed/target_1w_query_22/'
# test_path_for_pubmed = './dataset/for_test_pubmed/target_1w_query_22/'
# train_path_for_deezer = './dataset/for_train_deezer/0_3/'
# test_path_for_deezer = './dataset/for_case_deezer/'
# train_path_for_facebook = './dataset/for_train_facebook/0_2/'
# test_path_for_facebook = './dataset/for_case_facebook/'
#
# train_target_features = torch.load(test_path_for_facebook+'target_features.pt')
# train_target_adj = torch.load(test_path_for_facebook+'target_adj.pt')
# train_query_features = torch.load(test_path_for_facebook+'masked_0.7_query_features.pt')
# train_query_adj = torch.load(test_path_for_facebook+'query_adj.pt')

def get_h_index(graph,u):

    hi = 0
    ns = {n: graph.degree[n] for n in graph[u]}
    # Sorted the neighbors in increasing order with node degree.
    sns = sorted(zip(ns.keys(), ns.values()), key=lambda x: x[1], reverse=True)
    # print(sns)
    for i, n in enumerate(sns):
        if i >= n[1]:
            hi = i
            break
        hi = i + 1
    return hi

def min_max_normalization(data_features):
    max_value = max(data_features)
    min_value = min(data_features)
    if max_value != min_value:
        for i in range(len(data_features)):
            data_features[i] = (data_features[i]-min_value)/(max_value-min_value)
    else:
        for i in range(len(data_features)):
            data_features[i] = data_features[i] / max_value
    return data_features
def cat_structural_information(graph_features,graph_adj):

    # features cat structure info
    target_graph = nx.from_numpy_array(graph_adj)
    target_features = graph_features

    # get degree
    target_degree = np.sum(graph_adj, axis=1)
    list_target_degree = list(target_degree)
    list_target_degree = min_max_normalization(list_target_degree)
    # print(len(list_target_degree)) #400

    # get clustering coefficient
    target_clustering = nx.clustering(target_graph)
    list_target_clustering = list(target_clustering.values())

    # get h-index
    list_target_h_indexs = [get_h_index(target_graph, n) for n in target_graph.nodes()]
    list_target_h_indexs = min_max_normalization(list_target_h_indexs)

    # get k-core
    target_coreness = nx.core_number(target_graph)
    list_target_coreness = list(target_coreness.values())
    # print(list_target_coreness)
    list_target_coreness = min_max_normalization(list_target_coreness)
    
    target_features_cat_topo = []
    for j in range(target_features.shape[0]):
        one_cat_degree = np.append(target_features[j],
                                   [list_target_degree[j], list_target_clustering[j], list_target_h_indexs[j],
                                    list_target_coreness[j]])
        target_features_cat_topo.append(one_cat_degree)
    return np.array(target_features_cat_topo)

    