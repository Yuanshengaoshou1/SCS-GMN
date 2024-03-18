import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
#import torch

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

def cat_structure_information(features,graph,adj):

    # get degree
    target_degree = np.sum(adj, axis=1)
    list_target_degree = list(target_degree)
    list_target_degree = min_max_normalization(list_target_degree)
    # print(len(list_target_degree)) #400

    # get clustering coefficient
    target_clustering = nx.clustering(graph)
    list_target_clustering = list(target_clustering.values())

    # get h-index
    list_target_h_indexs = [get_h_index(graph, n) for n in graph.nodes()]
    list_target_h_indexs = min_max_normalization(list_target_h_indexs)

    # get k-core
    target_coreness = nx.core_number(graph)
    list_target_coreness = list(target_coreness.values())
    # print(list_target_coreness)
    list_target_coreness = min_max_normalization(list_target_coreness)

    # print('target_clustering',len(list(target_clustering.values())))
    target_features_cat_topo = []
    for j in range(features.shape[0]):
        one_cat_degree = np.append(features[j],
                                   [list_target_degree[j], list_target_clustering[j], list_target_h_indexs[j],
                                    list_target_coreness[j]])
        target_features_cat_topo.append(one_cat_degree)
    features_cat_structure = np.array(target_features_cat_topo)
    return features_cat_structure

raw_data = pd.read_csv('dataset/pubmed/pubmed.content', sep='\t', header=None)
target_size = raw_data.shape[0]
a = list(raw_data.index)
b = list(raw_data[0])
paper_id = [str(i) for i in b]
c = zip(paper_id, a)
map = dict(c)

features = raw_data.iloc[:, 1:-1]
target_features = features.to_numpy()
# cosine similarity
norm = np.linalg.norm(target_features,axis=1,keepdims=True)
norm_target_features = target_features / norm
cos_martrix = norm_target_features.dot(norm_target_features.T)
# inner product
# inner_product = target_features.dot(target_features.T)
# cos_martrix = (inner_product - np.min(inner_product))/(np.max(inner_product)-np.min(inner_product))

labels = raw_data.iloc[:,-1]

raw_data_cites = pd.read_csv('dataset/pubmed/pubmed.cites', sep='\t', header=None)
matrix = np.zeros((target_size, target_size))
for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
    x = map[str(i)]
    y = map[str(j)]
    matrix[x][y] = matrix[y][x] = 1
#print(np.nonzero(matrix)[1].shape)
#print(np.nonzero(matrix)[1].shape)

# ori graph
target_graph = nx.from_numpy_array(matrix)
target_graph.remove_edges_from(nx.selfloop_edges(target_graph))
matrix = nx.adjacency_matrix(target_graph).todense()

filter_by_ori = cos_martrix * matrix
dis_matrix = np.absolute(matrix-filter_by_ori)
# attribute graph filter
for i in range(cos_martrix.shape[0]):
    for j in range(cos_martrix.shape[1]):
        if cos_martrix[i][j] < 0.5:
            cos_martrix[i][j] = 0
        else:
            cos_martrix[i][j] = 1

att_graph1 = nx.from_numpy_array(cos_martrix)
att_graph1.remove_edges_from(nx.selfloop_edges(att_graph1))
# further process
filtered_cos_matrix = cos_martrix * matrix
att_graph2 = nx.from_numpy_array(filtered_cos_matrix)
att_graph2.remove_edges_from(nx.selfloop_edges(att_graph2))

# structure information
print('target_graph edge',nx.number_of_edges(target_graph))
print('att_graph1 edge',nx.number_of_edges(att_graph1))
print('att_graph2 edge',nx.number_of_edges(att_graph2))

print('target_graph density',nx.density(target_graph))
print('att_graph1 density',nx.density(att_graph1))
print('att_graph2 density',nx.density(att_graph2))

print('target_graph core_number',max(nx.core_number(target_graph).values()))
print('att_graph1 core_number',max(nx.core_number(att_graph1).values()))
print('att_graph2 core_number',max(nx.core_number(att_graph2).values()))

print('target_graph average_clustering',nx.average_clustering(target_graph))
print('att_graph1 average_clustering',nx.average_clustering(att_graph1))
print('att_graph2 average_clustering',nx.average_clustering(att_graph2))

print('target_graph degree_histogram',nx.degree_histogram(target_graph))
print('att_graph1 degree_histogram',nx.degree_histogram(att_graph1))
print('att_graph2 degree_histogram',nx.degree_histogram(att_graph2))

print('target_graph average_degree',sum(dict(nx.degree(target_graph)).values())/len(target_graph.nodes))
print('att_graph1 average_degree',sum(dict(nx.degree(att_graph1)).values())/len(att_graph1.nodes))
print('att_graph2 average_degree',sum(dict(nx.degree(att_graph2)).values())/len(att_graph2.nodes))

# distance graph
# plt.figure(1)
# sns.heatmap(dis_matrix,vmax=1,square=True)
#
#
# plt.savefig('C:/Users/Chen/Desktop/3.14-3.17/pubmed_dis_inner.png')
# plt.figure(2)
# sns.heatmap(matrix,vmax=1,square=True)
# plt.savefig('C:/Users/Chen/Desktop/3.14-3.17/pubmed_adj_inner.png')