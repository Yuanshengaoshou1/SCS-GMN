import torch
import networkx as nx
import numpy as np
import pandas as pd
import random
import dgl
from dgl.data.utils import save_graphs
import json
from sample_dataset import *

pickle_protocol = 4


def remove_dumb_nodes(graph):
    remove = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    graph.remove_nodes_from(remove)
    return graph


def community_similarity(target_adj, query_adj):
    # 取出查询图和目标图

    target_adj_remove_self = target_adj - np.eye(target_adj.shape[0])
    query_adj_remove_self = query_adj - np.eye(query_adj.shape[0])
    target_graph = nx.from_numpy_array(target_adj_remove_self)
    query_graph = nx.from_numpy_array(query_adj_remove_self)
    query_graph_remove_dumb_nodes = remove_dumb_nodes(query_graph)
    query_adj_remove_dumb_nodes = nx.adjacency_matrix(query_graph_remove_dumb_nodes).todense()
    query_adj_add_self = query_adj_remove_dumb_nodes + np.eye(query_adj_remove_dumb_nodes.shape[0])

    # 1st measure - avg degree
    # 计算查询图的平均度
    target_degree = dict(nx.degree(target_graph))
    target_avg_degree = sum(target_degree.values()) / len(target_graph.nodes)

    # 计算预测图的平均度
    query_degree = dict(nx.degree(query_graph_remove_dumb_nodes))
    query_avg_degree = sum(query_degree.values()) / len(query_graph_remove_dumb_nodes.nodes)

    # normalize
    # scaled_query_avg_degree,scaled_predict_avg_degree = normalize_for_commnunity_similarity(query_avg_degree,predict_avg_degree)
    # query_avg_degree = math.log(scaled_query_avg_degree,10)
    # predict_avg_degree = math.log(scaled_predict_avg_degree,10)

    # 2nd measure - edges
    target_edges = nx.number_of_edges(target_graph)
    query_edges = nx.number_of_edges(query_graph_remove_dumb_nodes)
    # normalize
    # scaled_query_edges,scaled_predict_edges = normalize_for_commnunity_similarity(query_edges,predict_edges)
    # query_edges = math.log(scaled_query_edges,10)
    # predict_edges = math.log(scaled_predict_edges,10)

    # 3rd measure - coreness
    target_coreness = nx.core_number(target_graph)
    list_target_coreness = list(target_coreness.values())
    target_avg_coreness = sum(list_target_coreness) / len(list_target_coreness)

    query_coreness = nx.core_number(query_graph_remove_dumb_nodes)
    list_query_coreness = list(query_coreness.values())
    query_avg_coreness = sum(list_query_coreness) / len(list_query_coreness)
    # normalize
    # scaled_query_avg_coreness,scaled_predict_avg_coreness = normalize_for_commnunity_similarity(query_avg_coreness,predict_avg_coreness)
    # query_avg_coreness = math.log(scaled_query_avg_coreness,10)
    # predict_avg_coreness = math.log(scaled_predict_avg_coreness,10)

    # 4th measure - nodes
    target_nodes = nx.number_of_nodes(target_graph)
    query_nodes = nx.number_of_nodes(query_graph)
    # normalize
    # scaled_query_nodes,scaled_predict_nodes = normalize_for_commnunity_similarity(query_nodes,predict_nodes)
    # query_nodes = math.log(scaled_query_nodes,10)
    # predict_nodes = math.log(scaled_predict_nodes,10)

    # 5th measure - density
    target_nodes_for_density = target_nodes
    target_edged_for_density = target_edges
    target_density = 2 * target_edged_for_density / (target_nodes_for_density * (target_nodes_for_density - 1))

    query_nodes_for_density = query_nodes
    query_edged_for_density = query_edges
    query_density = 2 * query_edged_for_density / (query_nodes_for_density * (query_nodes_for_density - 1))

    # count measure
    constant = 0.01
    # first_measure = (2*predict_avg_degree*query_avg_degree + constant)/\
    #                 (pow(predict_avg_degree,2)+pow(query_avg_degree,2)+constant)
    first_measure = (2 * target_density * query_density + constant) / \
                    (pow(target_density, 2) + pow(query_density, 2) + constant)

    second_measure = (2 * target_avg_coreness * query_avg_coreness + constant) / \
                     (pow(target_avg_coreness, 2) + pow(query_avg_coreness, 2) + constant)

    third_measure = (2 * target_nodes * query_nodes + constant) / \
                    (pow(target_nodes, 2) + pow(query_nodes, 2) + constant)
    # return first_measure * second_measure * third_measure * fourth_measure,first_measure,second_measure,third_measure,fourth_measure
    return first_measure * second_measure * third_measure


def load_facebook_data(train_size, test_size,train_path,test_path):

    max_value = -1
    with open('dataset/facebook/musae_facebook_features.json', 'r') as f:
        data_dict = json.load(f)
    target_size = len(data_dict)
    # get features dimension
    for key, value in data_dict.items():
        for one in value:
            if one > max_value:
                max_value = one
    print('max', max_value)

    target_features = np.zeros((target_size, max_value + 1))
    for key, value in data_dict.items():
        target_features[int(key)][value] = 1


    # print(target_features.shape)

    raw_data_cites = np.loadtxt('dataset/facebook/musae_facebook_edges.csv', delimiter=",", skiprows=1, dtype=int)

    matrix = np.zeros((target_size, target_size))

    for edge in raw_data_cites:
        matrix[edge[0]][edge[1]] = matrix[edge[1]][edge[0]] = 1
    # print(np.nonzero(matrix)[1].shape)

    # get target graph
    target_graph = nx.from_numpy_array(matrix)
    target_graph.remove_edges_from(nx.selfloop_edges(target_graph))
    # print(nx.core_number(target_graph))

    # k max = 56 core，k=17 3545/18 or k=43 129/60
    k_distribution, nor_k_distribution = get_k_core_distribution(target_graph)

    all_target_adjs = []
    all_query_adjs = []
    all_target_features = []
    all_query_features = []
    all_labels = []
    all_sampled_k = sample_k_from_distribution(nor_k_distribution, train_size)
    print('sampled_k ', all_sampled_k)
    all_connected_component_nodes, max_node_number = sample_core_to_query(all_sampled_k, k_distribution, target_graph)
    print('max_node_number ', max_node_number)

    for i in range(train_size):
        sampled_k = all_sampled_k[i]
        connected_component_nodes = all_connected_component_nodes[i]
        print('sampled_k', sampled_k)
        print('connected_component_nodes', connected_component_nodes)
        while (1):
            nodes_number = len(connected_component_nodes)
            if nodes_number > 30:
                sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.65))
                subgraph = target_graph.subgraph(sample_nodes)
            else:
                subgraph = target_graph.subgraph(connected_component_nodes)
            subsubgraph = nx.Graph(subgraph)
            # remove edges
            if nodes_number > 30:
                for k in range(20):
                    g_query_edge = random.choice(list(subsubgraph.edges()))
                    subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
            # public
            if nx.is_connected(subsubgraph):
                selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                # print(selected_nodes)
                query_features = target_features[selected_nodes]
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                # supplement dumb nodes
                if nx.number_of_nodes(query_graph) < max_node_number:
                    for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                        # new query graph node
                        query_graph.add_node(add_node)
                        # new query node features
                        dumb_node_features = np.zeros((1, target_features.shape[1]))
                        query_features = np.row_stack((query_features, dumb_node_features))
                query_adj_for_measure = nx.adjacency_matrix(query_graph).todense() + np.eye(
                    nx.number_of_nodes(query_graph))
                ori_graph = target_graph.subgraph(connected_component_nodes)
                ori_adj_for_measure = nx.adjacency_matrix(ori_graph).todense() + np.eye(nx.number_of_nodes(ori_graph))
                similarity_value = community_similarity(ori_adj_for_measure, query_adj_for_measure)
                print('similarity_value :', similarity_value)
                if 0.8 <= similarity_value <= 0.9:
                    print('selected_nodes', selected_nodes)
                    print('get!')
                    break

        labels = np.zeros((1, target_size))
        for node in connected_component_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        save_graph_path = './dataset/train_facebook/'
        graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')
        D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        # save_graphs(path, [D_target, D_query], graph_labels)

        query_adj = nx.adjacency_matrix(query_graph).todense()
        target_adj = nx.adjacency_matrix(target_graph).todense()
        query_adj = query_adj + np.eye(nx.number_of_nodes(query_graph))
        target_adj = target_adj + np.eye(nx.number_of_nodes(target_graph))
        ori_graph = target_graph.subgraph(connected_component_nodes)
        ori_adj = nx.adjacency_matrix(ori_graph).todense() + np.eye(nx.number_of_nodes(ori_graph))
        print('similarity_value :', community_similarity(ori_adj, query_adj))

        if i == 0:
            all_target_adjs.append(target_adj)
            all_target_features.append(target_features)

        all_query_adjs.append(query_adj)
        all_query_features.append(query_features)
        all_labels.append(labels)
        print(i)

    fin_target_features = np.array(all_target_features)
    fin_target_features = fin_target_features.astype(np.float32)

    fin_target_adjs = np.array(all_target_adjs)
    fin_target_adjs = fin_target_adjs.astype(np.float32)

    fin_query_features = np.array(all_query_features)
    fin_query_features = fin_query_features.astype(np.float32)

    fin_query_adjs = np.array(all_query_adjs)
    fin_query_adjs = fin_query_adjs.astype(np.float32)

    fin_labels = np.array(all_labels)
    fin_labels = fin_labels.astype(np.float32)
    # print(type(fin_target_features[0][0][0]))

    torch.save(fin_target_features, train_path + 'target_features.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_target_adjs, train_path + 'target_adj.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_query_features, train_path + 'query_features.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_query_adjs, train_path + 'query_adj.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_labels, train_path + 'labels.pt', pickle_protocol=pickle_protocol)

    # sample test data
    all_target_adjs = []
    all_query_adjs = []
    all_target_features = []
    all_query_features = []
    all_labels = []
    component_number = len(all_connected_component_nodes)
    sampled_number = random.sample(range(component_number), test_size)
    print('sampled_number', sampled_number)
    test_connected_component_nodes = []
    test_sampled_k = []
    for one in sampled_number:
        test_connected_component_nodes.append(all_connected_component_nodes[one])
        test_sampled_k.append(all_sampled_k[one])

    for i in range(test_size):
        sampled_k = test_sampled_k[i]
        connected_component_nodes = test_connected_component_nodes[i]
        print('sampled_k', sampled_k)
        print('connected_component_nodes', connected_component_nodes)
        while (1):
            nodes_number = len(connected_component_nodes)
            if nodes_number > 30:
                sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.65))
                subgraph = target_graph.subgraph(sample_nodes)
            else:
                subgraph = target_graph.subgraph(connected_component_nodes)
            subsubgraph = nx.Graph(subgraph)
            # remove edges
            if nodes_number > 30:
                for k in range(20):
                    g_query_edge = random.choice(list(subsubgraph.edges()))
                    subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
            # public
            if nx.is_connected(subsubgraph):
                selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                # print(selected_nodes)
                query_features = target_features[selected_nodes]
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                # supplement dumb nodes
                if nx.number_of_nodes(query_graph) < max_node_number:
                    for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                        # new query graph node
                        query_graph.add_node(add_node)
                        # new query node features
                        dumb_node_features = np.zeros((1, target_features.shape[1]))
                        query_features = np.row_stack((query_features, dumb_node_features))
                query_adj_for_measure = nx.adjacency_matrix(query_graph).todense() + np.eye(
                    nx.number_of_nodes(query_graph))
                ori_graph = target_graph.subgraph(connected_component_nodes)
                ori_adj_for_measure = nx.adjacency_matrix(ori_graph).todense() + np.eye(nx.number_of_nodes(ori_graph))
                similarity_value = community_similarity(ori_adj_for_measure, query_adj_for_measure)
                print('similarity_value :', similarity_value)
                if 0.8 <= similarity_value <= 0.9:
                    print('selected_nodes', selected_nodes)
                    print('get!')
                    break

        labels = np.zeros((1, target_size))
        for node in connected_component_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        save_graph_path = './dataset/test_facebook/'
        graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')
        D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        # save_graphs(path, [D_target, D_query], graph_labels)

        query_adj = nx.adjacency_matrix(query_graph).todense()
        target_adj = nx.adjacency_matrix(target_graph).todense()
        query_adj = query_adj + np.eye(nx.number_of_nodes(query_graph))
        target_adj = target_adj + np.eye(nx.number_of_nodes(target_graph))

        if i == 0:
            all_target_adjs.append(target_adj)
            all_target_features.append(target_features)

        all_query_adjs.append(query_adj)
        all_query_features.append(query_features)
        all_labels.append(labels)
        print(i)

    fin_target_features = np.array(all_target_features)
    fin_target_features = fin_target_features.astype(np.float32)

    fin_target_adjs = np.array(all_target_adjs)
    fin_target_adjs = fin_target_adjs.astype(np.float32)

    fin_query_features = np.array(all_query_features)
    fin_query_features = fin_query_features.astype(np.float32)

    fin_query_adjs = np.array(all_query_adjs)
    fin_query_adjs = fin_query_adjs.astype(np.float32)

    fin_labels = np.array(all_labels)
    fin_labels = fin_labels.astype(np.float32)
    # print(type(fin_target_features[0][0][0]))

    torch.save(fin_target_features, test_path + 'target_features.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_target_adjs, test_path + 'target_adj.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_query_features, test_path + 'query_features.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_query_adjs, test_path + 'query_adj.pt',
               pickle_protocol=pickle_protocol)
    torch.save(fin_labels, test_path + 'labels.pt', pickle_protocol=pickle_protocol)


train_size = 30
test_size = 20
# load_facebook_data(train_size, test_size)