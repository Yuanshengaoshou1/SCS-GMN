import torch
import networkx as nx
import numpy as np
import pandas as pd
import random
import dgl
from dgl.data.utils import save_graphs
import json
from sample_dataset import *
from mask_feature_for_dgl import process_to_masked_features
from cat_structural_info_for_dgl import cat_structural_information

pickle_protocol = 4


def load_facebook_data(train_size, test_size):
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
        matrix[edge[0]][edge[1]] = matrix[edge[1]][edge[0]] = 1  # 有引用关系的样本点之间取1
    # print(np.nonzero(matrix)[1].shape)

    # get target graph
    target_graph = nx.from_numpy_array(matrix)
    target_graph.remove_edges_from(nx.selfloop_edges(target_graph))
    target_adj = nx.adjacency_matrix(target_graph).todense()
    target_features_cat_stru = cat_structural_information(target_features,target_adj)
    # print(nx.core_number(target_graph))

    # k max = 56 different core，k=17 3545/18 or k=43 129/60
    k_distribution, nor_k_distribution = get_k_core_distribution(target_graph)
    for i in range(len(nor_k_distribution)):
        if i != 45:
            nor_k_distribution[i] = 0
        else:
            nor_k_distribution[i] = 1
    # all_target_adjs = []
    # all_query_adjs = []
    # all_target_features = []
    # all_query_features = []
    # all_labels = []
    all_labels = []
    all_query_graph = []
    all_target_graph = []
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
            if nodes_number > 30 and sampled_k < 43:
                sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.96))
                subgraph = target_graph.subgraph(sample_nodes)
            else:
                subgraph = target_graph.subgraph(connected_component_nodes)
            subsubgraph = nx.k_core(subgraph, k=sampled_k)
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # remove edges
            if sampled_k > 45:
                for k in range(10):
                    g_query_edge = random.choice(list(subsubgraph.edges()))
                    subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
            subsubgraph = nx.k_core(subsubgraph, k=sampled_k)
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # print(nx.number_of_nodes(subsubgraph))
            flag = False
            if nodes_number > 30 and sampled_k < 43:
                print('int(nodes_number * 0.9)',int(nodes_number * 0.9))
                if nx.number_of_nodes(subsubgraph) == int(nodes_number * 0.9):
                    flag = True
            else:
                if nx.number_of_nodes(subsubgraph) == nodes_number:
                    flag = True
            # public
            if nx.is_connected(subsubgraph) and flag:
                # print(nx.is_connected(subsubgraph))
                # selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                selected_nodes = list(nx.nodes(subsubgraph))
                print(selected_nodes)
                query_features = target_features[selected_nodes]
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                query_features = process_to_masked_features(query_features,masked_range=0.8)
                query_adj = nx.adjacency_matrix(query_graph).todense()
                query_features = cat_structural_information(query_features,query_adj)
                # supplement dumb nodes
                # if nx.number_of_nodes(query_graph) < max_node_number:
                #     for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                #         # new query graph node
                #         query_graph.add_node(add_node)
                #         # new query node features
                #         dumb_node_features = np.zeros((1, target_features.shape[1]))
                #         query_features = np.row_stack((query_features, dumb_node_features))
                break
                # label
        labels = np.zeros((1, target_size))
        for node in selected_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        all_labels.append(labels)
        # print(np.nonzero(labels))
        if i == 0:
            D_target = dgl.from_networkx(target_graph)
            D_target.ndata['feat'] = torch.tensor(target_features_cat_stru)
            all_target_graph.append(D_target)
        D_query = dgl.from_networkx(query_graph)
        D_query.ndata['feat'] = torch.tensor(query_features)
        all_query_graph.append(D_query)
        
    graph_labels = {'glabel': torch.tensor(all_labels)}
    # dgl.save_graphs('/mnt/HDD/crh/facebook/for_train_facebook/0.1/for_train_facebook_target.bin',all_target_graph)
    # dgl.save_graphs('/mnt/HDD/crh/facebook/for_train_facebook/0.1/for_train_facebook_query.bin',all_query_graph,graph_labels)
    print('save done')
        # save_graph_path = './dataset/train_facebook/'
        # graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        # D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')  # 这个是有向图
        # D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        # path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        # save_graphs(path, [D_target, D_query], graph_labels)

    #     query_adj = nx.adjacency_matrix(query_graph).todense()
    #     target_adj = nx.adjacency_matrix(target_graph).todense()
    #     query_adj = query_adj + np.eye(nx.number_of_nodes(query_graph))
    #     target_adj = target_adj + np.eye(nx.number_of_nodes(target_graph))

    #     if i == 0:
    #         all_target_adjs.append(target_adj)
    #         all_target_features.append(target_features)

    #     all_query_adjs.append(query_adj)
    #     all_query_features.append(query_features)
    #     all_labels.append(labels)
    #     print(i)

    # fin_target_features = np.array(all_target_features)
    # fin_target_features = fin_target_features.astype(np.float32)

    # fin_target_adjs = np.array(all_target_adjs)
    # fin_target_adjs = fin_target_adjs.astype(np.float32)

    # fin_query_features = np.array(all_query_features)
    # fin_query_features = fin_query_features.astype(np.float32)

    # fin_query_adjs = np.array(all_query_adjs)
    # fin_query_adjs = fin_query_adjs.astype(np.float32)

    # fin_labels = np.array(all_labels)
    # fin_labels = fin_labels.astype(np.float32)
    # # print(type(fin_target_features[0][0][0]))

    # torch.save(fin_target_features, train_path + 'target_features.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_target_adjs, train_path + 'target_adj.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_query_features, train_path + 'query_features.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_query_adjs, train_path + 'query_adj.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_labels, train_path + 'labels.pt', pickle_protocol=pickle_protocol)

    # sample test data
    # all_target_adjs = []
    # all_query_adjs = []
    # all_target_features = []
    # all_query_features = []
    # all_labels = []
    all_labels = []
    all_query_graph = []
    all_target_graph = []
    
    component_number = len(all_connected_component_nodes)
    sampled_number = random.sample(range(component_number), test_size)
    print('sampled_number', sampled_number)
    test_connected_component_nodes = []
    test_sampled_k = []
    for one in sampled_number:
        test_connected_component_nodes.append(all_connected_component_nodes[one])
        test_sampled_k.append(all_sampled_k[one])
    # case
    # while(len(test_sampled_k)<test_size):
    #     component_number = len(all_connected_component_nodes)
    #     sampled_number = random.sample(range(component_number), 1)
    #     if all_sampled_k[sampled_number[0]] == 43 and len(all_connected_component_nodes[sampled_number[0]])==60:
    #         print('all_sampled_k[sampled_number]',all_sampled_k[sampled_number[0]])
    #         test_connected_component_nodes.append(all_connected_component_nodes[sampled_number[0]])
    #         test_sampled_k.append(all_sampled_k[sampled_number[0]])

    for i in range(test_size):
        sampled_k = test_sampled_k[i]
        connected_component_nodes = test_connected_component_nodes[i]
        print('sampled_k', sampled_k)
        print('connected_component_nodes', connected_component_nodes)
        while (1):
            nodes_number = len(connected_component_nodes)
            if nodes_number > 30 and sampled_k < 43:
                sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.96))  # 此时节点编号为原图编号
                subgraph = target_graph.subgraph(sample_nodes)
            else:
                subgraph = target_graph.subgraph(connected_component_nodes)
            subsubgraph = nx.k_core(subgraph, k=sampled_k)
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # remove edges
            if sampled_k > 45:
                for k in range(10):
                    g_query_edge = random.choice(list(subsubgraph.edges()))  # 返回tuple
                    subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
                subsubgraph = nx.k_core(subsubgraph, k=sampled_k)
                if nx.number_of_nodes(subsubgraph) == 0:
                    continue
            # print(nx.number_of_nodes(subsubgraph))
            flag = False
            if nodes_number > 30 and sampled_k < 43:
                if nx.number_of_nodes(subsubgraph) == int(nodes_number * 0.9):
                    flag = True
            else:
                if nx.number_of_nodes(subsubgraph) == nodes_number:
                    flag = True
            # public
            if nx.is_connected(subsubgraph) and flag:
                # selected_nodes = sorted(list(nx.nodes(subsubgraph)))  # 对应原图的节点编号
                selected_nodes = list(nx.nodes(subsubgraph))
                print(selected_nodes)
                query_features = target_features[selected_nodes]
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                query_features = process_to_masked_features(query_features,masked_range=0.8)
                query_adj = nx.adjacency_matrix(query_graph).todense()
                query_features = cat_structural_information(query_features,query_adj)
                # supplement dumb nodes
                # if nx.number_of_nodes(query_graph) < max_node_number:
                #     for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                #         # new query graph node
                #         query_graph.add_node(add_node)
                #         # new query node features
                #         dumb_node_features = np.zeros((1, target_features.shape[1]))
                #         query_features = np.row_stack((query_features, dumb_node_features))
                break
                # label
        labels = np.zeros((1, target_size))
        for node in selected_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        all_labels.append(labels)
        # print(np.nonzero(labels))
        if i == 0:
            D_target = dgl.from_networkx(target_graph)
            D_target.ndata['feat'] = torch.tensor(target_features_cat_stru)
            all_target_graph.append(D_target)          
        D_query = dgl.from_networkx(query_graph)
        D_query.ndata['feat'] = torch.tensor(query_features)
        all_query_graph.append(D_query)

    graph_labels = {'glabel': torch.tensor(all_labels)}
    dgl.save_graphs('/mnt/HDD/crh/facebook/for_test_facebook/0.1/more_test_samples/for_test_facebook_target_45.bin',all_target_graph)
    dgl.save_graphs('/mnt/HDD/crh/facebook/for_test_facebook/0.1/more_test_samples/for_test_facebook_query_45.bin',all_query_graph,graph_labels)
    print('save done')
        # save_graph_path = './dataset/test_facebook/'
        # graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        # D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')  # 这个是有向图
        # D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        # path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        # save_graphs(path, [D_target, D_query], graph_labels)

    #     query_adj = nx.adjacency_matrix(query_graph).todense()
    #     target_adj = nx.adjacency_matrix(target_graph).todense()
    #     query_adj = query_adj + np.eye(nx.number_of_nodes(query_graph))
    #     target_adj = target_adj + np.eye(nx.number_of_nodes(target_graph))

    #     if i == 0:
    #         all_target_adjs.append(target_adj)
    #         all_target_features.append(target_features)

    #     all_query_adjs.append(query_adj)
    #     all_query_features.append(query_features)
    #     all_labels.append(labels)
    #     print(i)

    # fin_target_features = np.array(all_target_features)
    # fin_target_features = fin_target_features.astype(np.float32)

    # fin_target_adjs = np.array(all_target_adjs)
    # fin_target_adjs = fin_target_adjs.astype(np.float32)

    # fin_query_features = np.array(all_query_features)
    # fin_query_features = fin_query_features.astype(np.float32)

    # fin_query_adjs = np.array(all_query_adjs)
    # fin_query_adjs = fin_query_adjs.astype(np.float32)

    # fin_labels = np.array(all_labels)
    # fin_labels = fin_labels.astype(np.float32)
    # # print(type(fin_target_features[0][0][0]))

    # torch.save(fin_target_features, test_path + 'target_features.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_target_adjs, test_path + 'target_adj.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_query_features, test_path + 'query_features.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_query_adjs, test_path + 'query_adj.pt', pickle_protocol=pickle_protocol)
    # torch.save(fin_labels, test_path + 'labels.pt', pickle_protocol=pickle_protocol)


train_size = 110
test_size = 100
load_facebook_data(train_size, test_size)