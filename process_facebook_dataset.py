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
    # print(nx.core_number(target_graph))

    # k max = 56 different core，k=17 3545/18 or k=43 129/60
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
            if nodes_number > 30 and sampled_k < 43:
                sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.96))
                subgraph = target_graph.subgraph(sample_nodes)
            else:
                subgraph = target_graph.subgraph(connected_component_nodes)
            subsubgraph = nx.k_core(subgraph, k=sampled_k)
            print('b', nx.number_of_nodes(subsubgraph))
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
                print(nx.is_connected(subsubgraph))
                selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                print(selected_nodes)
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
                break
                # label
        labels = np.zeros((1, target_size))
        for node in connected_component_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        save_graph_path = './dataset/train_facebook/'
        graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')  # 这个是有向图
        D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        save_graphs(path, [D_target, D_query], graph_labels)

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

    torch.save(fin_target_features, './dataset/for_train_facebook/target_features.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_target_adjs, './dataset/for_train_facebook/target_adj.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_query_features, './dataset/for_train_facebook/query_features.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_query_adjs, './dataset/for_train_facebook/query_adj.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_labels, './dataset/for_train_facebook/labels.pt', pickle_protocol=pickle_protocol)

    # sample test data
    all_target_adjs = []
    all_query_adjs = []
    all_target_features = []
    all_query_features = []
    all_labels = []
    # component_number = len(all_connected_component_nodes)
    # sampled_number = random.sample(range(component_number), test_size)
    #print('sampled_number', sampled_number)
    test_connected_component_nodes = []
    test_sampled_k = []
    # for one in sampled_number:
    #     test_connected_component_nodes.append(all_connected_component_nodes[one])
    #     test_sampled_k.append(all_sampled_k[one])
    while(len(test_sampled_k)<test_size):
        component_number = len(all_connected_component_nodes)
        sampled_number = random.sample(range(component_number), 1)
        if all_sampled_k[sampled_number[0]] == 43 and len(all_connected_component_nodes[sampled_number[0]])==60:
            print('all_sampled_k[sampled_number]',all_sampled_k[sampled_number[0]])
            test_connected_component_nodes.append(all_connected_component_nodes[sampled_number[0]])
            test_sampled_k.append(all_sampled_k[sampled_number[0]])

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
            print('b', nx.number_of_nodes(subsubgraph))
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
                selected_nodes = sorted(list(nx.nodes(subsubgraph)))  # 对应原图的节点编号
                print(selected_nodes)
                query_features = target_features[selected_nodes]
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
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
        for node in connected_component_nodes:
            labels[0][node] = 1
        # print(np.nonzero(labels))
        save_graph_path = './dataset/test_facebook/'
        graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}

        D_target = dgl.DGLGraph(target_graph, ntype='_N', etype='_E')  # 这个是有向图
        D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        path = save_graph_path + 'target_2w_query_mixed_' + str(i) + '.bin'
        save_graphs(path, [D_target, D_query], graph_labels)

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

    torch.save(fin_target_features, './dataset/for_case_facebook/target_features.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_target_adjs, './dataset/for_case_facebook/target_adj.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_query_features, './dataset/for_case_facebook/query_features.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_query_adjs, './dataset/for_case_facebook/query_adj.pt', pickle_protocol=pickle_protocol)
    torch.save(fin_labels, './dataset/for_case_facebook/labels.pt', pickle_protocol=pickle_protocol)


train_size = 30
test_size = 1
load_facebook_data(train_size, test_size)