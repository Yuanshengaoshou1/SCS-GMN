import torch
import networkx as nx
import numpy as np
import pandas as pd
import random
import dgl
from dgl.data.utils import save_graphs
import os
import json
# from sample_dataset import *
import sys
import time
from torch_geometric.utils import from_networkx
from mask_feature import process_to_masked_features

pickle_protocol=4
def check_nodes(node_list1,node_list2):

    intersection_nodes = set(node_list1) & set(node_list2)
    if len(intersection_nodes) == 0:
        print('node_list1',node_list1)
        print('node_list2',node_list2)
        return True
    else:
        return False
    
def get_k_core_distribution(graph):
    coreness = nx.core_number(graph)
    max_corenss = max(list(coreness.values()))
    coreness_distribution = [0] * (max_corenss+1)
    # low_bound_corenss = math.ceil(max_corenss * 0.7) #citeseer pubmed facebook deezer 0.7/cora 0.8
    low_bound_corenss = 30
    for i in range(low_bound_corenss, max_corenss + 1):
        if i == 150 or i == 200 or i == 212:
            print('k', i)
            core_graph = nx.k_core(graph, k=i)
            count = 0
            for j in nx.connected_components(core_graph):
                count = count + 1
            coreness_distribution[i] = count
            print('count',count)
            if count == 1:
                print("connected components' length",len(j))
    sum_coress = sum(coreness_distribution)
    nor_coreness_distribution = [item/sum_coress for item in coreness_distribution]
    return coreness_distribution,nor_coreness_distribution

def get_reddit_data(train_size,test_size,train_path,test_path):

    # if os.path.isfile('./dataset/reddit/target_graph.gml') == True:
    #     print('there exists target graph')
    #
    #     Graph = nx.read_gml('./dataset/reddit/target_graph.gml')
    #     print(nx.number_of_nodes(Graph))
    #     print(nx.number_of_edges(Graph))
    # else:
    dataName = 'reddit'
    # nodes
    with open(os.path.join('./dataset/reddit/', 'id_map.json'), 'r', encoding='utf8') as fp:
        name_id = json.load(fp)
        id_name = {id: name for name, id in name_id.items()}
        nodes = list(name_id.values())
        NodeNum = len(nodes)
        fp.close()

    # edges
    with open(os.path.join('./dataset/reddit', '{}.cites').format(dataName), 'r') as fp:
        edges = []
        line = fp.readline()
        while line:
            line = line.split()
            edges.append([name_id.get(line[0]), name_id.get(line[1])])
            line = fp.readline()
        EdgeNum = len(edges)
        fp.close()
    target_graph = nx.Graph()
    target_graph.add_nodes_from(nodes)
    target_graph.add_edges_from(edges)
    print(nx.number_of_edges(target_graph))
    target_graph.remove_edges_from(nx.selfloop_edges(target_graph))
    print(nx.number_of_edges(target_graph))
    del nodes
    del edges
    # print('start save graph')
    # nx.write_gml(Graph,'./dataset/reddit/target_graph.gml')


    # features
    target_features = np.load(os.path.join('./dataset/reddit/', 'features.npy')).astype(np.float32)
    FeatureShape = target_features.shape[1]
    print(FeatureShape)
    # sys.exit()
    # coreness_distribution,nor_coreness_distribution = get_k_core_distribution(Graph)
    # for i in range(len(coreness_distribution)):
    #     print("coreness_distribution_{coreness_idx}:{component_number}".format(coreness_idx= i,component_number = coreness_distribution[i]))
    print('get base query graph')
    base_k = 202
    base_query_graph = nx.k_core(target_graph,k=base_k)
    # print('nx.number_of_edges(base_query_graph)',nx.number_of_nodes(base_query_graph))
    query_node_list = list(base_query_graph.nodes())
    # sub_graph = target_graph.subgraph(query_node_list)
    # print('nx.number_of_edges(sub_graph)',nx.number_of_nodes(sub_graph))
    node_number = len(query_node_list)
    node_dict = {}
    edge_dict = {}
    test_round = 1
    found_count = 0

    # validate cover rate
    # while test_round < 10000:
    #     print('test round',test_round)
    #     sample_node_number = int(node_number * 0.05)
    #     # print('sample_node_number',sample_node_number)
    #     sample_node = np.random.choice(query_node_list, size=sample_node_number).tolist()
    #     # print('sample_node',sample_node)
    #     sub_graph = Graph.subgraph(sample_node)
    #     min_coreness = min(nx.core_number(sub_graph).values())
    #     subgraph_edges = nx.number_of_edges(sub_graph)
    #     subgraph_nodes = sub_graph.nodes()
        # if min_coreness > 8:
    #     print('get core')
    #     if min_coreness > 7:
    #         if min_coreness in edge_dict:
    #             print('type', type(edge_dict[min_coreness]))
    #             if subgraph_edges in edge_dict[min_coreness]:
    #                 record_node_lists = node_dict[min_coreness]
    #                 if len(record_node_lists) > 0:
    #                     for i in range(len(record_node_lists)):
    #                         one_node_list = record_node_lists[i]
    #                         flag = check_nodes(subgraph_nodes, one_node_list)
    #                         if flag == True:
    #                             found_count = found_count + 1
    #                             print('found', found_count)
    #                             print('found rate', found_count / test_round)
    #                             break
    #             node_dict[min_coreness].append(subgraph_nodes)
    #             edge_dict[min_coreness].append(subgraph_edges)
    #
    #         else:
    #             edge_list_in_dict = []
    #             edge_list_in_dict.append(subgraph_edges)
    #             edge_dict[min_coreness] = edge_list_in_dict
    #             node_list_in_dict = []
    #             node_list_in_dict.append(subgraph_nodes)
    #             node_dict[min_coreness] = node_list_in_dict
    #         test_round = test_round + 1
    # print('found rate', found_count / test_round)

    # validate mean coreness
    # while test_round < 100:
    #     print('test round',test_round)
    #     sample_node_number = int(node_number * 0.1)
    #     # print('sample_node_number',sample_node_number)
    #     sample_node = np.random.choice(query_node_list, size=sample_node_number).tolist()
    #     # print('sample_node',sample_node)
    #     sub_graph = Graph.subgraph(sample_node)
    #     min_coreness = min(nx.core_number(sub_graph).values())
    #     subgraph_edges = nx.number_of_edges(sub_graph)
    #     subgraph_nodes = sub_graph.nodes()
    #     print('min_coreness',min_coreness)
    #     print('subgraph_edges',subgraph_edges)
    #     print('subgraph_nodes',subgraph_nodes)
    #     test_round = test_round + 1
    start_get_data = time.time()
    # train_size = 60
    # all_node_number_range = [int(node_number * 0.1),int(node_number * 0.2)]
    # all_node_number_range = [node_number,int(node_number*0.9),int(node_number*0.8)]
    all_node_number_range = [node_number]
    # max_node_number = int(node_number * 0.2)
    max_node_number = node_number
    sample_node_number_range = np.random.choice(all_node_number_range,train_size)

    # for train
    all_labels = []
    all_query_graph = []
    all_target_graph = []
    for i in range(train_size):
        print('sample number', i)
        sample_node_number = sample_node_number_range[i]
        # print('sample_node_number',sample_node_number)

        while (1):
            # sample_node_number = sample_node_number_range[i]
            # print('sample_node_number',sample_node_number)
            sample_node = np.random.choice(query_node_list, size=sample_node_number).tolist()
            # sample_node = query_node_list
            # print('sample_node',sample_node)
            sub_graph = target_graph.subgraph(sample_node)
            min_coreness = min(nx.core_number(sub_graph).values())
            # subgraph_edges = nx.number_of_edges(sub_graph)
            # subgraph_nodes = sub_graph.nodes()
            # subgraph_nodes = nx.number_of_nodes(sub_graph)
            # subgraph_density = 2 * subgraph_edges/(subgraph_nodes * (subgraph_nodes-1))
            # subgraph_avg_coreness = sum(nx.core_number(sub_graph).values())/subgraph_nodes
            print('min_coreness',min_coreness)
            # print('subgraph_edges',subgraph_edges)
            # print('subgraph_nodes',subgraph_nodes)
            if min_coreness > 10:
                sampled_k = min_coreness
                print('sampled_k',sampled_k)
                connected_component_nodes = sample_node
                print('before operate nodes',len(sample_node))
                nodes_number = len(connected_component_nodes)
                # print('sampled_k', sampled_k)
                # print('connected_component_nodes', len(connected_component_nodes))
                if nodes_number > 10:
                    # sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.8))
                    sample_nodes = connected_component_nodes
                    subgraph = target_graph.subgraph(sample_nodes)
                else:
                    subgraph = target_graph.subgraph(connected_component_nodes)
                subsubgraph = nx.k_core(subgraph, k=sampled_k)
                print('after remove nodes',nx.number_of_nodes(subsubgraph))
                print('after remove nodes',nx.number_of_edges(subsubgraph))
                if nx.number_of_nodes(subsubgraph) == 0:
                    continue
                # remove edges
                if nodes_number > 10:
                    for k in range(100):
                        g_query_edge = random.choice(list(subsubgraph.edges()))
                        subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
                elif 5 < nodes_number < 10:
                    for k in range(2):
                        g_query_edge = random.choice(list(subsubgraph.edges()))
                        subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
                subsubgraph = nx.k_core(subsubgraph, k=sampled_k)
                print('after remove edges',nx.number_of_nodes(subsubgraph))
                print('after remove edges',nx.number_of_edges(subsubgraph))
                if nx.number_of_nodes(subsubgraph) == 0:
                    continue
                # flag = False
                # # print(nx.number_of_nodes(subsubgraph))
                # if nodes_number > 10:
                #     print('check subsubgraph node number')
                    
                #     if nx.number_of_nodes(subsubgraph) == int(nodes_number * 0.9):
                #         flag = True
                #     else:
                #         print('nx.number_of_nodes(subsubgraph)',nx.number_of_nodes(subsubgraph))
                #         print('nodes_number * 0.9',nodes_number * 0.9)
                #         print('False')
                # else:
                #     if nx.number_of_nodes(subsubgraph) == nodes_number:
                #         flag = True
                # public
                # if nx.is_connected(subsubgraph) and flag:
                if nx.is_connected(subsubgraph):
                    # selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                    selected_nodes = list(nx.nodes(subsubgraph))
                    print('selected_nodes',selected_nodes)
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
        # querygraph_edges = nx.number_of_edges(query_graph)
        # querygraph_nodes = nx.number_of_nodes(query_graph)
        # querygraph_density = 2 * querygraph_edges/(querygraph_nodes * (querygraph_nodes-1))
        # querygraph_avg_coreness = sum(nx.core_number(query_graph).values())/querygraph_nodes
        # print('subgraph min coreness',min(nx.core_number(sub_graph).values()))
        # print('querygraph min coreness',min(nx.core_number(query_graph).values()))
        # constant = 0.01
        # first_measure = (2*subgraph_density*querygraph_density + constant)/\
        #         (pow(subgraph_density,2)+pow(querygraph_density,2)+constant)
        # print('first',first_measure)
        # second_measure = (2*subgraph_avg_coreness*querygraph_avg_coreness + constant)/\
        #         (pow(subgraph_avg_coreness,2)+pow(querygraph_avg_coreness,2)+constant)
        # print('second',second_measure)
        # third_measure = (2*subgraph_nodes*querygraph_nodes + constant)/\
        #         (pow(subgraph_nodes,2)+pow(querygraph_nodes,2)+constant)
        # print('third',third_measure)
        # final = first_measure * second_measure * third_measure
        # print('final',final)
        labels = np.zeros((1, nx.number_of_nodes(target_graph)))
        for node in selected_nodes:
            labels[0][node] = 1
        all_labels.append(labels)
        # print(np.nonzero(labels))
        if i == 0:
            D_target = dgl.from_networkx(target_graph)
            D_target.ndata['feat'] = torch.tensor(target_features)
            all_target_graph.append(D_target)
        D_query = dgl.from_networkx(query_graph)
        D_query.ndata['feat'] = torch.tensor(query_features)
        all_query_graph.append(D_query)
        # graph_labels = {'glabel': torch.tensor(labels, dtype=torch.float32)}
        # if i == 0:
        #     start_time = time.time()
        #     D_target = dgl.from_networkx(target_graph)
        #     print('end dgl',time.time()-start_time)
            
        #     start_time = time.time()
        #     D_target = from_networkx(target_graph)
        #     print('end torch',time.time()-start_time)
        # D_query = dgl.DGLGraph(query_graph, ntype='_N', etype='_E')
        # path = save_graph_path + 'target_3312_query_mixed_' + str(i) + '.bin'
        # save_graphs(path, [D_target, D_query], graph_labels)
    start_save_time = time.time()
    graph_labels = {'glabel': torch.tensor(all_labels)}
    dgl.save_graphs('./dataset/reddit/for_train_reddit/0.1/for_train_reddit_target_bigger.bin',all_target_graph)
    dgl.save_graphs('./dataset/reddit/for_train_reddit/0.1/for_train_reddit_query_bigger.bin',all_query_graph,graph_labels)
    print('save done')
    print('save time',time.time() - start_save_time)

    # for test
    # all_node_number_range = [int(node_number * 0.1), int(node_number * 0.2)]
    # all_node_number_range = [node_number,int(node_number*0.9),int(node_number*0.8)]
    all_node_number_range = [node_number]
    # max_node_number = int(node_number * 0.2)
    max_node_number = node_number
    sample_node_number_range = np.random.choice(all_node_number_range, train_size)

    # test_size = 5
    all_labels = []
    all_query_graph = []
    all_target_graph = []
    
    for i in range(test_size):
        print('sample number',i)
        sample_node_number = sample_node_number_range[i]
        # print('sample_node_number',sample_node_number)

        while (1):
            # sample_node_number = sample_node_number_range[i]
            # print('sample_node_number',sample_node_number)
            sample_node = np.random.choice(query_node_list, size=sample_node_number).tolist()
            # sample_node = query_node_list
            # print('sample_node',sample_node)
            sub_graph = target_graph.subgraph(sample_node)
            min_coreness = min(nx.core_number(sub_graph).values())
            # subgraph_edges = nx.number_of_edges(sub_graph)
            # subgraph_nodes = sub_graph.nodes()
            # subgraph_nodes = nx.number_of_nodes(sub_graph)
            # subgraph_density = 2 * subgraph_edges/(subgraph_nodes * (subgraph_nodes-1))
            # subgraph_avg_coreness = sum(nx.core_number(sub_graph).values())/subgraph_nodes
            print('min_coreness',min_coreness)
            # print('subgraph_edges',subgraph_edges)
            # print('subgraph_nodes',subgraph_nodes)
            if min_coreness > 10:
                sampled_k = min_coreness
                print('sampled_k',sampled_k)
                connected_component_nodes = sample_node
                nodes_number = len(connected_component_nodes)
                # print('sampled_k', sampled_k)
                # print('connected_component_nodes', len(connected_component_nodes))
                if nodes_number > 10:
                    # sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.8))
                    sample_nodes = connected_component_nodes
                    subgraph = target_graph.subgraph(sample_nodes)
                else:
                    subgraph = target_graph.subgraph(connected_component_nodes)
                subsubgraph = nx.k_core(subgraph, k=sampled_k)
                print('after remove nodes',nx.number_of_nodes(subsubgraph))
                print('after remove nodes',nx.number_of_edges(subsubgraph))
                if nx.number_of_nodes(subsubgraph) == 0:
                    continue
                # remove edges
                if nodes_number > 10:
                    for k in range(100):
                        g_query_edge = random.choice(list(subsubgraph.edges()))
                        subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
                elif 5 < nodes_number < 10:
                    for k in range(2):
                        g_query_edge = random.choice(list(subsubgraph.edges()))
                        subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
                subsubgraph = nx.k_core(subsubgraph, k=sampled_k)
                print('after remove edges',nx.number_of_nodes(subsubgraph))
                print('after remove edges',nx.number_of_edges(subsubgraph))
                if nx.number_of_nodes(subsubgraph) == 0:
                    continue
                # flag = False
                # # print(nx.number_of_nodes(subsubgraph))
                # if nodes_number > 10:
                #     print('nx.number_of_nodes(subsubgraph)',nx.number_of_nodes(subsubgraph))
                #     if nx.number_of_nodes(subsubgraph) == int(nodes_number * 0.9):
                #         flag = True
                # else:
                #     if nx.number_of_nodes(subsubgraph) == nodes_number:
                #         flag = True
                # public
                # if nx.is_connected(subsubgraph) and flag:
                if nx.is_connected(subsubgraph) :
                    # selected_nodes = sorted(list(nx.nodes(subsubgraph)))
                    selected_nodes = list(nx.nodes(subsubgraph))
                    print('selected_nodes',selected_nodes)
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
        labels = np.zeros((1, nx.number_of_nodes(target_graph)))
        for node in selected_nodes:
            labels[0][node] = 1
        all_labels.append(labels)
        if i == 0:
            D_target = dgl.from_networkx(target_graph)
            D_target.ndata['feat'] = torch.tensor(target_features)
            all_target_graph.append(D_target)
        D_query = dgl.from_networkx(query_graph)
        D_query.ndata['feat'] = torch.tensor(query_features)
        all_query_graph.append(D_query)


    start_save_time = time.time()
    graph_labels = {'glabel': torch.tensor(all_labels)}
    dgl.save_graphs('./dataset/reddit/for_test_reddit/0.1/for_test_reddit_target_bigger.bin',all_target_graph)
    dgl.save_graphs('./dataset/reddit/for_test_reddit/0.1/for_test_reddit_query_bigger.bin',all_query_graph,graph_labels)
    print('save done')
    print('save time',time.time() - start_save_time)
    print('end get data',time.time()-start_get_data)
train_path = ''
test_path = ''
train_size = 60
test_size = 20
get_reddit_data(train_size,test_size,train_path,test_path)

# k=150之后只有1个core


