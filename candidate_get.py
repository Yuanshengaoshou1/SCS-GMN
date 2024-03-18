import torch
import numpy as np
import networkx as nx
# import pandas as pd
import time


# target_adj = torch.Tensor(train_target_adj[7])
# query_adj = torch.Tensor(train_query_adj[7])
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

def candidate_generate_depend_on_degree(target_adj, query_adj):
    start = time.time()

    # remove self loop
    target_node_number = target_adj.shape[0]
    query_node_number = query_adj.shape[0]
    target_adj = target_adj - torch.eye(target_node_number)
    query_adj = query_adj - torch.eye(query_node_number)

    target_degree = torch.sum(target_adj, dim=1)
    query_degree = torch.sum(query_adj, dim=1)
    # print(target_degree.shape)  #400
    # print(query_degree.shape)   #60
    all_candidate = []
    for i in range(len(query_degree)):
        one_candidate = [j for j in range(len(target_degree)) if target_degree[j] >= query_degree[i]]
        all_candidate.append(set(one_candidate))
    candidate_range = list(set.union(*all_candidate))
    i = list(range(len(candidate_range)))
    candidate_mapping = dict(list(zip(candidate_range, i)))

    candidate_adj = torch.zeros(len(all_candidate), len(candidate_range))
    column_indices = [candidate_mapping[n] for one in all_candidate for n in one]
    row_indices = [i for i in range(len(all_candidate)) for j in range(len(all_candidate[i]))]
    candidate_adj[row_indices, column_indices] = 1
    # print(candidate_range)
    end = time.time()
    # print(len(candidate_range))
    return candidate_range, candidate_adj


# candidate_generate_depend_on_degree(target_adj,query_adj)

def faster_candidate_generate_depend_on_core(target_adj, query_adj):

    # remove self loop
    target_node_number = target_adj.shape[0]
    query_node_number = query_adj.shape[0]
    #target_adj = target_adj - torch.eye(target_node_number).to('cuda:1')
    #query_adj = query_adj - torch.eye(query_node_number).to('cuda:1')
    target_adj = target_adj - torch.eye(target_node_number)
    query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    np_query_adj = query_adj.cpu().numpy()
    query_graph = nx.from_numpy_array(np_query_adj)
    # print(type(query_graph.number_of_nodes())) # type-int
    number_of_query_nodes = query_graph.number_of_nodes()
    np_target_adj = target_adj.cpu().numpy()
    target_graph = nx.from_numpy_array(np_target_adj)
    # get target coreness
    target_coreness = nx.core_number(target_graph)

    # get min coreness
    query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
    remove = [node for node,degree in dict(query_graph.degree()).items() if degree == 0]
    query_graph.remove_nodes_from(remove)
    query_graph_coreness = nx.core_number(query_graph)
    # print(type(query_graph_coreness))
    min_coreness_key = min(query_graph_coreness, key=lambda k: query_graph_coreness[k])
    min_coreness_value = query_graph_coreness[min_coreness_key]
    # print(min_coreness_value)

    # get global candidate
    target_degree = torch.sum(target_adj, dim=1)
    query_degree = torch.sum(query_adj, dim=1)
    all_candidate = {}
    for i in range(len(target_degree)):
        if target_degree[i] > min_coreness_value:
        #if target_coreness[i] >= min_coreness_value:
            all_candidate[i] = int(target_degree[i])
    # print(len(list(all_candidate.keys())))

    i = list(range(len(all_candidate)))
    candidate_mapping = dict(list(zip(list(all_candidate.keys()), i)))
    # print(candidate_mapping) #397:130
    candidate_adj = torch.zeros(number_of_query_nodes, len(all_candidate))

    # sorted_result = sorted(all_candidate.items(), key=lambda e: e[1],reverse=True)
    # sorted_all_candidate = {}
    # for item in sorted_result:
    #     sorted_all_candidate[item[0]] = item[1]
    # print('sorted_all_candidate:',sorted_all_candidate)
    # print('after_sort_candidate_mapping:',candidate_mapping)
    recorded_degree = {}

    for i in range(len(query_degree)):

        if int(query_degree[i]) in recorded_degree.keys():
            candidate_adj[i] = torch.tensor(recorded_degree[int(query_degree[i])])
        else:
            degree_candidate = torch.zeros(1, len(all_candidate))
            found_key = -1
            for key in recorded_degree.keys():
                if int(query_degree[i]) > key:
                    found_key = key
            if found_key == -1:
                for key, value in all_candidate.items():
                    if value >= query_degree[i]:
                        degree_candidate[0][candidate_mapping[key]] = 1
                        # degree_candidate[0][candidate_mapping[key]] = 1/((value - query_degree[i])+1)
                recorded_degree[int(query_degree[i])] = degree_candidate
                candidate_adj[i] = torch.tensor(degree_candidate)
            else:
                degree_candidate = torch.tensor(recorded_degree[found_key])
                for new_candidate_index in range(len(degree_candidate[0])):
                    candidate_node = list(candidate_mapping.keys())[
                        list(candidate_mapping.values()).index(new_candidate_index)]
                    if all_candidate[candidate_node] < int(query_degree[i]):
                        degree_candidate[0][new_candidate_index] = 0
                recorded_degree[int(query_degree[i])] = degree_candidate
                candidate_adj[i] = torch.tensor(degree_candidate)
            sorted_recorded_degree = {}
            for key in sorted(recorded_degree, reverse=True):
                sorted_recorded_degree[key] = recorded_degree[key]
            recorded_degree = sorted_recorded_degree
    # print('after refine',candidate_adj)

    # candidate_adj_2 = torch.zeros(number_of_query_nodes, len(all_candidate))
    #     for key,value in all_candidate.items():
    #         if value >= query_degree[i]:
    #             candidate_adj_2[i][candidate_mapping[key]] = 1
    # print('before refine',candidate_adj_2)
    # print(candidate_adj.equal(candidate_adj_2))
    end = time.time()
    candidate_range = list(all_candidate.keys())
    # print(len(candidate_range))

    return candidate_range, candidate_adj


# candidate_range_1,candidate_adj_1=faster_candidate_generate_depend_on_core(target_adj,query_adj)

def candidate_generate_depend_on_core(target_adj, query_adj):
    start = time.time()

    # remove self loop
    target_node_number = target_adj.shape[0]
    query_node_number = query_adj.shape[0]
    target_adj = target_adj - torch.eye(target_node_number)
    query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    np_query_adj = query_adj.cpu().numpy()
    query_graph = nx.from_numpy_array(np_query_adj)
    # print(type(query_graph.number_of_nodes())) # type-int
    number_of_query_nodes = query_graph.number_of_nodes()

    # get min coreness
    query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
    remove = [node for node,degree in dict(query_graph.degree()).items() if degree == 0]
    query_graph.remove_nodes_from(remove)
    query_graph_coreness = nx.core_number(query_graph)
    # print(type(query_graph_coreness))
    min_coreness_key = min(query_graph_coreness, key=lambda k: query_graph_coreness[k])
    min_coreness_value = query_graph_coreness[min_coreness_key]
    # print(min_coreness_value)

    # get candidate
    target_degree = torch.sum(target_adj, dim=1)
    query_degree = torch.sum(query_adj, dim=1)
    all_candidate = {}
    for i in range(len(target_degree)):
        if target_degree[i] > min_coreness_value:
            all_candidate[i] = int(target_degree[i])
    # print(len(list(all_candidate.keys())))

    i = list(range(len(all_candidate)))
    candidate_mapping = dict(list(zip(list(all_candidate.keys()), i)))
    # print(candidate_mapping) #397:130
    candidate_adj = torch.zeros(number_of_query_nodes, len(all_candidate))
    # sorted_result = sorted(all_candidate.items(), key=lambda e: e[1],reverse=True)
    # sorted_all_candidate = {}
    # for item in sorted_result:
    #     sorted_all_candidate[item[0]] = item[1]
    # print('sorted_all_candidate:',sorted_all_candidate)
    # print('after_sort_candidate_mapping:',candidate_mapping)

    recorded_degree = {}
    for i in range(len(query_degree)):

        if int(query_degree[i]) in recorded_degree.keys():
            candidate_adj[i] = torch.tensor(recorded_degree[int(query_degree[i])])
        else:
            degree_candidate = torch.zeros(1, len(all_candidate))
            for key, value in all_candidate.items():
                if value >= query_degree[i]:
                    degree_candidate[0][candidate_mapping[key]] = 1
            recorded_degree[int(query_degree[i])] = degree_candidate
            candidate_adj[i] = torch.tensor(degree_candidate)

    # candidate_adj_2 = torch.zeros(number_of_query_nodes, len(all_candidate))
    # for i in range(number_of_query_nodes):
    #     for key,value in all_candidate.items():
    #         if value >= query_degree[i]:
    #             candidate_adj_2[i][candidate_mapping[key]] = 1
    # print('before refine',candidate_adj_2)
    # print(candidate_adj.equal(candidate_adj_2))
    end = time.time()
    candidate_range = list(all_candidate.keys())
    # print(len(candidate_range))

    return candidate_range, candidate_adj


# candidate_range_2,candidate_adj_2 = before_refine_candidate_generate_depend_on_core(target_adj,query_adj)

# print(candidate_range_1 == candidate_range_2)
# print(candidate_adj_1.equal(candidate_adj_2))

def candidate_generate_all_depend_on_core(target_adj, query_adj):
    start = time.time()

    # remove self loop
    target_node_number = target_adj.shape[0]
    query_node_number = query_adj.shape[0]
    target_adj = target_adj - torch.eye(target_node_number)
    query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    np_query_adj = query_adj.cpu().numpy()
    query_graph = nx.from_numpy_array(np_query_adj)
    # print(type(query_graph.number_of_nodes())) # type-int
    number_of_query_nodes = query_graph.number_of_nodes()

    # get min coreness
    query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
    remove = [node for node,degree in dict(query_graph.degree()).items() if degree == 0]
    query_graph.remove_nodes_from(remove)
    query_graph_coreness = nx.core_number(query_graph)
    # print(query_graph_coreness)
    min_coreness_key = min(query_graph_coreness, key=lambda k: query_graph_coreness[k])
    min_coreness_value = query_graph_coreness[min_coreness_key]
    # print(min_coreness_value)

    # get candidate
    target_degree = torch.sum(target_adj, dim=1)
    # query_degree = torch.sum(query_adj,dim=1)
    query_coreness = list(query_graph_coreness.values())
    # print(len(query_coreness))
    all_candidate = {}
    for i in range(len(target_degree)):
        if target_degree[i] > min_coreness_value:
            all_candidate[i] = int(target_degree[i])
    # print(len(list(all_candidate.keys())))

    i = list(range(len(all_candidate)))
    candidate_mapping = dict(list(zip(list(all_candidate.keys()), i)))
    # print(candidate_mapping) #397:130
    candidate_adj = torch.zeros(number_of_query_nodes, len(all_candidate))
    # sorted_result = sorted(all_candidate.items(), key=lambda e: e[1],reverse=True)
    # sorted_all_candidate = {}
    # for item in sorted_result:
    #     sorted_all_candidate[item[0]] = item[1]
    # print('sorted_all_candidate:',sorted_all_candidate)
    # print('after_sort_candidate_mapping:',candidate_mapping)

    recorded_coreness = {}
    for i in range(len(query_coreness)):

        if int(query_coreness[i]) in recorded_coreness.keys():
            candidate_adj[i] = torch.tensor(recorded_coreness[int(query_coreness[i])])
        else:
            coreness_candidate = torch.zeros(1, len(all_candidate))
            found_key = -1
            for key in recorded_coreness.keys():
                if int(query_coreness[i]) > key:
                    found_key = key
            if found_key == -1:
                for key, value in all_candidate.items():
                    if value >= query_coreness[i]:
                        coreness_candidate[0][candidate_mapping[key]] = 1
                recorded_coreness[int(query_coreness[i])] = coreness_candidate
                candidate_adj[i] = torch.tensor(coreness_candidate)
            else:
                coreness_candidate = torch.tensor(recorded_coreness[found_key])
                for new_candidate_index in range(len(coreness_candidate[0])):
                    candidate_node = list(candidate_mapping.keys())[
                        list(candidate_mapping.values()).index(new_candidate_index)]
                    if all_candidate[candidate_node] < int(query_coreness[i]):
                        coreness_candidate[0][new_candidate_index] = 0
                recorded_coreness[int(query_coreness[i])] = coreness_candidate
                candidate_adj[i] = torch.tensor(coreness_candidate)
            sorted_recorded_coreness = {}
            for key in sorted(recorded_coreness, reverse=True):
                sorted_recorded_coreness[key] = recorded_coreness[key]
            recorded_coreness[key] = sorted_recorded_coreness[key]
    # print('after refine',candidate_adj)

    # candidate_adj_2 = torch.zeros(number_of_query_nodes, len(all_candidate))
    #     for key,value in all_candidate.items():
    #         if value >= query_degree[i]:
    #             candidate_adj_2[i][candidate_mapping[key]] = 1
    # print('before refine',candidate_adj_2)
    # print(candidate_adj.equal(candidate_adj_2))
    end = time.time()
    candidate_range = list(all_candidate.keys())
    # print(len(candidate_range))

    return candidate_range, candidate_adj


# candidate_range_2,candidate_adj_2 = candidate_generate_all_depend_on_core(target_adj,query_adj)

def candidate_generate_depend_on_core_for_large_graph(target_adj, query_adj):
    start = time.time()

    # remove self loop
    target_node_number = target_adj.shape[0]
    query_node_number = query_adj.shape[0]
    #target_adj = target_adj - torch.eye(target_node_number).to('cuda:1')
    #query_adj = query_adj - torch.eye(query_node_number).to('cuda:1')
    target_adj = target_adj - torch.eye(target_node_number)
    query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    np_query_adj = query_adj.cpu().numpy()
    query_graph = nx.from_numpy_array(np_query_adj)
    # print(type(query_graph.number_of_nodes())) # type-int
    number_of_query_nodes = query_graph.number_of_nodes()

    # get min coreness
    query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
    remove = [node for node,degree in dict(query_graph.degree()).items() if degree == 0]
    query_graph.remove_nodes_from(remove)
    query_graph_coreness = nx.core_number(query_graph)
    # print(query_graph_coreness)
    #print('query_graph',nx.core_number(query_graph))
    min_coreness_key = min(query_graph_coreness, key=lambda k: query_graph_coreness[k])
    min_coreness_value = query_graph_coreness[min_coreness_key]
    #print('min_coreness_value',min_coreness_value)
    # print(min_coreness_value)

    # get candidate
    target_degree = torch.sum(target_adj, dim=1)
    # query_degree = torch.sum(query_adj,dim=1)
    query_coreness = list(query_graph_coreness.values())
    # print(len(query_coreness))
    all_candidate = {}
    for i in range(len(target_degree)):
        if target_degree[i] > min_coreness_value:
            all_candidate[i] = int(target_degree[i])
    # print(len(list(all_candidate.keys())))

    i = list(range(len(all_candidate)))
    candidate_mapping = dict(list(zip(list(all_candidate.keys()), i)))
    # print(candidate_mapping) #397:130
    candidate_adj = torch.ones(number_of_query_nodes, len(all_candidate))

    end = time.time()
    candidate_range = list(all_candidate.keys())
    # print(len(candidate_range))

    return candidate_range, candidate_adj

# candidate_generate_depend_on_core_for_large_graph(target_adj,query_adj)
