import torch
import networkx as nx
import numpy as np
import pandas as pd
import random
import dgl
import math

def get_k_core_distribution(graph):
    coreness = nx.core_number(graph)
    max_corenss = max(list(coreness.values()))
    coreness_distribution = [0] * (max_corenss+1)
    low_bound_corenss = math.ceil(max_corenss * 0.8) #citeseer pubmed facebook deezer 0.7/cora dblp 0.8
    for i in range(low_bound_corenss, max_corenss + 1):
        print('k',i)
        core_graph = nx.k_core(graph, k=i)
        count = 0
        for j in nx.connected_components(core_graph):
            count = count + 1
            print('len(j)',len(j))
        coreness_distribution[i] = count
    sum_coress = sum(coreness_distribution)
    nor_coreness_distribution = [item/sum_coress for item in coreness_distribution]
    return coreness_distribution,nor_coreness_distribution

def sample_k_from_distribution(distribution,data_size=1):
    max_corenss = len(distribution)-1
    samples = np.random.choice(range(max_corenss+1), size=data_size, p=distribution).tolist()
    return samples

def sample_core_to_query(sampled_k,distribution,graph):
    all_connected_component_nodes = []
    max_nodes_number = -1
    for k in sampled_k:
        core_graph = nx.k_core(graph,k=k)
        all_components = distribution[k]
        selected_components = random.randint(1,all_components)
        count = 0
        for i in nx.connected_components(core_graph):
            count = count + 1
            if count == selected_components:
                connected_component_nodes = list(i)
                if len(connected_component_nodes) > max_nodes_number:
                    max_nodes_number = len(connected_component_nodes)
                all_connected_component_nodes.append(connected_component_nodes)
                break
    return all_connected_component_nodes,max_nodes_number