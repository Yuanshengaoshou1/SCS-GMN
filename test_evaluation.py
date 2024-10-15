import numpy as np
import torch
import networkx as nx
from collections import Counter
import math
import dgl

def remove_dumb_nodes(graph):
    remove = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    graph.remove_nodes_from(remove)
    return graph


def label_cover_rate(predict_result, label):
    list_predict_result = predict_result[0].tolist()
    list_label_result = label[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    label_nodes = []
    for i in range(len(list_label_result)):
        if list_label_result[i] == 1:
            label_nodes.append(i)
    covered_nodes = []
    for node in predict_nodes:
        if node in label_nodes:
            covered_nodes.append(node)
    return len(covered_nodes) / len(label_nodes)


def normalize_for_commnunity_similarity(query_value, predict_value):
    # compute mean value
    mean_value = (query_value + predict_value) / 2
    scale_factor = 5 / mean_value  # normalize to the half of 10
    scaled_query_value = query_value * scale_factor
    scaled_predict_value = predict_value * scale_factor
    return scaled_query_value, scaled_predict_value


def community_similarity(target_adj, query_adj, predict_result):
    # get query and target
    # target_adj = target_adj.cpu().numpy()
    # query_adj = query_adj.cpu().numpy()
    # target_adj_remove_self = target_adj - np.eye(target_adj.shape[0])
    # query_adj_remove_self = query_adj - np.eye(query_adj.shape[0])
    # target_graph = nx.from_numpy_array(target_adj_remove_self)
    # query_graph = nx.from_numpy_array(query_adj_remove_self)
    query_graph = nx.Graph(dgl.to_networkx(query_adj.cpu()))
    query_graph_remove_dumb_nodes = remove_dumb_nodes(query_graph)
    query_adj_remove_dumb_nodes = nx.adjacency_matrix(query_graph_remove_dumb_nodes).todense()

    list_predict_result = predict_result[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    print('predict_nodes', predict_nodes)
    # extract predict subgraph
    predict_subgraph = nx.Graph(dgl.to_networkx(dgl.node_subgraph(target_adj,predict_nodes).cpu()))
    # predict_subgraph = target_graph.subgraph(predict_nodes)
    # predict_sub_adj = target_adj[predict_nodes]
    # predict_adj = predict_sub_adj[:, predict_nodes]

    # 1st measure - avg degree
    query_degree = dict(nx.degree(query_graph_remove_dumb_nodes))
    query_avg_degree = sum(query_degree.values()) / len(query_graph_remove_dumb_nodes.nodes)

    predict_degree = dict(nx.degree(predict_subgraph))
    if len(predict_subgraph.nodes) == 0:
        predict_avg_degree = 0
    else:
        predict_avg_degree = sum(predict_degree.values()) / len(predict_subgraph.nodes)

    # normalize
    # scaled_query_avg_degree,scaled_predict_avg_degree = normalize_for_commnunity_similarity(query_avg_degree,predict_avg_degree)
    # query_avg_degree = math.log(scaled_query_avg_degree,10)
    # predict_avg_degree = math.log(scaled_predict_avg_degree,10)

    # 2nd measure - edges
    query_edges = nx.number_of_edges(query_graph_remove_dumb_nodes)
    if len(predict_subgraph.nodes) == 0:
        predict_edges = 0
    else:
        predict_edges = nx.number_of_edges(predict_subgraph)
    # normalize
    # scaled_query_edges,scaled_predict_edges = normalize_for_commnunity_similarity(query_edges,predict_edges)
    # query_edges = math.log(scaled_query_edges,10)
    # predict_edges = math.log(scaled_predict_edges,10)

    # 3rd measure - coreness
    query_coreness = nx.core_number(query_graph_remove_dumb_nodes)
    list_query_coreness = list(query_coreness.values())
    query_avg_coreness = sum(list_query_coreness) / len(list_query_coreness)
    if len(predict_subgraph.nodes) == 0:
        predict_avg_coreness = 0
    else:
        predict_coreness = nx.core_number(predict_subgraph)
        list_predict_coreness = list(predict_coreness.values())
        predict_avg_coreness = sum(list_predict_coreness) / (len(list_predict_coreness))
    # normalize
    # scaled_query_avg_coreness,scaled_predict_avg_coreness = normalize_for_commnunity_similarity(query_avg_coreness,predict_avg_coreness)
    # query_avg_coreness = math.log(scaled_query_avg_coreness,10)
    # predict_avg_coreness = math.log(scaled_predict_avg_coreness,10)

    # 4th measure - nodes
    query_nodes = nx.number_of_nodes(query_graph)
    if len(predict_subgraph.nodes) == 0:
        predict_nodes = 0
    else:
        predict_nodes = nx.number_of_nodes(predict_subgraph)
    # normalize
    # scaled_query_nodes,scaled_predict_nodes = normalize_for_commnunity_similarity(query_nodes,predict_nodes)
    # query_nodes = math.log(scaled_query_nodes,10)
    # predict_nodes = math.log(scaled_predict_nodes,10)

    # 5th measure - density
    query_nodes_for_density = query_nodes
    query_edged_for_density = query_edges
    query_density = 2 * query_edged_for_density / (query_nodes_for_density * (query_nodes_for_density - 1))
    # query_density = query_edged_for_density / query_nodes_for_density
    
    predict_nodes_for_density = predict_nodes
    predict_edged_for_density = predict_edges
    predict_density = 2 * predict_edged_for_density / (
                predict_nodes_for_density * (predict_nodes_for_density - 1) + 0.0001)
    # predict_density = predict_edged_for_density / (predict_nodes_for_density  + 0.0001)
    # count measure
    constant = 0.01
    # first_measure = (2*predict_avg_degree*query_avg_degree + constant)/\
    #                 (pow(predict_avg_degree,2)+pow(query_avg_degree,2)+constant)
    first_measure = (2 * predict_density * query_density + constant) / \
                    (pow(predict_density, 2) + pow(query_density, 2) + constant)

    second_measure = (2 * predict_avg_coreness * query_avg_coreness + constant) / \
                     (pow(predict_avg_coreness, 2) + pow(query_avg_coreness, 2) + constant)

    third_measure = (2 * predict_nodes * query_nodes + constant) / \
                    (pow(predict_nodes, 2) + pow(query_nodes, 2) + constant)
    
    # get coreness
    min_query_coreness = min(list_query_coreness)
    if len(predict_subgraph.nodes) == 0:
        min_predict_coreness = 0
    else:
        min_predict_coreness = min(list_predict_coreness)
    print('min_query_coreness',min_query_coreness,'min_predict_coreness',min_predict_coreness)
    print('query_density', query_density, 'predict_density', predict_density)
    print('query_avg_coreness', query_avg_coreness, 'predict_avg_coreness', predict_avg_coreness)
    print('query_nodes', query_nodes, 'predict_nodes', predict_nodes)
    print('first_measure', first_measure, 'second_measure', second_measure, 'third_measure', third_measure)
    print('community similarity',first_measure*second_measure*third_measure)
    # return first_measure * second_measure * third_measure ,first_measure,second_measure,third_measure
    return first_measure * second_measure * third_measure, first_measure, second_measure, third_measure, \
           query_density, predict_density, query_avg_coreness, predict_avg_coreness, query_nodes, predict_nodes,min_query_coreness,min_predict_coreness


def found_more_than_threshold(predict_result, label):
    list_predict_result = predict_result[0].tolist()
    list_label_result = label[0].tolist()
    predict_nodes = []

    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    print('more_than_threshold_nodes', predict_nodes)

    label_nodes = []
    for i in range(len(list_label_result)):
        if list_label_result[i] == 1:
            label_nodes.append(i)
    covered_nodes = []
    for node in predict_nodes:
        if node in label_nodes:
            covered_nodes.append(node)

    return len(covered_nodes) / len(label_nodes)


def triangle_participation_ratio(query_adj, target_adj, predict_result):
    # get query and target
    target_adj = target_adj.cpu().numpy()
    query_adj = query_adj.cpu().numpy()
    target_adj_remove_self = target_adj - np.eye(target_adj.shape[0])
    query_adj_remove_self = query_adj - np.eye(query_adj.shape[0])
    target_graph = nx.from_numpy_array(target_adj_remove_self)
    query_graph = nx.from_numpy_array(query_adj_remove_self)
    query_graph_remove_dumb_nodes = remove_dumb_nodes(query_graph)

    list_predict_result = predict_result[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    # extract predict subgraph
    predict_subgraph = target_graph.subgraph(predict_nodes)

    # query TPR
    query_tpn = nx.triangles(query_graph_remove_dumb_nodes)
    query_tpn_nodes = 0
    for key, value in query_tpn.items():
        if value != 0:
            query_tpn_nodes = query_tpn_nodes + 1
    query_tpr = query_tpn_nodes / float(nx.number_of_nodes(query_graph_remove_dumb_nodes))

    # predict TPR
    predict_tpn = nx.triangles(predict_subgraph)
    predict_tpn_nodes = 0
    for key, value in predict_tpn.items():
        if value != 0:
            predict_tpn_nodes = predict_tpn_nodes + 1
    predict_tpr = predict_tpn_nodes / (float(nx.number_of_nodes(predict_subgraph)) + 0.001)
    print('query_tpr', query_tpr, 'predict_tpr', predict_tpr)
    com_tpr = (2 * predict_tpr * query_tpr) / (pow(predict_tpr, 2) + pow(query_tpr, 2))
    return query_tpr, predict_tpr


def diameter(query_adj, target_adj, predict_result):
    # get query and target
    target_adj = target_adj.cpu().numpy()
    query_adj = query_adj.cpu().numpy()
    target_adj_remove_self = target_adj - np.eye(target_adj.shape[0])
    query_adj_remove_self = query_adj - np.eye(query_adj.shape[0])
    target_graph = nx.from_numpy_array(target_adj_remove_self)
    query_graph = nx.from_numpy_array(query_adj_remove_self)
    query_graph_remove_dumb_nodes = remove_dumb_nodes(query_graph)

    list_predict_result = predict_result[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    # extract predict subgraph
    predict_subgraph = target_graph.subgraph(predict_nodes)

    # query diameter
    query_diameter = nx.diameter(query_graph_remove_dumb_nodes)

    # predict diameter
    if nx.number_of_nodes(predict_subgraph) != 0:
        max_diameter = -1
        for connected_nodes in nx.connected_components(predict_subgraph):
            sub_subgraph = predict_subgraph.subgraph(list(connected_nodes))
            subgraph_diameter = nx.diameter(sub_subgraph)
            if subgraph_diameter > max_diameter:
                max_diameter = subgraph_diameter
        predict_diameter = max_diameter
    else:
        predict_diameter = 0
    print('query_diameter', query_diameter, 'predict_diameter', predict_diameter)
    com_diameter = (2 * predict_diameter * query_diameter) / (pow(predict_diameter, 2) + pow(query_diameter, 2))
    return query_diameter, predict_diameter


def cluster_coefficient(query_adj, target_adj, predict_result):
    # get query and target
    target_adj = target_adj.cpu().numpy()
    query_adj = query_adj.cpu().numpy()
    target_adj_remove_self = target_adj - np.eye(target_adj.shape[0])
    query_adj_remove_self = query_adj - np.eye(query_adj.shape[0])
    target_graph = nx.from_numpy_array(target_adj_remove_self)
    query_graph = nx.from_numpy_array(query_adj_remove_self)
    query_graph_remove_dumb_nodes = remove_dumb_nodes(query_graph)

    list_predict_result = predict_result[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)
    # extract predict subgraph
    predict_subgraph = target_graph.subgraph(predict_nodes)

    # query cluster coefficient
    query_cluster_coefficient = nx.average_clustering(query_graph_remove_dumb_nodes)

    # predict cluster coefficient
    if nx.number_of_nodes(predict_subgraph) == 0:
        predict_cluster_coefficient = 0
    else:
        predict_cluster_coefficient = nx.average_clustering(predict_subgraph)
    print('query_cluster_coefficient', query_cluster_coefficient, 'predict_cluster_coefficient',
          predict_cluster_coefficient)
    com_cluster_coefficient = (2 * predict_cluster_coefficient * query_cluster_coefficient) / (
                pow(predict_cluster_coefficient, 2) + pow(query_cluster_coefficient, 2))
    return query_cluster_coefficient, predict_cluster_coefficient


def f1_score(predict_result, label):
    list_predict_result = predict_result[0].tolist()
    list_label_result = label[0].tolist()
    predict_nodes = []
    for i in range(len(list_predict_result)):
        if list_predict_result[i] > 0.5:
            predict_nodes.append(i)

    label_nodes = []
    for i in range(len(list_label_result)):
        if list_label_result[i] == 1:
            label_nodes.append(i)

    covered_nodes = []
    for node in predict_nodes:
        if node in label_nodes:
            covered_nodes.append(node)
    precision = len(covered_nodes) / (len(predict_nodes) + 0.001)
    recall = len(covered_nodes) / (len(label_nodes) + 0.001)
    f1_socre = (2 * precision * recall) / (precision + recall + 0.001)
    return f1_socre