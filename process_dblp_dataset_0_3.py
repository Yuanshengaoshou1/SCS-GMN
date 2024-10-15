import networkx as nx
import time
import tqdm
import os
import math
from sample_dataset import *
import dgl


def community_similarity(ori_graph,query_graph):
    # 取出查询图和目标图


    # 2nd measure - edges
    target_edges = nx.number_of_edges(ori_graph)
    query_edges = nx.number_of_edges(query_graph)
    # normalize
    # scaled_query_edges,scaled_predict_edges = normalize_for_commnunity_similarity(query_edges,predict_edges)
    # query_edges = math.log(scaled_query_edges,10)
    # predict_edges = math.log(scaled_predict_edges,10)

    # 3rd measure - coreness
    target_coreness = nx.core_number(ori_graph)
    list_target_coreness = list(target_coreness.values())
    target_avg_coreness = sum(list_target_coreness) / len(list_target_coreness)

    query_coreness = nx.core_number(query_graph)
    list_query_coreness = list(query_coreness.values())
    query_avg_coreness = sum(list_query_coreness) / len(list_query_coreness)
    # normalize
    # scaled_query_avg_coreness,scaled_predict_avg_coreness = normalize_for_commnunity_similarity(query_avg_coreness,predict_avg_coreness)
    # query_avg_coreness = math.log(scaled_query_avg_coreness,10)
    # predict_avg_coreness = math.log(scaled_predict_avg_coreness,10)

    # 4th measure - nodes
    target_nodes = nx.number_of_nodes(ori_graph)
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
    # print('query_avg_coreness',query_avg_coreness,'target_avg_coreness',target_avg_coreness)
    # print('query_density',query_density,'target_density',target_density)
    # print('query_nodes',query_nodes,'target_nodes',target_nodes)
    # print('first_measure',first_measure,'second_measure',second_measure,'third_measure',third_measure)
    # return first_measure * second_measure * third_measure * fourth_measure,first_measure,second_measure,third_measure,fourth_measure
    return first_measure * second_measure * third_measure

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

def construct_init_features(graph,type):
    print('construct init features')
    if type == 'degree':
        dict_degree = dict(nx.degree(graph))
        graph_degree = list(dict_degree.values())
        print('graph_degree',len(graph_degree))
        max_degree = max(graph_degree)
        print('max degree',max_degree)
        target_features = torch.zeros(nx.number_of_nodes(graph),(max_degree+1),dtype=torch.float32)
        for i in range(len(dict_degree)):
            target_features[i][dict_degree[i]] = 1
        return target_features
    elif type == 'h_index':
        list_target_h_indexs = [get_h_index(graph, n) for n in graph.nodes()]
        print('list_target_h_indexs',len(list_target_h_indexs))
        max_h_indexs = max(list_target_h_indexs)
        print('max_h_indexs',max_h_indexs)
        target_features = torch.zeros(nx.number_of_nodes(graph),(max_h_indexs+1),dtype=torch.float32)
        for n in graph.nodes():
            neighbors = list(graph[n])
            for i in range(len(neighbors)):
                target_features[n][list_target_h_indexs[neighbors[i]]] = target_features[n][list_target_h_indexs[neighbors[i]]] + 1
            target_features[n,:] = target_features[n,:]/len(neighbors)
        return target_features
def LoadLarges(data_path,dataName):
    # print('Loading {}...'.format(self.dataName))
    Graph = nx.Graph()
    name_id = {}
    id_name = {}
    NodeNum = 0
    begin = time.time()

    # read nodes
    nodeSet = set()
    with open(os.path.join(data_path, 'com-{}.ungraph.txt'.format(dataName)), 'r') as f:
        line = f.readline()
        while line:
            edge = line.split()
            if '#' in edge or len(edge) > 2:
                # print('skip format')
                line = f.readline()
                continue
            nodeSet.add(edge[0])
            nodeSet.add(edge[1])
            line = f.readline()
        f.close()
    for index, name in enumerate(nodeSet):
        id_name[index] = name
        name_id[name] = index
    del nodeSet
        
    # read edges
    with open(os.path.join(data_path, 'com-{}.ungraph.txt'.format(dataName)), 'r') as f:
        line = f.readline()
        while line:
            edge = line.split()
            if '#' in edge or len(edge) > 2:
                # print('skip format')
                line = f.readline()
                continue
            Graph.add_edge(name_id.get(edge[0]), name_id.get(edge[1]))
            line = f.readline()
        f.close()
    NodeNum = len(Graph.nodes())
    EdgeNum = len(Graph.edges())
    print('Node Num:',NodeNum)
    print('Edge Num:',EdgeNum)
    # print('Using:{}s'.format(time.time() - begin))
    return Graph,name_id

def LoadCommunity_graph(data_path,dataName,name_id,all=False):
    # all measueres whether load all communities
    if all:
        all = 'all'
    else:
        all = 'top5000'
    communities = []

    # 获取完整进度
    # with open(os.path.join(data_path, 'com-{}.{}.cmty.txt'.format(dataName, all)), 'r') as file:
    #     num_lines = sum(1 for line in file)
    #     file.close()
    # file_tqdm = tqdm(total=num_lines, desc='Loading {}.Community'.format(dataName), ncols=100)

    # read community
    with open(os.path.join(data_path, 'com-{}.{}.cmty.txt'.format(dataName, all)), 'r') as f:
        line = f.readline()
        while line:
            communities.append(line.split())
            # file_tqdm.update(1)
            # time.sleep(1)
            line = f.readline()
        f.close()
    # print('Community Num:', len(communities))
    communities = sorted(communities, key=len, reverse=True)
    # print('Biggest Community Scale', len(communities[0]))
    # print('Smallest Community Scale', len(communities[-1]))
    label_ids = {}
    name_label = {}
    label_set = set()
    for index, line in enumerate(communities):
        label_set.add(index) # get community number
        label_ids[index] = [name_id.get(node) for node in line] # get node number
        for node in line: # record node in which community
            if node in name_label:
                name_label[node].add(index)
            else:
                name_label[node] = set([index])
    GTNum = len(list(label_set))
    return communities,name_label

def load_dblp_data(target_graph,train_size,test_size):
    
    # get k distribution
    # note nodes in the same k-core doesn't mean in the same label(community)
    k_distribution, nor_k_distribution = get_k_core_distribution(target_graph)

    all_sampled_k = sample_k_from_distribution(nor_k_distribution, train_size)
    print('sampled_k ', all_sampled_k)
    all_connected_component_nodes, max_node_number = sample_core_to_query(all_sampled_k, k_distribution, target_graph)
    print('max_node_number ', max_node_number)
    target_features = construct_init_features(target_graph,'degree')
    # for train
    all_labels = []
    all_query_graph = []
    all_target_graph = []
    for i in range(train_size):
        print('sample number', i)
        sampled_k = all_sampled_k[i]
        connected_component_nodes = all_connected_component_nodes[i]
        print('sampled_k', sampled_k)
        print('connected_component_nodes', connected_component_nodes)
        while (1):
            nodes_number = len(connected_component_nodes)
            # print('before sampling core_number',min(nx.core_number(target_graph.subgraph(connected_component_nodes)).values()))
            sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.6))
            subgraph = target_graph.subgraph(sample_nodes)
            min_coreness = min(nx.core_number(subgraph).values())
            # print('after sampling core_number',min(nx.core_number(subgraph).values()))
            subsubgraph = nx.k_core(subgraph, k=int(min_coreness * 0.6))
            # print('done remove nodes')
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # remove edges
            for k in range(35):
                g_query_edge = random.choice(list(subsubgraph.edges()))
                subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
            # print('after remove edges core_number',min(nx.core_number(subsubgraph).values()))
            subsubgraph = nx.k_core(subsubgraph, k=int(min_coreness * 0.55))
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # public
            # if nx.is_connected(subsubgraph) and flag:
            if nx.is_connected(subsubgraph) :
                selected_nodes = list(nx.nodes(subsubgraph))
                print(selected_nodes)
                query_features = target_features[selected_nodes]
                # print('nx.degree(subsubgraph)',list(dict(nx.degree(subsubgraph)).values()))
                # print('nx.degree(subsubgraph)',nx.degree(subsubgraph))
                # list_degree = nx.degree(target_graph)
                # target_degree = [list_degree[selected_nodes[i]] for i in range(len(selected_nodes))]
                # print('nx.degree(target_graph)',target_degree)
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                # supplement dumb nodes
                # if nx.number_of_nodes(query_graph) < max_node_number:
                #     for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                #         # new query graph node
                #         query_graph.add_node(add_node)
                #         # new query node features
                #         dumb_node_features = np.zeros((1, target_features.shape[1]))
                #         query_features = np.row_stack((query_features, dumb_node_features))
                ori_graph = target_graph.subgraph(connected_component_nodes)
                similarity_value = community_similarity(ori_graph, query_graph)
                print('similarity_value :', similarity_value)
                if 0.7 <= similarity_value <= 0.8:
                    print('selected_nodes', selected_nodes)
                    print('get!')
                    break
        # label
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

    graph_labels = {'glabel': torch.tensor(all_labels)}
    dgl.save_graphs('/mnt/HDD/crh/dblp/for_train_dblp/0.1/for_train_dblp_target_0_3.bin',all_target_graph)
    dgl.save_graphs('/mnt/HDD/crh/dblp/for_train_dblp/0.1/for_train_dblp_query_0_3.bin',all_query_graph,graph_labels)
    print('save done')
    
    # for test
    all_labels = []
    all_query_graph = []
    all_target_graph = []
    
    # sample test data
    component_number = len(all_connected_component_nodes)
    sampled_number = random.sample(range(component_number), test_size)
    print('sampled_number', sampled_number)
    test_connected_component_nodes = []
    test_sampled_k = []
    for one in sampled_number:
        test_connected_component_nodes.append(all_connected_component_nodes[one])
        test_sampled_k.append(all_sampled_k[one])
    
    
    for i in range(test_size):
        print('sample number', i)
        sampled_k = test_sampled_k[i]
        connected_component_nodes = test_connected_component_nodes[i]
        print('sampled_k', sampled_k)
        print('connected_component_nodes', connected_component_nodes)
        while (1):
            nodes_number = len(connected_component_nodes)
            # print('before sampling core_number',min(nx.core_number(target_graph.subgraph(connected_component_nodes)).values()))
            sample_nodes = random.sample(connected_component_nodes, int(nodes_number * 0.6))
            subgraph = target_graph.subgraph(sample_nodes)
            min_coreness = min(nx.core_number(subgraph).values())
            # print('after sampling core_number',min(nx.core_number(subgraph).values()))
            subsubgraph = nx.k_core(subgraph, k=int(min_coreness * 0.6))
            # print('done remove nodes')
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # remove edges
            for k in range(35):
                g_query_edge = random.choice(list(subsubgraph.edges()))
                subsubgraph.remove_edge(g_query_edge[0], g_query_edge[1])
            # print('after remove edges core_number',min(nx.core_number(subsubgraph).values()))
            subsubgraph = nx.k_core(subsubgraph, k=int(min_coreness * 0.55))
            if nx.number_of_nodes(subsubgraph) == 0:
                continue
            # public
            # if nx.is_connected(subsubgraph) and flag:
            if nx.is_connected(subsubgraph) :
                selected_nodes = list(nx.nodes(subsubgraph))
                print(selected_nodes)
                query_features = target_features[selected_nodes]
                # print('nx.degree(subsubgraph)',list(dict(nx.degree(subsubgraph)).values()))
                # print('nx.degree(subsubgraph)',nx.degree(subsubgraph))
                # list_degree = nx.degree(target_graph)
                # target_degree = [list_degree[selected_nodes[i]] for i in range(len(selected_nodes))]
                # print('nx.degree(target_graph)',target_degree)
                query_graph = nx.convert_node_labels_to_integers(subsubgraph, first_label=0)
                # supplement dumb nodes
                # if nx.number_of_nodes(query_graph) < max_node_number:
                #     for add_node in range(nx.number_of_nodes(query_graph), max_node_number):
                #         # new query graph node
                #         query_graph.add_node(add_node)
                #         # new query node features
                #         dumb_node_features = np.zeros((1, target_features.shape[1]))
                #         query_features = np.row_stack((query_features, dumb_node_features))
                ori_graph = target_graph.subgraph(connected_component_nodes)
                similarity_value = community_similarity(ori_graph, query_graph)
                print('similarity_value :', similarity_value)
                if 0.7 <= similarity_value <= 0.8:
                    print('selected_nodes', selected_nodes)
                    print('get!')
                    break
        # label
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

    graph_labels = {'glabel': torch.tensor(all_labels)}
    dgl.save_graphs('/mnt/HDD/crh/dblp/for_test_dblp/0.1/for_test_dblp_target_0_3.bin',all_target_graph)
    dgl.save_graphs('/mnt/HDD/crh/dblp/for_test_dblp/0.1/for_test_dblp_query_0_3.bin',all_query_graph,graph_labels)
    print('save done')
    
        
def load_community_data(dataset,train_size,test_size):
    data_path = '/mnt/HDD/crh/dblp/'
    target_graph,name_id = LoadLarges(data_path,dataset)
    communities,name_label = LoadCommunity_graph(data_path,dataset,name_id,True)
    print('Communities Num',len(communities))
    target_graph.remove_edges_from(nx.selfloop_edges(target_graph))
    print('Edge Num:',nx.number_of_edges(target_graph))
    if dataset == 'dblp':
        load_dblp_data(target_graph,train_size,test_size)
        
load_community_data('dblp',60,20)