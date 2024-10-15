import torch
import numpy as np
import networkx as nx
# import pandas as pd
import time
import dgl

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

def faster_candidate_generate_depend_on_core(target_graph, query_graph):
    
    query_degree = query_graph.in_degrees().tolist()
    query_graph = nx.Graph(dgl.to_networkx(query_graph.cpu()))
    # remove self loop
    # target_node_number = target_adj.shape[0]
    # query_node_number = query_adj.shape[0]
    # target_adj = target_adj - torch.eye(target_node_number).to('cuda:1')
    # query_adj = query_adj - torch.eye(query_node_number).to('cuda:1')
    #target_adj = target_adj - torch.eye(target_node_number)
    #query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    # np_query_adj = query_adj.cpu().numpy()
    # query_graph = nx.from_numpy_array(np_query_adj)
    # # print(type(query_graph.number_of_nodes())) # type-int
    # number_of_query_nodes = query_graph.number_of_nodes()
    # np_target_adj = target_adj.cpu().numpy()
    # target_graph = nx.from_numpy_array(np_target_adj)
    # get target coreness
    # target_coreness = nx.core_number(target_graph)
    number_of_query_nodes = nx.number_of_nodes(query_graph)

    # get min coreness
    # query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
    remove = [node for node,degree in dict(query_graph.degree()).items() if degree == 0]
    query_graph.remove_nodes_from(remove)
    query_graph_coreness = nx.core_number(query_graph)
    # print(type(query_graph_coreness))
    min_coreness_key = min(query_graph_coreness, key=lambda k: query_graph_coreness[k])
    min_coreness_value = query_graph_coreness[min_coreness_key]
    # print(min_coreness_value)

    # get global candidate
    # target_degree = torch.sum(target_adj, dim=1)
    target_degree = target_graph.in_degrees().tolist()
    # query_degree = torch.sum(query_adj, dim=1)
    
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
    # print('len(candidate_range)',len(candidate_range))

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

def candidate_generate_depend_on_core_for_large_graph(target_graph, query_graph):
    start = time.time()
    # query_graph = query_graph.cpu().to_networkx(query_graph)
    query_graph = nx.Graph(dgl.to_networkx(query_graph.cpu()))
    # remove self loop
    # target_node_number = target_graph.num_nodes()
    # query_node_number = query_graph.num_nodes()
    #target_adj = target_adj - torch.eye(target_node_number).to('cuda:1')
    #query_adj = query_adj - torch.eye(query_node_number).to('cuda:1')
    # target_adj = target_adj - torch.eye(target_node_number)
    # query_adj = query_adj - torch.eye(query_node_number)

    # transfrom to graph
    # np_query_adj = query_adj.cpu().numpy()
    # query_graph = nx.from_numpy_array(np_query_adj)
    # print(type(query_graph.number_of_nodes())) # type-int
    number_of_query_nodes = nx.number_of_nodes(query_graph)

    # get min coreness
    # query_graph.remove_edges_from(nx.selfloop_edges(query_graph))
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
    # target_degree = torch.sum(target_adj, dim=1)
    target_degree = target_graph.in_degrees().tolist()
    # query_degree = torch.sum(query_adj,dim=1)
    query_coreness = list(query_graph_coreness.values())
    # print(len(query_coreness))
    all_candidate = {}
    for i in range(len(target_degree)):
        if target_degree[i] > min_coreness_value:
            all_candidate[i] = int(target_degree[i])
    # print(len(list(all_candidate.keys())))
    # for reddit
    # based_candidate = [49161, 49162, 106506, 49170, 57362, 24596, 114713, 131099, 147496, 163880, 221225, 229421, 41007, 114739, 52, 73784, 204858, 24643, 57412, 8261, 16457, 98386, 204884, 221272, 57435, 92, 172125, 147552, 97, 180322, 65640, 131192, 163964, 221309, 196734, 163967, 106627, 188549, 180367, 49307, 41119, 49312, 49317, 82085, 221351, 204968, 164012, 123053, 229549, 155824, 172208, 221362, 204980, 182, 131255, 57533, 32958, 229567, 164036, 24776, 188619, 90325, 131292, 147676, 65760, 164065, 205025, 229604, 41194, 221421, 155891, 180470, 33016, 180473, 172285, 24832, 257, 82180, 205062, 114952, 123146, 164109, 172302, 180495, 41235, 41240, 41244, 213277, 155934, 16673, 188706, 123173, 65834, 82220, 147762, 33079, 180536, 229689, 41274, 82248, 123209, 65867, 123213, 41295, 16720, 57685, 352, 229731, 164205, 139634, 221556, 180598, 188794, 213376, 98691, 188810, 172428, 147860, 106902, 123288, 8601, 164249, 33180, 74143, 156066, 74150, 156070, 205222, 213418, 188848, 188850, 8629, 197049, 123322, 33211, 131514, 197053, 16830, 82369, 74181, 454, 164304, 8660, 221653, 139741, 74208, 25066, 213482, 164334, 57839, 156144, 205296, 131570, 213497, 507, 74236, 107007, 221706, 8719, 131607, 538, 107034, 188959, 205356, 139826, 98871, 229949, 205375, 172612, 148037, 156229, 221769, 172643, 172644, 8805, 107110, 25191, 172647, 172650, 197230, 230003, 164469, 148086, 164473, 164477, 639, 131720, 90763, 205451, 205459, 25246, 205477, 221862, 156327, 33455, 82614, 107194, 66235, 148155, 230083, 180933, 230089, 49866, 123595, 49871, 205520, 17105, 221904, 123603, 123605, 90838, 197338, 189155, 164582, 66280, 221935, 33522, 230132, 99073, 771, 205583, 164636, 205603, 17194, 9004, 815, 131898, 66369, 115521, 148294, 205640, 33609, 172876, 90957, 90958, 41808, 50002, 58197, 131932, 222060, 99181, 230269, 9092, 123781, 17286, 82822, 131980, 181133, 140183, 41880, 148386, 91047, 140209, 148401, 50099, 172979, 222129, 132027, 58303, 181188, 148421, 205766, 33737, 140234, 189413, 99302, 82920, 189420, 123889, 115698, 66547, 123892, 181236, 17398, 140278, 91128, 50173, 164870, 66573, 115731, 148499, 99353, 214041, 66588, 115746, 148524, 33845, 123959, 99385, 33851, 58428, 156734, 173118, 107585, 42059, 17484, 42063, 25680, 205910, 50266, 173152, 1122, 230500, 197736, 83056, 140400, 189558, 91257, 214139, 50306, 230534, 33929, 156811, 222354, 189591, 66712, 9369, 140459, 189612, 107698, 156850, 148661, 25794, 189646, 140513, 58595, 58608, 66802, 124148, 50421, 99574, 58621, 214273, 25861, 99589, 173320, 107785, 222473, 17679, 189711, 17683, 83221, 50458, 66843, 25886, 42270, 91425, 99617, 132385, 17707, 156972, 124205, 173355, 58671, 1331, 1332, 148791, 99653, 42311, 34128, 214352, 9562, 50527, 17760, 197985, 206181, 83307, 214387, 17780, 173429, 99707, 230779, 132480, 230784, 132484, 1418, 99724, 26001, 198035, 165280, 75170, 222628, 91557, 75174, 107944, 58803, 107955, 132538, 173500, 116158, 198081, 26068, 230872, 165340, 9693, 67038, 99817, 181740, 75250, 222706, 214516, 132597, 116215, 173559, 67065, 124417, 50699, 189974, 108060, 198176, 17962, 108075, 50733, 124462, 181810, 75319, 140861, 91712, 149063, 124488, 34379, 206413, 132687, 83553, 9826, 214638, 91761, 1653, 198268, 157310, 116357, 181895, 198283, 1685, 42649, 181915, 108192, 34465, 83618, 9891, 157350, 181927, 231080, 42666, 18093, 140979, 42677, 34486, 157372, 91837, 124610, 75465, 149196, 165581, 18142, 173796, 198388, 67317, 67325, 206590, 34565, 173830, 116487, 149256, 214791, 231178, 50956, 206607, 173842, 50963, 100115, 141075, 206610, 222999, 173870, 1842, 190272, 1858, 34635, 223053, 108366, 173903, 231247, 231252, 42838, 206680, 182111, 173930, 190315, 92012, 108396, 173938, 26488, 42873, 75646, 26498, 59274, 133007, 92054, 51111, 157608, 231337, 75695, 100274, 83897, 83903, 116677, 83912, 165849, 223194, 51166, 124899, 198632, 182252, 116726, 165884, 215037, 165886, 223231, 149507, 2055, 100364, 59413, 67608, 231448, 116763, 84002, 43043, 206887, 215082, 157744, 10293, 215094, 34876, 75837, 84032, 34888, 125000, 182347, 59470, 51287, 43106, 10341, 165990, 10351, 10353, 206961, 149619, 198771, 34934, 198775, 231546, 141436, 231548, 108672, 92289, 116868, 116869, 67721, 43151, 10397, 141478, 75943, 133287, 198829, 190642, 190649, 75964, 231613, 166082, 133320, 190664, 108750, 18639, 59599, 182478, 26836, 35029, 215262, 231646, 10482, 215298, 2307, 133382, 26887, 2323, 133396, 100634, 117020, 10525, 35103, 92447, 166177, 100644, 108836, 51495, 190761, 182574, 141616, 182577, 231731, 43316, 84282, 207168, 18753, 174401, 223553, 174409, 182607, 117072, 174417, 174426, 231774, 215394, 2403, 26984, 117097, 125288, 125291, 182632, 67951, 108916, 133493, 149878, 223606, 18813, 199050, 35212, 59789, 141710, 158096, 2455, 158107, 108958, 92575, 18857, 117164, 76209, 158131, 76214, 2491, 174523, 10687, 10700, 125388, 149973, 92631, 35290, 109020, 76254, 59872, 149986, 10723, 141796, 68074, 158192, 68083, 150007, 166392, 59904, 68100, 231945, 150026, 174604, 35343, 68113, 117273, 231980, 92718, 141871, 100925, 191037, 158272, 92747, 76370, 166482, 158296, 133721, 133722, 43621, 207461, 100970, 174699, 84592, 150134, 158327, 51832, 191111, 60047, 232092, 92830, 182942, 232096, 2721, 51874, 51876, 133796, 92842, 174769, 109241, 2748, 150209, 27338, 232138, 150225, 101077, 68310, 10979, 109288, 101098, 19187, 133885, 117528, 224027, 191263, 224034, 207652, 76581, 166694, 125736, 224043, 11052, 2873, 76605, 2878, 150333, 224064, 68417, 183106, 92996, 199495, 207691, 43866, 207706, 183133, 166753, 93029, 109414, 142187, 35697, 232306, 27509, 93045, 191353, 19322, 52101, 11150, 142226, 207765, 68502, 158617, 93085, 76702, 125853, 224157, 232354, 2987, 166834, 84915, 52150, 158651, 166850, 134083, 52169, 68557, 27600, 68563, 207831, 125919, 134117, 52197, 11238, 35820, 134127, 19443, 134132, 224256, 101377, 216071, 11273, 191499, 60434, 158741, 60440, 224281, 158755, 3111, 158769, 11314, 216113, 134215, 142410, 60496, 224338, 60503, 216153, 44123, 175197, 27743, 35939, 68710, 158823, 52340, 19587, 224388, 126087, 208008, 167053, 208023, 134298, 167070, 224428, 175277, 68791, 76985, 183482, 117958, 175307, 150733, 109779, 224472, 93404, 142556, 199906, 60645, 44265, 216301, 11513, 68863, 167168, 232704, 158984, 175374, 216357, 118055, 142637, 85295, 68916, 93492, 68919, 216375, 68922, 3390, 52544, 191808, 3397, 19786, 175438, 191828, 77142, 19810, 224612, 101745, 101746, 134516, 142709, 44409, 77182, 142718, 19840, 208254, 44419, 36230, 60806, 118153, 109963, 101773, 216464, 69009, 232868, 85415, 167339, 60849, 19892, 11707, 118205, 183742, 142790, 232907, 36310, 52704, 232931, 167396, 183780, 60907, 183794, 36343, 142839, 36349, 93695, 208384, 60932, 93702, 175624, 110089, 60943, 85520, 216592, 192019, 69142, 175640, 183844, 167461, 11820, 208430, 60995, 200264, 134729, 224854, 20069, 77415, 11884, 20084, 3721, 28302, 102031, 36519, 102078, 151231, 143042, 20171, 3795, 159444, 208604, 143072, 3814, 20204, 36590, 69371, 102142, 159491, 102155, 53008, 167698, 53014, 200473, 44830, 208672, 3875, 208676, 143149, 3887, 208690, 184119, 3904, 216898, 85827, 53061, 126792, 85852, 20322, 44906, 94065, 110452, 192381, 126852, 3975, 225159, 44938, 118666, 135051, 225163, 20379, 126878, 4016, 69552, 176060, 110528, 143300, 184262, 53193, 53196, 184287, 20453, 20454, 159719, 53225, 36844, 61430, 167927, 176123, 77824, 4107, 94220, 126989, 110616, 167960, 20510, 28708, 12337, 53297, 176180, 118840, 110653, 192575, 53316, 143432, 127049, 77901, 61520, 28767, 102497, 184428, 12397, 20589, 12401, 94326, 192630, 217206, 209018, 45180, 200834, 77956, 53386, 110743, 135319, 143514, 135329, 151717, 77990, 37033, 151735, 159929, 86202, 102586, 78018, 192707, 69830, 102607, 78036, 61663, 127203, 151780, 45286, 110822, 4330, 102635, 209138, 209140, 217333, 200957, 20738, 53507, 69891, 86278, 61712, 78096, 53522, 168209, 209168, 225581, 61744, 127280, 53554, 176438, 4415, 127305, 61770, 94538, 201042, 86359, 29016, 184663, 192857, 176479, 110945, 86370, 69995, 12666, 201082, 45440, 119174, 94600, 176523, 184715, 184718, 151952, 37276, 209308, 12704, 192934, 86439, 61870, 135599, 4528, 168368, 78260, 78262, 45504, 4545, 45506, 61889, 29125, 70093, 209361, 217553, 176599, 4571, 119269, 29158, 152043, 12786, 143859, 45556, 135668, 111097, 135673, 102912, 37381, 12816, 184849, 102937, 176666, 78364, 135708, 201244, 94751, 201246, 152099, 21032, 45612, 193068, 102963, 94772, 102965, 12856, 86592, 78407, 152141, 209491, 152148, 70229, 143957, 78430, 29279, 103012, 45700, 53901, 201369, 78494, 4781, 168628, 135865, 144063, 53954, 127693, 13007, 135894, 70368, 78561, 176865, 21220, 103140, 127719, 160490, 29426, 70386, 176887, 94976, 45832, 21257, 86798, 4882, 168728, 209699, 226097, 37683, 217907, 54077, 209735, 209736, 127818, 185171, 62297, 78692, 127858, 226168, 160635, 78724, 226181, 111501, 144270, 193428, 103321, 119707, 45982, 136096, 127914, 193452, 177070, 111535, 144305, 185265, 29620, 119733, 193462, 29623, 136127, 201666, 111559, 136137, 136146, 37848, 201693, 218081, 136162, 78820, 54245, 119784, 37866, 119786, 29677, 13294, 144373, 177145, 70658, 70660, 87047, 21517, 168985, 78874, 119834, 128030, 95272, 5174, 29752, 87104, 152646, 209999, 160849, 5209, 152669, 177245, 201830, 54376, 226409, 128107, 177261, 160896, 54401, 177288, 218250, 29835, 103564, 70800, 119952, 185494, 62621, 201892, 38056, 95405, 144557, 62647, 79047, 70856, 5322, 177357, 144592, 136407, 218345, 177388, 54512, 46327, 13565, 177412, 218378, 13587, 38175, 202015, 177445, 95527, 38189, 13613, 62771, 79156, 13622, 210230, 62782, 210240, 185666, 136517, 54603, 169294, 152912, 177490, 144723, 62807, 21854, 46431, 218465, 210278, 202092, 62839, 226679, 5497, 103802, 185724, 95619, 54660, 169349, 185748, 161177, 120218, 210331, 46492, 13730, 13731, 79267, 71080, 218550, 38331, 38332, 87485, 153025, 177614, 112084, 218581, 62934, 21985, 161256, 169450, 226797, 169460, 202230, 30203, 120315, 13824, 136708, 128533, 71195, 153125, 112172, 144946, 103988, 103991, 128573, 144959, 226885, 22086, 79439, 185939, 46687, 194143, 79464, 63083, 128630, 120440, 169593, 177787, 202364, 177796, 161416, 104073, 169624, 194202, 145052, 153251, 79524, 13992, 153257, 161460, 95932, 87755, 38612, 38613, 227038, 63204, 218872, 55035, 112383, 218879, 218887, 145181, 202531, 46885, 79654, 128806, 137000, 169768, 169770, 63282, 96053, 46903, 202555, 55100, 79680, 177989, 5959, 87895, 186209, 79715, 178020, 218980, 120680, 63337, 112499, 227187, 161654, 55159, 55161, 6012, 153468, 79751, 112519, 128912, 227216, 87960, 38814, 153506, 161699, 178083, 219043, 153515, 120751, 30647, 6072, 202686, 63431, 219082, 14284, 38861, 186320, 169942, 120792, 120793, 169950, 71651, 47079, 227304, 120811, 63471, 112627, 47096, 47099, 88063, 79871, 88065, 219143, 96268, 104463, 145424, 47131, 186396, 88096, 79905, 137256, 210985, 88106, 14385, 71739, 22588, 211009, 129096, 55371, 71760, 202839, 145497, 219227, 227430, 137320, 39023, 145521, 137331, 112761, 137342, 170112, 202880, 153730, 88195, 14470, 178312, 22665, 194701, 71831, 14493, 71838, 6303, 96414, 112797, 145571, 178342, 194727, 14504, 112808, 22700, 219319, 104645, 211144, 104654, 104655, 104660, 162006, 22747, 39141, 63717, 162027, 137454, 186618, 121091, 71945, 178444, 6414, 137486, 63762, 203026, 6423, 88349, 153886, 178472, 203048, 39210, 47404, 22829, 227628, 55600, 88368, 96569, 112954, 186682, 63806, 121151, 14659, 129350, 153935, 219473, 104786, 55635, 63831, 112985, 153945, 39262, 63844, 112997, 178544, 162170, 14723, 96644, 63884, 80277, 31126, 203160, 170409, 104880, 203189, 154040, 137662, 63936, 39368, 219605, 55768, 113112, 96730, 55774, 39392, 154085, 211431, 154089, 203242, 129518, 195059, 203255, 88572, 113159, 64008, 186893, 39440, 72210, 129559, 31262, 162336, 170542, 47664, 211504, 227904, 88643, 121417, 186954, 211533, 137809, 178775, 129632, 31337, 14954, 203369, 96879, 219765, 88694, 72316, 113276, 72323, 195212, 80526, 6802, 137874, 178837, 178844, 96925, 39584, 113312, 113313, 96933, 170665, 137916, 113345, 64194, 203460, 88777, 146125, 187089, 39635, 72404, 56028, 88798, 31460, 64229, 96996, 23271, 187112, 162537, 105194, 121578, 121590, 195319, 170744, 121593, 219897, 39675, 105214, 64261, 23313, 72468, 15126, 47894, 211734, 219935, 47905, 228134, 15146, 97068, 72503, 15162, 154430, 146250, 39761, 211798, 23391, 80735, 121695, 154464, 31587, 15208, 64380, 72579, 211848, 228240, 48019, 154518, 64408, 154534, 48045, 113587, 64452, 187334, 48074, 7116, 162768, 15317, 220117, 15324, 130016, 162788, 23526, 195569, 170996, 228341, 97271, 39932, 130047, 56320, 80896, 105477, 211974, 228357, 195592, 80906, 179211, 105484, 211980, 203792, 121875, 138264, 15392, 220193, 105509, 154665, 203825, 228401, 162871, 146490, 105543, 56394, 72778, 80970, 7245, 171084, 187468, 195660, 64596, 171094, 40028, 97375, 195687, 23667, 130163, 64629, 212094, 31872, 171139, 187526, 89224, 220308, 48281, 81052, 7337, 130220, 97453, 154807, 171191, 64697, 56508, 56513, 122050, 187589, 81099, 56525, 97485, 97489, 7378, 146644, 195799, 212190, 179425, 64739, 187626, 228611, 187652, 105739, 56589, 23830, 212248, 171294, 146750, 122175, 228676, 138575, 212306, 23897, 89440, 163175, 32104, 212328, 56684, 7535, 171386, 32127, 89490, 155026, 171412, 40343, 7576, 15770, 130460, 81310, 81314, 64934, 56752, 122290, 122294, 138681, 187837, 228804, 105926, 122311, 64975, 187856, 146900, 15835, 114140, 64993, 64999, 171500, 228847, 228851, 212468, 65023, 179711, 48643, 105988, 97797, 65037, 122381, 204305, 24084, 24086, 179738, 40479, 73248, 114215, 81452, 155180, 212526, 171568, 204337, 220721, 220724, 15928, 204351, 228934, 48712, 138831, 171600, 32340, 15957, 97883, 114267, 40542, 56930, 97890, 122469, 48743, 56938, 212591, 73329, 114291, 40564, 24182, 40569, 147068, 56957, 65155, 97928, 65169, 7829, 97948, 130718, 114338, 32420, 48806, 212655, 229043, 122563, 204483, 57029, 130760, 220882, 24280, 106212, 73448, 81642, 138988, 7923, 106230, 32507, 48891, 122626, 7946, 16143, 106263, 179991, 139033, 212762, 220951, 16156, 188191, 212767, 139041, 89898, 73517, 147247, 48957, 48958, 106303, 89925, 8007, 8008, 180039, 204616, 221003, 106317, 221014, 229207, 212825, 204634, 130909, 106336, 57189, 130918, 147312, 49016, 204669, 130942, 204671, 65408, 106371, 98180, 204675, 229255, 32649, 171916, 196500, 8090, 90017, 24485, 171943, 122792, 65461, 204733, 106434, 155604, 221142, 221153, 221155, 16357, 65510, 221161, 40938, 40942, 212975, 98289, 196595, 24568, 147449, 114682]
    # for i in range(len(based_candidate)):
    #     all_candidate[based_candidate[i]] = int(target_degree[based_candidate[i]])
        
    i = list(range(len(all_candidate)))
    candidate_mapping = dict(list(zip(list(all_candidate.keys()), i)))
    # print(candidate_mapping) #397:130
    candidate_adj = torch.ones(number_of_query_nodes, len(all_candidate))

    end = time.time()
    candidate_range = list(all_candidate.keys())
    # print(len(candidate_range))

    return candidate_range, candidate_adj

# candidate_generate_depend_on_core_for_large_graph(target_adj,query_adj)
