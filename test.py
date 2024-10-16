import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import dgl
import argparse
import datetime
from net import scs_GMN

from sklearn.metrics import f1_score

import time
from candidate_get import candidate_generate_depend_on_degree, candidate_generate_depend_on_core \
    , faster_candidate_generate_depend_on_core, candidate_generate_all_depend_on_core \
    , candidate_generate_depend_on_core_for_large_graph
from test_evaluation import community_similarity, f1_score, triangle_participation_ratio, diameter, \
    cluster_coefficient, label_cover_rate
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs


def remove_dumb_nodes(adj):
    # np_adj = adj.numpy()
    np_adj = adj.cpu().numpy()
    graph = nx.from_numpy_array(np_adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    remove = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    graph.remove_nodes_from(remove)
    removed_adj = nx.adjacency_matrix(graph).todense()
    removed_adj_add_self = removed_adj + np.eye(removed_adj.shape[0])
    return torch.tensor(removed_adj_add_self)

def record_single_query(save_path, dataset, community_similarity, com_density, com_coreness, com_nodes, query_density, predict_density,
                query_avg_coreness, predict_avg_coreness, query_nodes, predict_nodes, min_query_coreness, min_predict_coreness, train_type):
    file_path = save_path + 'single_record_query_' + train_type +'.txt'
    now_time = datetime.datetime.now()
    with open(file_path, 'a') as f:
        f.write('record time :' + str(now_time) + 's' + ' ' + 'dataset:' + str(dataset) + ' '
                + 'community_similarity :' + str(community_similarity) + ' ' + 'compared_density :' + str(
            com_density) + ' '
                + 'compared_coreness :' + str(com_coreness) + ' ' + 'compared_nodes :' + str(com_nodes) + ' '
                + 'query_density :' + str(query_density) + ' ' + 'predict_density :' + str(predict_density) + ' '
                + 'query_avg_coreness :' + str(query_avg_coreness) + ' ' + 'predict_avg_coreness :' + str(predict_avg_coreness) + ' '
                + 'query_nodes :' + str(query_nodes) + ' ' + 'predict_nodes :' + str(predict_nodes) + ' ' 
                + 'min_query_coreness :' + str(min_query_coreness) + ' ' + 'min_predict_coreness :' + str(min_predict_coreness) + ' ' + '\n')
        f.close()
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora') # cora/citeseer/pubmed/deezer/facebook
parser.add_argument('--cs_perturbation', type=float, default=0.1) # 0.1/0.2/0.3
parser.add_argument('--GCN_out_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test_size', type=int, default=1)
parser.add_argument('--candidate_method', type=str, default='for_large_graph',help='faster_coreness/for_large_graph')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--trade_off_for_re1', type=float, default=0.1,help='balance joint loss')
parser.add_argument('--trade_off_for_re2', type=float, default=0.01,help='balance joint loss')
parser.add_argument('--trade_off_for_re3', type=float, default=0.001,help='balance joint loss')
parser.add_argument('--train_type', type=str, default='both',help='both/only label/only structure')
parser.add_argument('--save_model', type=str, default=False,help='save model')
parser.add_argument('--model_path', type=str,help='the path of saved model')
args = parser.parse_args()

threshold = 1
# data
test_path = './dataset/' + args.dataset +'/for_test_' + args.dataset + '/' + str(args.cs_perturbation) + '/more_test_samples/'
# test_path = '/mnt/HDD/crh/dblp_data/for_test_dblp/model_test/'
test_target_path = test_path + 'for_test_' + args.dataset + '_target_8.bin'
test_query_path = test_path + 'for_test_' + args.dataset + '_query_8.bin'
#test_path = './dataset/' + args.dataset +'/for_test_' + args.dataset + '/for_case_citeseer/'
test_size = args.test_size
epoch = args.epoch
candidate_method = args.candidate_method  # degree\coreness\faster_coreness\rough_faster_coreness\for_large_graph
save_model = False
trade_off_for_re1 = args.trade_off_for_re1
trade_off_for_re2 = args.trade_off_for_re2
trade_off_for_re3 = args.trade_off_for_re3
device = torch.device('cuda:0')


# test target
test_target_graphs, _ = load_graphs(test_target_path)
test_target_features = test_target_graphs[0].ndata['feat']
test_target_features_to_tensor = torch.tensor(test_target_features,dtype=torch.float32)
test_target_features_to_tensor = test_target_features_to_tensor.to(device)
test_target_adj_to_tensor = test_target_graphs[0]
# test_target_adj_to_tensor = torch.tensor(test_target_adj[0])
test_target_adj_to_tensor = test_target_adj_to_tensor.to(device)

# test query
test_query_graphs,test_target_labels = load_graphs(test_query_path)
test_query_features = []
test_query_adj = []
test_labels = []
for i in range(len(test_query_graphs)):  
    test_query_features.append(torch.tensor(test_query_graphs[i].ndata['feat'],dtype=torch.float32))
    test_query_adj.append(test_query_graphs[i])
    test_labels.append(test_target_labels['glabel'][i].type(torch.float32))
test_data_size = len(test_query_graphs)

testdata = []

# prepare test data
for i in range(len(test_query_features)):
    one_testdata = []
    # one_testdata.append(test_target_features[i])
    # one_testdata.append(test_target_adj[i])
    one_testdata.append(test_query_features[i])
    one_testdata.append(test_query_adj[i])
    one_testdata.append(test_labels[i])
    testdata.append(one_testdata)
    in_size = test_query_features[i].shape[1]

data_test = DataLoader(testdata, batch_size=test_size, shuffle=True)

# setting
target_size = test_target_features_to_tensor.shape[0]
GCN_in_size = in_size  # features_dimension # cora 1433/Citeseer 3703/pubmed 500/deezer 4463/facebook 4714
GCN_out_size = args.GCN_out_size
da_size = target_size # target size Cora 2708/Citeseer 3312/pubmed 19717/deezer 28281/facebook 22470

# model
model = scs_GMN(GCN_in_size, GCN_out_size, da_size)
model.load_state_dict(torch.load(args.model_path))
model.eval()

model.to(device)

# setting
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Adam会自动调整学习率
loss_fun1 = torch.nn.MSELoss(reduction='mean')
loss_fun1 = loss_fun1.to(device)
l1_loss = torch.nn.L1Loss(reduction='mean')
l1_loss = l1_loss.to(device)

# about plot
# for community similarity
test_community_similarity = []

# time
all_for_train_time = 0
all_train_candidate_time = 0
all_test_time = 0
all_test_candidate_time = 0

# test
all_target_adj_for_test = []
all_query_adj_for_test = []
all_predict_res = []
for i in np.arange(epoch):
    print('epoch:  ', i)

    model.eval()  # test

    # plt
    # for community similarity
    mean_similarity_value = 0
    mean_com_density = 0
    mean_com_coreness = 0
    mean_com_nodes = 0
    mean_query_density = 0
    mean_predict_density = 0
    mean_query_avg_coreness = 0
    mean_predict_avg_coreness = 0
    mean_query_nodes = 0
    mean_predict_nodes = 0
    # others
    mean_query_tpr = 0
    mean_predict_tpr = 0
    mean_query_diameter = 0
    mean_predict_diameter = 0
    mean_query_cluster_coefficient = 0
    mean_predict_cluster_coefficient = 0
    count = 0

    for x, test_batch in enumerate(testdata):  # test data
        print('test stage')
        # test_target_features = torch.tensor(test_batch[0])
        # test_target_adjs = test_batch[1]
        test_query_features = torch.tensor(test_batch[0])
        test_query_features = test_query_features.to(device)
        test_query_adjs = test_batch[1]
        test_query_adjs = test_query_adjs.to(device)

        test_all_y_hat = torch.zeros(test_size, 1, da_size)

        # reconstruct loss
        # test_reconstruct_query_adj = torch.empty(test_size, q_size, q_size)
        # test_reconstruct_target_adj = torch.empty(test_size, da_size, da_size)

        batch_positive_samples_for_test = []
        batch_negative_samples_for_test = []
        batch_anchor_samples_for_test = []

        # reconstruct structural information
        test_reconstruct_degree = torch.zeros(test_size, 1)
        test_reconstruct_edges = torch.zeros(test_size, 1)
        test_reconstruct_nodes = torch.zeros(test_size, 1)
        test_reconstruct_degree_from_query = torch.zeros(test_size, 1)
        test_reconstruct_edges_from_query = torch.zeros(test_size, 1)
        test_reconstruct_nodes_from_query = torch.zeros(test_size, 1)

        test_batch_adj_loss = torch.zeros(1, 1)
        #test_batch_adj_loss = test_batch_adj_loss.to(device)
        for y in range(test_size):  # one batch
            # get candidate
            # print('test_labels_shape', test_labels[y].shape)
            test_start_candidate = time.time()
            if candidate_method == 'degree':
                test_candidate_set, test_candidate_adj = candidate_generate_depend_on_degree(
                    test_target_adj_to_tensor,
                    test_query_adjs[y])
            elif candidate_method == 'coreness':
                test_candidate_set, test_candidate_adj = candidate_generate_depend_on_core(
                    test_target_adj_to_tensor,
                    test_query_adjs[y])
            elif candidate_method == 'faster_coreness':
                test_candidate_set, test_candidate_adj = faster_candidate_generate_depend_on_core(
                    test_target_adj_to_tensor,
                    test_query_adjs)
            elif candidate_method == 'rough_faster_coreness':
                test_candidate_set, test_candidate_adj = candidate_generate_all_depend_on_core(
                    test_target_adj_to_tensor,
                    test_query_adjs[y])
            elif candidate_method == 'for_large_graph':
                test_candidate_set, test_candidate_adj = candidate_generate_depend_on_core_for_large_graph(
                    test_target_adj_to_tensor, test_query_adjs)
            # model test output
            test_end_candidate = time.time() - test_start_candidate
            all_test_candidate_time = all_test_candidate_time + test_end_candidate
            test_candidate_adj = test_candidate_adj.to(device)
            start_test = time.time()
            with torch.no_grad():

                test_y_hat, att_da2, att_q2, \
                test_avg_degree, test_density, test_avg_nodes\
                    = model(test_target_adj_to_tensor,
                            test_target_features_to_tensor,
                            test_query_adjs, test_query_features,
                            test_candidate_set, test_candidate_adj, threshold)
                end_test = time.time() - start_test
                all_test_time = all_test_time + end_test
                test_all_y_hat[y] = test_y_hat
                test_reconstruct_degree[y] = test_avg_degree
                test_reconstruct_edges[y] = test_density
                test_reconstruct_nodes[y] = test_avg_nodes
                #test_re_adj = test_re_adj.to(device)
                #test_ori_adj = test_ori_adj.to(device)

                # get reconstruct adj
                # test_reconstruct_target_adj[y] = re_target_adj
                # test_reconstruct_query_adj[y] = re_query_adj
                # reconstruct adj loss
                # test_re_adj_loss = loss_fun1(test_re_adj, test_ori_adj)
                # test_batch_adj_loss = test_batch_adj_loss + test_re_adj_loss

                # remove dumb nodes
                # remove_dumb_adj = remove_dumb_nodes(test_query_adjs[y])
                # if test_avg_degree == 0:
                #     avg_degree_from_query = 0
                # else:
                #     avg_degree_from_query = torch.mean(torch.sum(remove_dumb_adj, dim=1))
                # if test_density == 0:
                #     density_from_query = 0
                # else:
                #     density_from_query = 2 * torch.sum(remove_dumb_adj) / (
                #             torch.trace(remove_dumb_adj) * (torch.trace(remove_dumb_adj) - 1) + 0.0001)
                # if test_avg_nodes == 0:
                #     avg_nodes_from_query = 0
                # else:
                #     avg_nodes_from_query = torch.trace(remove_dumb_adj)
                # if i == 0 and x==0 and y==0:
                #     plt_adj_in_hot(test_query_adjs[y])
                # print('torch.mean(nor_test_query_adj',avg_degree_from_query)
                # print('torch.sum(nor_test_query_adj)',avg_edges_from_query)
                # test_reconstruct_degree_from_query[y] = float(avg_degree_from_query)
                # test_reconstruct_edges_from_query[y] = float(density_from_query)
                # test_reconstruct_nodes_from_query[y] = float(avg_nodes_from_query)

                all_query_adj_for_test.append(test_query_adjs)
                if len(all_target_adj_for_test) == 0:
                    all_target_adj_for_test.append(test_target_adj_to_tensor)
                all_predict_res.append(test_y_hat)
                # test community similarity
                similarity_value, com_density, com_coreness, com_nodes, query_density, predict_density, query_avg_coreness, predict_avg_coreness, query_nodes, predict_nodes, min_query_coreness, min_predict_coreness = community_similarity(test_target_adj_to_tensor,
                                                                                              test_query_adjs,
                                                                                              test_y_hat)
                # record_value_path = './model_value/'+args.dataset+'/'
                # record_single_query(record_value_path, test_path, similarity_value, com_density,
                #             com_coreness, com_nodes, query_density, predict_density, query_avg_coreness, predict_avg_coreness,
                #             query_nodes, predict_nodes, min_query_coreness, min_predict_coreness, args.train_type)
                # # test tpr
                # query_tpr_value,predict_tpr_value = triangle_participation_ratio(test_query_adjs[y], test_target_adj_to_tensor,
                #                                          test_y_hat)
                # # test diameter
                # query_diameter_value,predict_diameter_value = diameter(test_query_adjs[y], test_target_adj_to_tensor,
                #                                           test_y_hat)
                # # test cluster coefficient
                # query_cluster_coefficient_value,predict_cluster_coefficient_value = cluster_coefficient(test_query_adjs[y], test_target_adj_to_tensor,
                #                                                 test_y_hat)

            
            # print('community_similarity:  ', similarity_value)
            # community similarity
            # mean_similarity_value = mean_similarity_value + similarity_value
            # mean_com_density = mean_com_density + com_density
            # mean_com_coreness = mean_com_coreness + com_coreness
            # mean_com_nodes = mean_com_nodes + com_nodes
            #         # tpr
            #         mean_query_tpr = mean_query_tpr + query_tpr_value
            #         mean_predict_tpr = mean_predict_tpr + predict_tpr_value
            #         # diameter
            #         mean_query_diameter = mean_query_diameter + query_diameter_value
            #         mean_predict_diameter = mean_predict_diameter + predict_diameter_value
            #         # cluster coefficient
            #         mean_query_cluster_coefficient = mean_query_cluster_coefficient + query_cluster_coefficient_value
            #         mean_predict_cluster_coefficient = mean_predict_cluster_coefficient + predict_cluster_coefficient_value
            #         count = count + 1

    # count mean evluation
    # community similarity
    # print('test_data_size',test_data_size)
    # mean_similarity_value = mean_similarity_value / test_data_size
    # mean_com_density = mean_com_density / test_data_size
    # mean_com_coreness = mean_com_coreness / test_data_size
    # mean_com_nodes = mean_com_nodes / test_data_size
    # print('mean_similarity_value',mean_similarity_value)
    # print('mean_com_density',mean_com_density)
    # print('mean_com_coreness',mean_com_coreness)
    # print('mean_com_nodes',mean_com_nodes)
    
    #     # for community similarity
    #     test_community_similarity.append(mean_similarity_value)
    #     # for f1 score
    #     test_f1_score.append(mean_f1_score)
    #     # for tpr
    #     #test_tpr.append(mean_tpr)
    #     # for diameter
    #     #test_diameter.append(mean_diameter)
    #     # for cluster coefficient
    #     #test_cluster_coefficient.append(mean_cluster_coefficient)
    #     plt_epoch.append(i)
    # update threshold
    epoch_coefficient = max(100 - i, 1)
    threshold = 0.5

count = 0
for i in range(len(all_query_adj_for_test)):
    # test community similarity
    similarity_value, com_density, com_coreness, com_nodes, \
    query_density, predict_density, query_avg_coreness, predict_avg_coreness, query_nodes, predict_nodes,min_query_coreness,min_predict_coreness\
        = community_similarity(test_target_adj_to_tensor, all_query_adj_for_test[i], all_predict_res[i])
    print('community_similarity:  ', similarity_value)
        
    # test tpr
    query_tpr_value, predict_tpr_value = triangle_participation_ratio(all_query_adj_for_test[i],
                                                                      test_target_adj_to_tensor,
                                                                      all_predict_res[i])
    # test diameter
    query_diameter_value, predict_diameter_value = diameter(all_query_adj_for_test[i], test_target_adj_to_tensor,
                                                            all_predict_res[i])
    # test cluster coefficient
    query_cluster_coefficient_value, predict_cluster_coefficient_value = cluster_coefficient(
        all_query_adj_for_test[i], test_target_adj_to_tensor,
        all_predict_res[i])

    # print('tpr_value', predict_tpr_value)
    # print('diameter_value', predict_diameter_value)
    # print('cluster_coefficient_value', predict_cluster_coefficient_value)
    # community similarity
    # if similarity_value > 0.9:
    # print('start')
    mean_similarity_value = mean_similarity_value + similarity_value
    mean_com_density = mean_com_density + com_density
    mean_com_coreness = mean_com_coreness + com_coreness
    mean_com_nodes = mean_com_nodes + com_nodes
    mean_query_density = mean_query_density + query_density
    mean_predict_density = mean_predict_density + predict_density
    mean_query_avg_coreness = mean_query_avg_coreness + query_avg_coreness
    mean_predict_avg_coreness = mean_predict_avg_coreness + predict_avg_coreness
    mean_query_nodes = mean_query_nodes + query_nodes
    mean_predict_nodes = mean_predict_nodes + predict_nodes
    # # tpr
    mean_query_tpr = mean_query_tpr + query_tpr_value
    mean_predict_tpr = mean_predict_tpr + predict_tpr_value
    # # diameter
    mean_query_diameter = mean_query_diameter + query_diameter_value
    mean_predict_diameter = mean_predict_diameter + predict_diameter_value
    # # cluster coefficient
    mean_query_cluster_coefficient = mean_query_cluster_coefficient + query_cluster_coefficient_value
    mean_predict_cluster_coefficient = mean_predict_cluster_coefficient + predict_cluster_coefficient_value
    count = count + 1
print('count',count)
mean_similarity_value = mean_similarity_value / count
mean_com_density = mean_com_density / count
mean_com_coreness = mean_com_coreness / count
mean_com_nodes = mean_com_nodes / count
mean_query_density = mean_query_density / count
mean_predict_density = mean_predict_density / count
mean_query_avg_coreness = mean_query_avg_coreness / count
mean_predict_avg_coreness = mean_predict_avg_coreness / count
mean_query_nodes = mean_query_nodes / count
mean_predict_nodes = mean_predict_nodes / count
# tpr
mean_query_tpr = mean_query_tpr / count
mean_predict_tpr = mean_predict_tpr / count
# diameter
mean_query_diameter = mean_query_diameter / count
mean_predict_diameter = mean_predict_diameter / count
# cluster coefficient
mean_query_cluster_coefficient = mean_query_cluster_coefficient / count
mean_predict_cluster_coefficient = mean_predict_cluster_coefficient / count
print('mean_similarity_value', mean_similarity_value)
print('mean_com_density', mean_com_density)
print('mean_com_coreness', mean_com_coreness)
print('mean_com_nodes', mean_com_nodes)
print('mean_query_density', mean_query_density)
print('mean_predict_density', mean_predict_density)
print('mean_query_avg_coreness', mean_query_avg_coreness)
print('mean_predict_avg_coreness', mean_predict_avg_coreness)
print('mean_query_nodes', mean_query_nodes)
print('mean_predict_nodes', mean_predict_nodes)
# tpr
print('mean_query_tpr', mean_query_tpr)
print('mean_predict_tpr', mean_predict_tpr)
# diameter
print('mean_query_diameter', mean_query_diameter)
print('mean_predict_diameter', mean_predict_diameter)
# cluster coefficient
print('mean_query_cluster_coefficient', mean_query_cluster_coefficient)
print('mean_predict_cluster_coefficient', mean_predict_cluster_coefficient)
print('test time',all_test_time)