import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import dgl
from net import scs_GMN
from sklearn.metrics import f1_score
import argparse
import time
import datetime
from candidate_get import candidate_generate_depend_on_degree, candidate_generate_depend_on_core \
    , faster_candidate_generate_depend_on_core, candidate_generate_all_depend_on_core \
    , candidate_generate_depend_on_core_for_large_graph
from test_evaluation import community_similarity, f1_score, triangle_participation_ratio, diameter, cluster_coefficient, \
    label_cover_rate
import matplotlib.pyplot as plt


def record_time(save_path, dataset, all_for_train_time, train_candidate_time, all_test_time, test_candidate_time):
    file_path = save_path + 'time.txt'
    now_time = datetime.datetime.now()
    with open(file_path, 'a') as f:
        f.write('record time :' + str(now_time) + 's' + ' ' + 'dataset:' + str(
            dataset) + 's' + ' ' + 'all_for_train :' + str(all_for_train_time) + 's'
                + ' ' + 'train_candidate_time :' + str(train_candidate_time) + 's' + ' ' + 'all_test_time :' + str(
            all_test_time) + 's' +
                ' ' + 'test_candidate_time :' + str(test_candidate_time) + 's' + '\n')
        f.close()


def record_value(save_path, dataset, community_similarity, com_density, com_coreness, com_nodes, f1_score, tpr,
                 diameter, cluster_coefficient):
    file_path = save_path + 'record_value.txt'
    now_time = datetime.datetime.now()
    with open(file_path, 'a') as f:
        f.write('record time :' + str(now_time) + 's' + ' ' + 'dataset:' + str(dataset) + ' '
                + 'community_similarity :' + str(community_similarity) + ' ' + 'compared_density :' + str(
            com_density) + ' '
                + 'compared_coreness :' + str(com_coreness) + ' ' + 'compared_nodes :' + str(com_nodes) + ' '
                + 'f1_score :' + str(f1_score) + ' '
                + 'tpr :' + str(tpr) + ' '
                + 'diameter :' + str(diameter) + ' '
                + 'cluster_coefficient :' + str(cluster_coefficient) + '\n')
        f.close()



def remove_dumb_nodes(adj):
    np_adj = adj.cpu().numpy()  # 存在自环边
    graph = nx.from_numpy_array(np_adj)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    remove = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    graph.remove_nodes_from(remove)
    removed_adj = nx.adjacency_matrix(graph).todense()
    removed_adj_add_self = removed_adj + np.eye(removed_adj.shape[0])
    return torch.tensor(removed_adj_add_self)

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--GCN_in_size', type=int, default=1437)
parser.add_argument('--GCN_out_size', type=int, default=2)
parser.add_argument('--da_size', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_size', type=int, default=2)
parser.add_argument('--candidate_method', type=str, default='faster_coreness',help='faster_coreness or for large graph')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--trade_off_for_re1', type=float, default=1,help='balance joint loss')
parser.add_argument('--trade_off_for_re2', type=float, default=0.001,help='balance joint loss')
parser.add_argument('--trade_off_for_re3', type=float, default=0.01,help='balance joint loss')
parser.add_argument('--train_type', type=str, default='both',help='both,only label,only structure')
args = parser.parse_args()

threshold = 1
# data
train_path = './dataset/'+ args.dataset +'/for_train_' + args.dataset + '/' # ./dataset/for_train_4_17/\./dataset/for_train_cora/
test_path = './dataset/' + args.dataset +'/for_test_' + args.dataset + '/'
batch_size = args.batch_size
test_size = args.test_size # 自建数据集 test_size=40 cora数据集 test_size=20
epoch = args.epoch
#device = torch.device('cuda:1')
candidate_method = args.candidate_method  # degree\coreness\faster_coreness\rough_faster_coreness\for_large_graph
save_model = True
trade_off_for_re1 = args.trade_off_for_re1
trade_off_for_re2 = args.trade_off_for_re2
trade_off_for_re3 = args.trade_off_for_re3

train_target_features = torch.load(train_path + 'nor_target_features_cat_degree_cluster_h_index_coreness.pt')
train_target_features_to_tensor = torch.tensor(train_target_features[0])
#train_target_features_to_tensor = train_target_features_to_tensor.to(device)
# train_target_structure_features = torch.load(train_path + 'nor_target_only_degree_cluster_h_index_coreness')
train_target_adj = torch.load(train_path + 'target_adj.pt')
train_target_adj_to_tensor = torch.tensor(train_target_adj[0])
target_size = train_target_adj_to_tensor.shape[0]
#train_target_adj_to_tensor = train_target_adj_to_tensor.to(device)
train_target_att_adj = torch.load(train_path + 'degree_based_target_adjs.pt')
train_query_features = torch.load(train_path + 'nor_0.7_query_features_cat_degree_cluster_h_index_coreness.pt')
# train_query_structure_features = torch.load(train_path + 'nor_query_only_degree_cluster_h_index_coreness')
train_query_adj = torch.load(train_path + 'query_adj.pt')
train_query_att_adj = torch.load(train_path + 'degree_based_query_adjs.pt')
train_labels = torch.load(train_path + 'labels.pt')
# print(train_labels.shape)

test_target_features = torch.load(test_path + 'nor_target_features_cat_degree_cluster_h_index_coreness.pt')
test_target_features_to_tensor = torch.tensor(test_target_features[0])
#test_target_features_to_tensor = test_target_features_to_tensor.to(device)
test_target_adj = torch.load(test_path + 'target_adj.pt')
test_target_adj_to_tensor = torch.tensor(test_target_adj[0])
#test_target_adj_to_tensor = test_target_adj_to_tensor.to(device)
test_target_att_adj = torch.load(test_path + 'degree_based_target_adjs.pt')
test_query_features = torch.load(test_path + 'nor_0.7_query_features_cat_degree_cluster_h_index_coreness.pt')
test_query_adj = torch.load(test_path + 'query_adj.pt')
test_query_att_adj = torch.load(test_path + 'degree_based_query_adjs.pt')
test_labels = torch.load(test_path + 'labels.pt')
test_data_size = int(test_query_features.shape[0])

traindata = []
testdata = []

# print(train_target_features.shape[0]) #shape[0]可以取出第一维的长度
for i in range(int(train_query_features.shape[0])):
    one_traindata = []
    # one_traindata.append(train_target_features[i])
    # one_traindata.append(train_target_adj[i])
    one_traindata.append(train_query_features[i])
    one_traindata.append(train_query_adj[i])
    one_traindata.append(train_labels[i])
    # one_traindata.append(train_target_att_adj[i])
    one_traindata.append(train_query_att_adj[i])
    traindata.append(one_traindata)

# prepare test data
for i in range(int(test_query_features.shape[0])):
    one_testdata = []
    # one_testdata.append(test_target_features[i])
    # one_testdata.append(test_target_adj[i])
    one_testdata.append(test_query_features[i])
    one_testdata.append(test_query_adj[i])
    one_testdata.append(test_labels[i])
    # one_testdata.append(test_target_att_adj[i])
    one_testdata.append(test_query_att_adj[i])
    testdata.append(one_testdata)
    in_size = test_query_features[i].shape[1]

data_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
data_test = DataLoader(testdata, batch_size=test_size, shuffle=True)

# setting
GCN_in_size = in_size  # features_dimension # cora 1433/Citeseer 3703/pubmed 500/deezer 4463/facebook 4714
GCN_out_size = args.GCN_out_size
da_size = target_size # target size Cora 2708/Citeseer 3312/pubmed 19717/deezer 28281/facebook 22470

# model
model = scs_GMN(GCN_in_size, GCN_out_size,  da_size)

#model.to(device)

# setting
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Adam会自动调整学习率
loss_fun1 = torch.nn.MSELoss(reduction='mean')
#loss_fun1 = loss_fun1.to(device)
triplet_loss = torch.nn.TripletMarginLoss(reduction='mean')
crossentropyloss = torch.nn.CrossEntropyLoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')
#l1_loss = l1_loss.to(device)

# about plot
test_loss_record = []
# for community similarity
test_community_similarity = []
# for f1 score
test_f1_score = []
# for TPR
test_tpr = []
# for diameter
test_diameter = []
# for cluster coefficient
test_cluster_coefficient = []
plt_epoch = []

# time
all_for_train_time = 0
all_train_candidate_time = 0
all_test_time = 0
all_test_candidate_time = 0
for i in np.arange(epoch):
    print('epoch:  ', i)
    epoch_loss = []
    start = time.time()
    model.train()  # 切换训练模式
    for j, batch in enumerate(data_loader):  # j为索引，一个batch
        # print(batch[1].shape) batch[1].shape -> [batch_size,number_of_nodes,features_dimension]
        # for item in batch[1]:
        #     print(item.shape)
        # b_target_features = torch.tensor(batch[0])
        # b_target_features = b_target_features.to(device)
        # b_target_adjs = batch[1]
        # b_target_adjs = b_target_adjs.to(device)
        b_query_features = torch.tensor(batch[0])
        #b_query_features = b_query_features.to(device)
        b_query_adjs = batch[1]
        #b_query_adjs = b_query_adjs.to(device)
        b_labels = batch[2]
        #b_labels = b_labels.to(device)
        # b_target_att_adjs = batch[5]
        b_query_att_adjs = batch[3]

        # print('b_target_features :', b_target_features[0].shape)
        # print('b_target_adjs :', b_target_adjs[0].shape)
        # print('b_target_features :',b_target_features.shape)
        # print('b_target_adjs :',b_target_adjs.shape)
        # print('b_query_features',b_query_features.shape)
        # print('b_query_adjs',b_query_adjs.shape)

        predicted_y_hat = torch.zeros(batch_size, 1, da_size)
        target_graph_embedding = torch.zeros(batch_size, 1, GCN_out_size)
        query_graph_embedding = torch.zeros(batch_size, 1, GCN_out_size)
        # reconstruct loss
        # reconstruct_query_adj = torch.empty(batch_size,q_size,q_size)
        # reconstruct_target_adj = torch.empty(batch_size,da_size,da_size)
        # predicted_adj = torch.empty(batch_size,q_size,q_size)
        # triplet loss
        batch_positive_samples = []
        batch_negative_samples = []
        batch_anchor_samples = []
        # reconstruct structural information loss
        reconstruct_degree = torch.zeros(batch_size, 1)
        reconstruct_edges = torch.zeros(batch_size, 1)
        reconstruct_nodes = torch.zeros(batch_size, 1)
        reconstruct_degree_from_query = torch.zeros(batch_size, 1)
        reconstruct_edges_from_query = torch.zeros(batch_size, 1)
        reconstruct_nodes_from_query = torch.zeros(batch_size, 1)
        reconstruct_degree_from_neg = torch.zeros(batch_size, 1)
        reconstruct_edges_from_neg = torch.zeros(batch_size, 1)
        batch_re_adj_loss = torch.zeros(1, 1)
        #batch_re_adj_loss = batch_re_adj_loss.to(device)
        for k in range(b_query_features.shape[0]):
            # print('k',k)
            start_candidate = time.time()
            if candidate_method == 'degree':
                candidate_set, candidate_adj = candidate_generate_depend_on_degree(train_target_adj_to_tensor,
                                                                                   b_query_adjs[k])
            elif candidate_method == 'coreness':
                candidate_set, candidate_adj = candidate_generate_depend_on_core(train_target_adj_to_tensor,
                                                                                 b_query_adjs[k])
            elif candidate_method == 'faster_coreness':
                candidate_set, candidate_adj = faster_candidate_generate_depend_on_core(
                    train_target_adj_to_tensor,
                    b_query_adjs[k])
            elif candidate_method == 'rough_faster_coreness':
                candidate_set, candidate_adj = candidate_generate_all_depend_on_core(train_target_adj_to_tensor,
                                                                                     b_query_adjs[k])
            elif candidate_method == 'for_large_graph':
                candidate_set, candidate_adj = candidate_generate_depend_on_core_for_large_graph(
                    train_target_adj_to_tensor,
                    b_query_adjs[k])
            #candidate_adj = candidate_adj.to(device)
            ###################print candidate time
            end_candidate = time.time() - start_candidate
            all_train_candidate_time = all_train_candidate_time + end_candidate
            # print("候选选择时间:%.2f秒" % (end_candidate - start_candidate),k)
            # get degree_dis based adj
            att_target_adj = torch.tensor(train_target_att_adj[0])
            att_query_adj = b_query_att_adjs[k]
            # model
            y_hat, att_da2, att_q2, \
            avg_degree, density, avg_nodes, neg_avg_degree, neg_density, neg_avg_nodes, re_adj, ori_adj = \
                model(train_target_adj_to_tensor, att_target_adj, train_target_features_to_tensor,
                      b_query_adjs[k], att_query_adj, b_query_features[k], candidate_set, candidate_adj, threshold)
            # q1,q2是一张查询图的图节点嵌入
            # att_da1,att_da2是一张目标图的图节点嵌入
            predicted_y_hat[k] = y_hat
            reconstruct_degree[k] = avg_degree
            reconstruct_edges[k] = density
            reconstruct_nodes[k] = avg_nodes
            reconstruct_degree_from_neg[k] = neg_avg_degree
            reconstruct_edges_from_neg[k] = neg_density
            #re_adj = re_adj.to(device)
            #ori_adj = ori_adj.to(device)
            # get reconstruct adj
            # reconstruct_target_adj[k] = re_target_adj
            # reconstruct_query_adj[k] = re_query_adj
            # reconstruct adj loss
            re_adj_loss = loss_fun1(re_adj, ori_adj)
            batch_re_adj_loss = batch_re_adj_loss + re_adj_loss

            # degree weight sum
            target_degree = torch.sum(train_target_adj_to_tensor, dim=1)[candidate_set]
            # print('target_degree',target_degree)
            query_degree = torch.sum(b_query_adjs[k], dim=1)
            sum_target_degree = int(torch.sum(target_degree, dim=0))
            # print('sum_target_degree',sum_target_degree)
            sum_query_degree = int(torch.sum(query_degree, dim=0))
            nor_target_degree = torch.div(target_degree, sum_target_degree).float()  # 1xN
            nor_query_degree = torch.div(query_degree, sum_query_degree).float()  # 1xN
            # print('nor_target_degree.shape',nor_target_degree.shape)
            # print('nor_target_degree',nor_target_degree)

            # get pooled graph embedding
            # pool_target_graph_embedding = torch.mean(att_da2[candidate_set],dim=0)
            # pool_query_graph_embedding = torch.mean(att_q2,dim=0)
            pool_target_graph_embedding = torch.matmul(nor_target_degree, att_da2[candidate_set])
            pool_query_graph_embedding = torch.matmul(nor_query_degree, att_q2)
            target_graph_embedding[k] = pool_target_graph_embedding
            query_graph_embedding[k] = pool_query_graph_embedding

            # get query avg degree and query avg edges
            # nor_query_adj = torch.nn.functional.normalize(b_query_adjs[k],p=2,dim=1)
            # remove dumb nodes
            remove_dumb_adj = remove_dumb_nodes(b_query_adjs[k])
            if avg_degree == 0:
                avg_degree_from_query = 0
            else:
                avg_degree_from_query = torch.mean(torch.sum(remove_dumb_adj, dim=1))
            if density == 0:
                density_from_query = 0
            else:
                density_from_query = 2 * torch.sum(remove_dumb_adj) / (
                            torch.trace(remove_dumb_adj) * (torch.trace(remove_dumb_adj) - 1) + 0.0001)
            if avg_nodes == 0:
                avg_nodes_from_query = 0
            else:
                avg_nodes_from_query = torch.trace(remove_dumb_adj)

            reconstruct_degree_from_query[k] = float(avg_degree_from_query)
            reconstruct_edges_from_query[k] = float(density_from_query)
            reconstruct_nodes_from_query[k] = float(avg_nodes_from_query)


        # loss
        #predicted_y_hat = predicted_y_hat.to(device)
        #reconstruct_degree = reconstruct_degree.to(device)
        #reconstruct_edges = reconstruct_edges.to(device)
        #reconstruct_nodes = reconstruct_nodes.to(device)
        #reconstruct_degree_from_query = reconstruct_degree_from_query.to(device)
        #reconstruct_edges_from_query = reconstruct_edges_from_query.to(device)
        #reconstruct_nodes_from_query = reconstruct_nodes_from_query.to(device)


        loss1 = loss_fun1(predicted_y_hat, b_labels)
        re_loss1 = l1_loss(reconstruct_degree, reconstruct_degree_from_query)
        re_loss2 = l1_loss(reconstruct_edges, reconstruct_edges_from_query)
        re_loss3 = l1_loss(reconstruct_nodes, reconstruct_nodes_from_query)
        print('re_loss1', re_loss1)
        print('re_loss2', re_loss2)
        mean_re_adj_loss = torch.div(batch_re_adj_loss, batch_size)
        # loss2 = loss_fun1(target_graph_embedding,query_graph_embedding)
        # reconstruct loss
        # reconstruct_query_loss = loss_fun1(reconstruct_query_adj,b_query_adjs.float())
        # reconstruct_target_loss = loss_fun1(reconstruct_target_adj,b_target_adjs.float())
        # tri_loss = triplet_loss(re_anchor_samples,re_positive_samples,re_negative_samples)
        # csloss = crossentropyloss(predicted_y_hat,b_labels)

        # total_loss = loss1 + 0.1 * loss2
        # total_loss = loss1 + tri_loss
        # total_loss = loss1 + re_loss1 + re_loss2 + mean_re_adj_loss
        # for cora re1 - 0.01 re2 - 0.0001
        # for citeseer re1 - 0.01 re2 - 0.0001
        # total_loss = loss1 + 0.01*re_loss1 + 0.0001*re_loss2
        print('re_loss3', re_loss3)
        if args.train_type == 'both':
            if i < 100:
                total_loss = loss1
            else:
                total_loss = loss1 + 0.01 * trade_off_for_re1 * re_loss1 + 0.01 * trade_off_for_re2 * re_loss2 + 0.01 * trade_off_for_re3 * re_loss3
        elif args.train_type == 'only label':
            total_loss = loss1
        elif args.train_type == 'only structure':
            total_loss = 0.01 * trade_off_for_re1 * re_loss1 + 0.01 * trade_off_for_re2 * re_loss2 + 0.01 * trade_off_for_re3 * re_loss3
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss.append(float(total_loss.detach()))

        # save weight
        # state_dict = model.state_dict()
        # torch.save(state_dict, f'./dataset/2023.5.4/old_adj_one_hot/model_parameters_epoch_{i}_batch_{j}.pt')
    print('epoch_loss:  ', np.mean(epoch_loss))
    end = time.time() - start
    all_for_train_time = all_for_train_time + end
    print('训练总耗时 {:.0f}m {:.0f}s'.format(end // 60, end % 60))

    # test stage
    model.eval()  # 切换为测试模式

    # plt
    mean_similarity_value = 0
    mean_com_density = 0
    mean_com_coreness = 0
    mean_com_nodes = 0
    mean_f1_score = 0
    mean_tpr = 0
    mean_diameter = 0
    mean_cluster_coefficient = 0
    # label cover rate
    mean_cover_rate = 0

    for x, test_batch in enumerate(data_test):  # test data
        print('test stage')
        # test_target_features = torch.tensor(test_batch[0])
        # test_target_adjs = test_batch[1]
        test_query_features = torch.tensor(test_batch[0])
        #test_query_features = test_query_features.to(device)
        test_query_adjs = test_batch[1]
        #test_query_adjs = test_query_adjs.to(device)
        test_labels = test_batch[2]
        #test_labels = test_labels.to(device)
        # test_target_att_adjs = test_batch[5]
        test_query_att_adjs = test_batch[3]
        # test_query_att_adjs = test_query_att_adjs.to(device)

        test_all_y_hat = torch.zeros(test_size, 1, da_size)
        test_target_graph_embedding = torch.zeros(test_size, 1, GCN_out_size)
        test_query_graph_embedding = torch.zeros(test_size, 1, GCN_out_size)
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
        # negatigve samples
        test_reconstruct_degree_from_neg = torch.zeros(test_size, 1)
        test_reconstruct_edges_from_neg = torch.zeros(test_size, 1)
        test_batch_adj_loss = torch.zeros(1, 1)
        #test_batch_adj_loss = test_batch_adj_loss.to(device)
        for y in range(test_query_features.shape[0]):  # one batch
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
                    test_query_adjs[y])
            elif candidate_method == 'rough_faster_coreness':
                test_candidate_set, test_candidate_adj = candidate_generate_all_depend_on_core(
                    test_target_adj_to_tensor,
                    test_query_adjs[y])
            elif candidate_method == 'for_large_graph':
                test_candidate_set, test_candidate_adj = candidate_generate_depend_on_core_for_large_graph(
                    test_target_adj_to_tensor, test_query_adjs[y])
            # model test output
            test_end_candidate = time.time() - test_start_candidate
            all_test_candidate_time = all_test_candidate_time + test_end_candidate
            # print("候选选择时间:%.2f秒" % (test_end_candidate - test_start_candidate))
            #test_candidate_adj = test_candidate_adj.to(device)
            start_test = time.time()
            with torch.no_grad():
                # get degree_dis based adj
                test_att_target_adj = torch.tensor(test_target_att_adj[0])
                test_att_query_adj = test_query_att_adjs[y]
                test_y_hat, att_da2, att_q2, \
                test_avg_degree, test_density, test_avg_nodes, test_neg_avg_degree, test_neg_density, test_neg_avg_nodes, test_re_adj, test_ori_adj \
                    = model(test_target_adj_to_tensor, test_att_target_adj,
                            test_target_features_to_tensor,
                            test_query_adjs[y], test_att_query_adj, test_query_features[y],
                            test_candidate_set, test_candidate_adj, threshold)
                end_test = time.time() - start_test
                all_test_time = all_test_time + end_test
                test_all_y_hat[y] = test_y_hat
                test_reconstruct_degree[y] = test_avg_degree
                test_reconstruct_edges[y] = test_density
                test_reconstruct_nodes[y] = test_avg_nodes
                test_reconstruct_degree_from_neg[y] = test_neg_avg_degree
                test_reconstruct_edges_from_neg[y] = test_neg_density
                #test_re_adj = test_re_adj.to(device)
                #test_ori_adj = test_ori_adj.to(device)

                # get reconstruct adj
                # test_reconstruct_target_adj[y] = re_target_adj
                # test_reconstruct_query_adj[y] = re_query_adj
                # reconstruct adj loss
                test_re_adj_loss = loss_fun1(test_re_adj, test_ori_adj)
                test_batch_adj_loss = test_batch_adj_loss + test_re_adj_loss

                # degree weight sum
                test_target_degree = torch.sum(test_target_adj_to_tensor, dim=1)[test_candidate_set]
                test_query_degree = torch.sum(test_query_adjs[y], dim=1)
                test_sum_target_degree = int(torch.sum(test_target_degree, dim=0))
                test_sum_query_degree = int(torch.sum(test_query_degree, dim=0))
                test_nor_target_degree = torch.div(test_target_degree, test_sum_target_degree).float()
                test_nor_query_degree = torch.div(test_query_degree, test_sum_query_degree).float()

                # test_pool_target_graph_embedding = torch.mean(att_da2[test_candidate_set],dim=0)
                # test_pool_query_graph_embedding = torch.mean(att_q2,dim=0)
                test_pool_target_graph_embedding = torch.matmul(test_nor_target_degree, att_da2[test_candidate_set])
                test_pool_query_graph_embedding = torch.matmul(test_nor_query_degree, att_q2)
                test_target_graph_embedding[y] = test_pool_target_graph_embedding
                test_query_graph_embedding[y] = test_pool_query_graph_embedding

                # sample
                # positive_samples = select_positive_sample(test_labels[y], att_da2, sample_size, test_target_adjs[y],
                #                                           test_query_adjs[y])
                # negative_samples = select_negative_sample(test_labels[y], att_da2, sample_size, test_target_adjs[y],
                #                                           test_query_adjs[y])
                # anchor_samples = att_q2.repeat_interleave(sample_size, dim=0)
                # batch_positive_samples_for_test.append(positive_samples)
                # batch_negative_samples_for_test.append(negative_samples)
                # batch_anchor_samples_for_test.append(anchor_samples)

                # get query avg degree and query avg edges
                # nor_test_query_adj = torch.nn.functional.normalize(test_query_adjs[y], p=2, dim=1)
                # remove dumb nodes
                remove_dumb_adj = remove_dumb_nodes(test_query_adjs[y])
                if test_avg_degree == 0:
                    avg_degree_from_query = 0
                else:
                    avg_degree_from_query = torch.mean(torch.sum(remove_dumb_adj, dim=1))
                if test_density == 0:
                    density_from_query = 0
                else:
                    density_from_query = 2 * torch.sum(remove_dumb_adj) / (
                                torch.trace(remove_dumb_adj) * (torch.trace(remove_dumb_adj) - 1) + 0.0001)
                if test_avg_nodes == 0:
                    avg_nodes_from_query = 0
                else:
                    avg_nodes_from_query = torch.trace(remove_dumb_adj)

                # if i == 0 and x==0 and y==0:
                #     plt_adj_in_hot(test_query_adjs[y])
                # print('torch.mean(nor_test_query_adj',avg_degree_from_query)
                # print('torch.sum(nor_test_query_adj)',density_from_query)
                test_reconstruct_degree_from_query[y] = float(avg_degree_from_query)
                test_reconstruct_edges_from_query[y] = float(density_from_query)
                test_reconstruct_nodes_from_query[y] = float(avg_nodes_from_query)
                # plt query adj and reconstruct adj
                # if (i + 1) % 20 == 0 and x==0 and y==0 or i == 0 and x==0 and y==0:
                #     plt_adj_in_hot(test_re_adj)
                #     plt_adj_in_hot(test_ori_adj)
                #     print('mean_gap', test_avg_degree - avg_degree_from_query)

                # './dataset/2023.6.21/cora_dataset/', y_hat, q_size, da_size, i, j, k,save_matching_matrix

                # test community similarity
                similarity_value, com_density, com_coreness, com_nodes = community_similarity(test_target_adj_to_tensor,
                                                                                              test_query_adjs[y],
                                                                                              test_y_hat)
                # test f1 score
                f1_score_value = f1_score(test_y_hat, test_labels[y])
                # test tpr
                tpr_value = triangle_participation_ratio(test_query_adjs[y], test_target_adj_to_tensor,
                                                         test_y_hat)
                # test diameter
                diameter_value = diameter(test_query_adjs[y], test_target_adj_to_tensor,
                                          test_y_hat)
                # test cluster coefficient
                cluster_coefficient_value = cluster_coefficient(test_query_adjs[y], test_target_adj_to_tensor,
                                                                test_y_hat)
                # test label cover rate
                predicted_cover_rate = label_cover_rate(test_y_hat, test_labels[y])

            if (i + 1) % 1 == 0:
                print('community_similarity:  ', similarity_value)
                # community similarity
                mean_similarity_value = mean_similarity_value + similarity_value
                mean_com_density = mean_com_density + com_density
                mean_com_coreness = mean_com_coreness + com_coreness
                mean_com_nodes = mean_com_nodes + com_nodes
                # f1 score
                mean_f1_score = mean_f1_score + f1_score_value
                # tpr
                mean_tpr = mean_tpr + tpr_value
                # diameter
                mean_diameter = mean_diameter + diameter_value
                # cluster coefficient
                mean_cluster_coefficient = mean_cluster_coefficient + cluster_coefficient_value
                # label cover rate
                mean_cover_rate = mean_cover_rate + predicted_cover_rate

        # triplet loss for test
        # batch_positive_samples_toTensor_for_test = torch.stack(batch_positive_samples_for_test)
        # batch_negative_samples_toTensor_for_test = torch.stack(batch_negative_samples_for_test)
        # batch_anchor_samples_toTensor_for_test = torch.stack(batch_anchor_samples_for_test)
        # re_positive_samples_for_test = torch.reshape(batch_positive_samples_toTensor_for_test,
        #                                     (test_size * sample_size * q_size, GCN_out_size))
        # re_negative_samples_for_test = torch.reshape(batch_negative_samples_toTensor_for_test,
        #                                     (test_size * sample_size * q_size, GCN_out_size))
        # re_anchor_samples_for_test = torch.reshape(batch_anchor_samples_toTensor_for_test,
        #                                   (test_size * sample_size * q_size, GCN_out_size))
        # output test loss
        #test_all_y_hat = test_all_y_hat.to(device)
        #test_reconstruct_degree = test_reconstruct_degree.to(device)
        #test_reconstruct_edges = test_reconstruct_edges.to(device)
        #test_reconstruct_nodes = test_reconstruct_nodes.to(device)
        #test_reconstruct_degree_from_query = test_reconstruct_degree_from_query.to(device)
        #test_reconstruct_edges_from_query = test_reconstruct_edges_from_query.to(device)
        #test_reconstruct_nodes_from_query = test_reconstruct_nodes_from_query.to(device)
        loss1 = loss_fun1(test_all_y_hat, test_labels)
        re_loss1 = l1_loss(test_reconstruct_degree, test_reconstruct_degree_from_query)
        re_loss2 = l1_loss(test_reconstruct_edges, test_reconstruct_edges_from_query)
        re_loss3 = l1_loss(test_reconstruct_nodes, test_reconstruct_nodes_from_query)
        test_mean_adj_loss = torch.div(test_batch_adj_loss, test_size)
        if args.train_type == 'both':
            if i < 100:  # 按batch算loss
                test_loss = loss1
            else:
                test_loss = loss1 + 0.01 * trade_off_for_re1 * re_loss1 + 0.01 * trade_off_for_re2 * re_loss2 + 0.01 * trade_off_for_re3 * re_loss3
        elif args.train_type == 'only label':
            test_loss = loss1
        elif args.train_type == 'only structure':
            test_loss = 0.01 * trade_off_for_re1 * re_loss1 + 0.01 * trade_off_for_re2 * re_loss2 + 0.01 * trade_off_for_re3
        # test_loss = loss1
        # print('test_loss:  ', np.float(test_loss.detach()))

    # count mean evluation
    if (i + 1) % 1 == 0:
        # community similarity
        mean_similarity_value = mean_similarity_value / test_data_size
        mean_com_density = mean_com_density / test_data_size
        mean_com_coreness = mean_com_coreness / test_data_size
        mean_com_nodes = mean_com_nodes / test_data_size
        # f1 score
        mean_f1_score = mean_f1_score / test_data_size
        # tpr
        mean_tpr = mean_tpr / test_data_size
        # diameter
        mean_diameter = mean_diameter / test_data_size
        # cluster coefficient
        mean_cluster_coefficient = mean_cluster_coefficient / test_data_size
        # label cover rate
        mean_cover_rate = mean_cover_rate / test_data_size

        test_loss_record.append(test_loss.cpu())
        # for community similarity
        test_community_similarity.append(mean_similarity_value)
        # for f1 score
        test_f1_score.append(mean_f1_score)
        # for tpr
        test_tpr.append(mean_tpr)
        # for diameter
        test_diameter.append(mean_diameter)
        # for cluster coefficient
        test_cluster_coefficient.append(mean_cluster_coefficient)
        plt_epoch.append(i)
        if (i + 1) % 50 == 0:
            record_value('./model_value/deezer/', train_path, mean_similarity_value, mean_com_density,
                         mean_com_coreness, mean_com_nodes, mean_f1_score, mean_tpr, mean_diameter,
                         mean_cluster_coefficient)
    # update threshold
    epoch_coefficient = max(100 - i, 1)
    # threshold = epoch_coefficient * max(0.5, ((1 - mean_cover_rate) * (1 - mean_similarity_value)))
    threshold = epoch_coefficient * 0.5
    print('threshold', threshold)

    # save model state
    if save_model == True and (i + 1) % 100 == 0:
        model_save_path = './model_save/GMN_for_cora/GMN_for_cora_' + str(i + 1) + '.pth'
        torch.save(model.state_dict(), model_save_path)
# record time
record_time('./model_time/cora/', train_path, all_for_train_time, all_train_candidate_time, all_test_time,
            all_test_candidate_time)
# plt
plt.figure(1)
# loss
test_loss_line = plt.plot(plt_epoch, test_loss_record, 'r', lw=1)

plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./model_plt/cora/loss.png')
plt.close()
# plt.show()

# community similarity
plt.figure(2)
community_similarity_line = plt.plot(plt_epoch, test_community_similarity, 'g', lw=1, label='community similarity')

plt.legend()
# plt.ylim((0,1))
plt.xlabel('epoch')
plt.ylabel('community similarity')
plt.savefig('./model_plt/cora/community_similarity.png')
plt.close()
# plt.show()

# f1 score
plt.figure(3)
f1_score_line = plt.plot(plt_epoch, test_f1_score, 'g', lw=1, label='f1 score')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.savefig('./model_plt/cora/f1_score.png')
plt.close()
# plt.show()


# triangle participation ratio
plt.figure(4)
tpr_line = plt.plot(plt_epoch, test_tpr, 'g', lw=1, label='TPR')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('TPR')
plt.savefig('./model_plt/cora/TPR.png')
plt.close()
# plt.show()

# diameter
plt.figure(5)
diameter_line = plt.plot(plt_epoch, test_diameter, 'g', lw=1, label='diameter')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('diameter')
plt.savefig('./model_plt/cora/diameter.png')
plt.close()
# plt.show()

# cluster coefficient
plt.figure(6)
cluster_coefficient_line = plt.plot(plt_epoch, test_cluster_coefficient, 'g', lw=1, label='cluster coefficient')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('cluster coefficient')
plt.savefig('./model_plt/cora/cluster_coefficient.png')
plt.close()
# plt.show()
