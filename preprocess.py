# cora
from process_cora_dataset import load_cora_data as load_cora_0_1
from process_cora_dataset_0_2 import load_cora_data as load_cora_0_2
from process_cora_dataset_0_3 import load_cora_data as load_cora_0_3
# citeseer
from process_citeseer_dataset import load_citeseer_data as load_citeseer_0_1
from process_citeseer_dataset_0_2 import load_citeseer_data as load_citeseer_0_2
from process_citeseer_dataset_0_3 import load_citeseer_data as load_citeseer_0_3
# pubmed
from process_pubmed_dataset import load_pubmed_data as load_pubmed_0_1
from process_pubmed_dataset_0_2 import load_pubmed_data as load_pubmed_0_2
from process_pubmed_dataset_0_3 import load_pubmed_data as load_pubmed_0_3
# deezer
from process_deezer_dataset import load_deezer_data as load_deezer_0_1
from process_deezer_dataset_0_2 import load_deezer_data as load_deezer_0_2
from process_deezer_dataset_0_3 import load_deezer_data as load_deezer_0_3
# facebook
from process_facebook_dataset import load_facebook_data as load_facebook_0_1
from process_facebook_dataset_0_2 import load_facebook_data as load_facebook_0_2
from process_facebook_dataset_0_3 import load_facebook_data as load_facebook_0_3
# mask fature
from mask_feature import process_to_masked_features
# cat structural information
from cat_structural_info import cat_structural_information
import argparse
import torch
def preprocess_data(dataset,train_path,test_path,perturbation,train_size=60,test_size=20):
    if dataset == 'cora':
        if perturbation == 0.1:
            load_cora_0_1(train_size,test_size,train_path,test_path)
        elif perturbation == 0.2:
            load_cora_0_2(train_size,test_size,train_path,test_path)
        elif perturbation == 0.3:
            load_cora_0_3(train_size,test_size,train_path,test_path)
    elif dataset == 'citeseer':
        if perturbation == 0.1:
            load_citeseer_0_1(train_size,test_size,train_path,test_path)
        elif perturbation == 0.2:
            load_citeseer_0_2(train_size,test_size,train_path,test_path)
        elif perturbation == 0.3:
            load_citeseer_0_3(train_size,test_size,train_path,test_path)
    elif dataset == 'pubmed':
        if perturbation == 0.1:
            load_pubmed_0_1(train_size,test_size,train_path,test_path)
        elif perturbation == 0.2:
            load_pubmed_0_2(train_size,test_size,train_path,test_path)
        elif perturbation == 0.3:
            load_pubmed_0_3(train_size,test_size,train_path,test_path)
    elif dataset == 'deezer':
        if perturbation == 0.1:
            load_deezer_0_1(train_size,test_size,train_path,test_path)
        elif perturbation == 0.2:
            load_deezer_0_2(train_size,test_size,train_path,test_path)
        elif perturbation == 0.3:
            load_deezer_0_3(train_size,test_size,train_path,test_path)
    elif dataset == 'facebook':
        if perturbation == 0.1:
            load_facebook_0_1(train_size,test_size,train_path,test_path)
        elif perturbation == 0.2:
            load_facebook_0_2(train_size,test_size,train_path,test_path)
        elif perturbation == 0.3:
            load_facebook_0_3(train_size,test_size,train_path,test_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cora')  # cora/citeseer/pubmed/deezer/facebook
    parser.add_argument('--perturbation', type=float, default=0.1)  # 0.1/0.2/0.3
    parser.add_argument('--train_size', type=int,default=60)
    parser.add_argument('--test_size', type=int, default=20)
    parser.add_argument('--mask_rate',type=float,default=0.7)
    args = parser.parse_args()
    train_data_path = './dataset/' + args.dataset_name + '/for_train_' + args.dataset_name + '/' + str(args.perturbation) + '/'
    test_data_path = './dataset/' + args.dataset_name + '/for_test_' + args.dataset_name + '/' + str(args.perturbation) + '/'
    preprocess_data(args.dataset_name,train_data_path,test_data_path,args.perturbation,args.train_size,args.test_size)
    train_query_features = torch.load(train_data_path + 'query_features.pt')
    test_query_features = torch.load(test_data_path + 'query_features.pt')
    # mask feature
    process_to_masked_features(train_data_path,train_query_features,args.mask_rate)
    process_to_masked_features(test_data_path,test_query_features,args.mask_rate)
    # cat structural information
    # for train
    mask_query_features_filename = 'masked_' + str(args.mask_rate) + '_query_features.pt'
    train_target_features = torch.load(train_data_path + 'target_features.pt')
    train_target_adj = torch.load(train_data_path + 'target_adj.pt')
    train_query_features = torch.load(train_data_path + mask_query_features_filename)
    train_query_adj = torch.load(train_data_path + 'query_adj.pt')
    cat_structural_information(train_target_features,train_target_adj,train_query_features,train_query_adj,train_data_path)
    # for test
    mask_query_features_filename = 'masked_' + str(args.mask_rate) + '_query_features.pt'
    test_target_features = torch.load(test_data_path + 'target_features.pt')
    test_target_adj = torch.load(test_data_path + 'target_adj.pt')
    test_query_features = torch.load(test_data_path + mask_query_features_filename)
    test_query_adj = torch.load(test_data_path + 'query_adj.pt')
    cat_structural_information(test_target_features, test_target_adj, test_query_features, test_query_adj,test_data_path)

if __name__ == '__main__':
    main()
    print('Preprocessing completed !')




