import torch
import numpy as np
import random

train_path = './dataset/for_train_cora/0_3/'
test_path = './dataset/for_case_pubmed/'
pickle_protocol = 4
query_features = torch.load(test_path+'query_features.pt')

def process_to_masked_features(data_path,all_query_features,masked_range=0.5):
    masked_query_features = []
    for i in range(all_query_features.shape[0]):
        query_features = all_query_features[i]
        features_len = query_features.shape[1]
        print('before:',len(np.nonzero(query_features)[0]))
        # print('before',len(np.nonzero(query_features[0])[0]))
        for item in range(query_features.shape[0]):
            masked_indicator = np.ones((features_len))
            for _ in range(int(features_len * masked_range)):
                masked_index = random.randint(0, features_len - 1)
                masked_indicator[masked_index] = 0
            # masked features
            query_features[item] = query_features[item] * masked_indicator
        print('after:',len(np.nonzero(query_features)[0]))
        masked_query_features.append(query_features)
    fin_query_features = np.array(masked_query_features)
    torch.save(fin_query_features,data_path+'masked_'+str(masked_range)+'_query_features.pt',pickle_protocol=pickle_protocol)

process_to_masked_features(test_path,query_features,0.7)