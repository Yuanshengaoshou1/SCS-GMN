# Self-Supervised Similar Community Search based on Graph Matching Network

This repository contains the code and datesets used for the experiments described in the paper entitled with

_Self-Supervised Similar Community Search based on Graph Matching Network_

## Requirements

The experiments were conducted on 20 cores of a 2.3GHZ server (running Ubuntu Linux) with a Tesla M40 (24G memory).

## Datasets

Datasets: 

- [Cora](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html)

- [Citeseer](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html)

- [Pubmed](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz)

- [Deezer](http://snap.stanford.edu/data/feather-deezer-social.html)

- [Facebook](https://snap.stanford.edu/data/facebook-large-page-page-network.html)




## Usage

### Prerequisites

```
python==3.8.18
torch==1.13.1+cu116
torch-geometric==2.4.0 
torch-scatter==2.1.1+pt113cu116
torchvision==0.14.1+cu116 
scikit-learn==1.3.0 
numpy==1.22.3
networkx==3.1
dgl-cuda11.6==0.9.1  
```

Make sure that the environment is set up before the experiment.

### Effectiveness and Efficiency Evaluation

For our solution, We tested the effectiveness and efficiency of SCS-GMN on all datasets.

#### Training Phase

Before community search, we need to train SCS-GMN. You can use the following script command to train SCS-GMN:

```
python train.py --dataset cora
```

#### Search Phase

After training SCS-GMN, you can perform similarity community search based on SCS-GMN using the following command. 

```
python test.py --dataset cora --model_path './model_save/GMN_for_cora/GMN_for_cora_200.pth'
```

The results of the search can be found in the 'record' folder for evaluation of **effectiveness** and **efficiency**.


### Ablation Analysis

To verify the effectiveness of the proposed structural self-supervised loss
and label-based self-supervised loss, we implement three versions of SCS-GMN using different loss, including both of them, 
only label-based self-supervised loss, and only structural self-supervised loss. 
We validated the effectiveness of different methods using the following command:

```
python train.py --dataset cora train_type both(both/only label/only structure,different training type determines the different loss term)
```

### Parameter Sensitivity
We varied community similarity between the query graph and the target graph 
in dataset to study the impact of community similarity on SCS-GMN's 
performance over all dataset.(need to first obtain query graphs and target graphs under different community similarity thresholds, and then train the model)

```
process_[graph name]_dataset_0_2.py(The graph name can be any dataset)
process_cora_dataset_0_2.py
python train.py --dataset cora
```

### Main Parameters Settings
|   **Parameter**   | **Type** |                           **Description**                            |
|:-----------------:|:--------:|:--------------------------------------------------------------------:| 
|    GCN_in_size    |   int    |           dimension of fetures in a graph embedding model            |
|   GCN_out_size    |   int    |  dimension of hidden layers in a graph embedding model(default: 256  |
|      da_size      |   int    |                    Number of target graph's nodes                    |
|    batch_size     |   int    |                    Number of batch size to train                     |
|     test_size     |   int    |                     Number of batch size to test                     |
|       epoch       |   int    |                The number of rounds of model training                |
| candidate_method  |   str    |          way to get candidate set for cross-graph matching           |
| trade_off_for_re1 |  float   |          balance coefficient of the density similarity loss          |
| trade_off_for_re2 |  float   |        balance coefficient of the cohesiveness similariy loss        |
| trade_off_for_re3 |  float   |           balance coefficient of the size similarity loss            |
|      dataset      |   str    |                             dataset type                             |


**The repository will be continuously updated**.
