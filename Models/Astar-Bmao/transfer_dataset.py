import os
import pickle
import random
import time
import copy
import json
from tqdm import tqdm

# import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch_geometric as tg
from torch_geometric.data import Data
# from tqdm.auto import tqdm
from torch_geometric.data import DataLoader, Batch
from scipy.stats import spearmanr, kendalltau
import torch.nn.functional as F


from neuro import config, datasets, metrics, models, train, utils, viz
import pyged

from importlib import reload
reload(config)
reload(datasets)
reload(metrics)
reload(models)
reload(pyged)
reload(train)
reload(utils)
reload(viz)

def get_hyper(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    return data

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """

    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation
    
def _calculate_prec_at_k(k, target):
    target_increase = np.sort(target)
    target_value_sel = (target_increase <= target_increase[k-1]).sum()
    if target_value_sel > k:
        best_k_target = target.argsort()[:target_value_sel]
    else:
        best_k_target = target.argsort()[:k]
    return best_k_target


def calculate_prec_at_k(k, prediction, target, target_ged):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = _calculate_prec_at_k(k, -target)
    best_k_target_ged = _calculate_prec_at_k(k, target_ged)

    
    return len(set(best_k_pred).intersection(set(best_k_target_ged))) / k


args = get_hyper("./hyperparameters.json")

CUDA_INDEX = 0
NAME = args["dataset"]

if NAME == "GED_AIDS700nef":
    CLASSES = 29
    root = "datasets/AIDS700nef/"
elif NAME == "GED_LINUX":
    CLASSES = 1
    root = "datasets/LINUX/"
elif NAME == "GED_IMDBMulti":
    CLASSES = 1
    root = "datasets/IMDBMulti/"
else:
    raise TypeError("Invalid Dataset")


# root = "datasets/AIDS700nef/"
# val_ratio = 0.1

torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))


def read_file(filename):
    fea = []
    edge_index = [[], []]
    with open(filename, "r") as f:
        n, m, features = split_line(f.readline())
        num_features = features
        fea = [[] for _ in range(n)]
        for _ in range(n):
            x, y = split_line(f.readline())
            temp_y = [0.0] * num_features
            if num_features == 1:
                temp_y = [1.0]
            else:
                temp_y[y] = 1.0
            fea[x] = temp_y
        for _ in range(m):
            x, y = split_line(f.readline())
            edge_index[0].append(x)
            edge_index[1].append(y)
    fea = torch.Tensor(fea)
    edge_index = torch.LongTensor(edge_index)
    graph = Data(x=fea, edge_index=edge_index, num_nodes=n)

    return graph

def get_dataset(train):
    print("Reading {} graphs...".format("train" if train else "test"))
    folder_name = "train/" if train else "test/"
    num_graphs = 0 if train else num_train_set
    graphs = []
    for i in range(len(os.listdir(root + folder_name))):
        filename = str(i)
        temp_graph = read_file(root + folder_name + filename)
        temp_graph.i = num_graphs
        graphs.append(temp_graph)
        num_graphs += 1

    num_graphs -= 0 if train else num_train_set
    return graphs, num_graphs

# def split_train_val():
#     all = copy.deepcopy(train_set)
#     random.shuffle(all)
#     training_graph = all[:int(num_train_graphs * (1 - val_ratio))]
#     val_graph = all[int(num_train_graphs * (1 - val_ratio)):]
#     # self.training_graph = all[:self.num_train_graphs - self.num_test_graphs]
#     # self.val_graph = all[self.num_train_graphs - self.num_test_graphs:]
#     num_train_set = len(training_graph)
#     num_val_set = len(val_graph)
#     return training_graph, val_graph, num_train_set, num_val_set



num_train_graphs = 0
num_features = CLASSES
train_set, num_train_set = get_dataset(True)
test_set, num_test_set = get_dataset(False)


if args["model"] == "origin":
    model = models.NormGEDModel(8, CLASSES, 64, 64)

elif args["model"] == "Dual":
    model = models.DualNormGEDModel(8, CLASSES, 64, 64)
elif args["model"] == "NN":
    model = models.NeuralSiameseModel(8, CLASSES, 64, 64)

elif args["model"] == "Norm":
    model = models.NormGEDModel(8, CLASSES, 64, 64)
    model.weighted = True
else:
    raise TypeError("Invalid Model Type")


print("\n\nModel evaluation.\n")

path = "./results/{}-{}/".format(args["dataset"], args["model"])
folder_name = sorted(os.listdir(path))[args["num_model"]]


