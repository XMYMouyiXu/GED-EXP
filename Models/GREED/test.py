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

def get_sim_score(training_graph, testing_graph, num_train_graphs, num_test_graphs, filename, args):
    print("Getting GED Values...")
    num_graphs = num_train_graphs + num_test_graphs
    sim_score = [[0.0] * num_graphs for _ in range(num_graphs)]
    ged = [[0.0] * num_graphs for _ in range(num_graphs)]
    with open(filename + "ged", "r") as f:
        all_scores = f.readlines()

    num_n = []
    num_e = []
    for graph in training_graph + testing_graph:
        num_n.append(graph.num_nodes)
        num_e.append(graph.num_edges)
    
    x = 0
    for l in all_scores:
        y = 0
        l = l.strip()
        for s in l.split():
            s = torch.tensor(int(s))
            ged[x][y] = s
            ged[y][x] = s
            if args["normalization"] == "exp":
                s = torch.exp(-2 * s / (num_n[x] + num_n[y]))
            elif args["normalization"] == "linear":
                s = s / (max(num_n[x], num_n[y]) + max(num_e[x], num_e[y]) // 2) 
            elif args["normalization"] != "none":
                raise TypeError("Invalid normalization approach.")
            sim_score[x][y] = s
            sim_score[y][x] = s
            y += 1
        x += 1
        
    if os.path.exists(filename + "test_ged"):
        with open(filename + "test_ged", "r") as f:
            test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]

        for i in range(num_test_graphs):
            for j in range(num_test_graphs):
                g1 = testing_graph[i]
                g2 = testing_graph[j]
                ged[g1.i][g2.i] = test_scores[i][j]
                ged[g2.i][g1.i] = test_scores[i][j]

                if args["normalization"] == "exp":
                    s = torch.exp(-2 * torch.tensor(test_scores[i][j]) / (g1.num_nodes + g2.num_nodes))
                elif args["normalization"] == "linear":
                    s = test_scores[i][j] / (max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges) // 2) 
                elif args["normalization"] != "none":
                    raise TypeError("Invalid normalization approach.")
                
                sim_score[g1.i][g2.i] = s
                sim_score[g2.i][g1.i] = s
    
    return ged, sim_score

num_train_graphs = 0
num_features = CLASSES
train_set, num_train_set = get_dataset(True)
test_set, num_test_set = get_dataset(False)
gt_ged, _ = get_sim_score(train_set, test_set, 
                       num_train_set, num_test_set,
                       root, args)
gt_ged = torch.tensor(gt_ged)


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


model.load_state_dict(torch.load(path + folder_name + "/best_model.pt", map_location="cpu"))


if args["test_type"] == "traintest":
    other_test_set = train_set
elif args["test_type"] == "testtest":
    other_test_set = test_set

print(args["test_type"])

scores = []
scores_linear = []
mae = []
ground_truth = []
ground_truth_ged = []
prediction_mat = []
prediction_exp = []
rho_list = []
tau_list = []
prec_at_10_list = [] 
prec_at_20_list = []
test_time = 0

t = tqdm(total=len(test_set)*len(other_test_set))


for i, g in enumerate(test_set):
    # scores.append([])
    # scores_linear.append([])
    # mae.append([])
    ground_truth.append([])
    ground_truth_ged.append([])
    prediction_mat.append([])
    prediction_exp.append([])
    for j, g1 in enumerate(other_test_set):
        if gt_ged[g.i][g1.i] < 0:
            continue
        # print(g["i"], g1["i"], int(self.ged_matrix[g["i"], g1["i"]]), int(self.ged_matrix[g1["i"], g["i"]]))
        source_batch = Batch.from_data_list([g])
        target_batch = Batch.from_data_list([g1])

        # target = gt_ged[]
        # ground_truth[i].append(float(target))
        target_ged = gt_ged[g.i][g1.i]
        target_exp = torch.exp(-2 * torch.tensor(target_ged) / (g.num_nodes + g1.num_nodes))
        target_linear = target_ged / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 

    
        start_time = time.time()    
        prediction = model(source_batch, target_batch)
        # print(prediction)
        test_time += time.time() - start_time
        
        if prediction == 0:
            continue
        
        prediction = prediction[0].detach().cpu()
        
        pred_exp = torch.exp(-2 * torch.tensor(prediction) / (g.num_nodes + g1.num_nodes))
        pred_linear = prediction / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 

        # print(prediction, pred_exp, pred_linear)
        # print(target_ged, target_exp, target_linear)
        # print()
        # print(prediction, pred_exp, pred_linear)

        # print(prediction)
        # self.testing_time += time.time() - start_time

        ground_truth_ged[i].append(int(target_ged))
        ground_truth[i].append(float(target_exp))
        prediction_mat[i].append(float(prediction))
        prediction_exp[i].append(float(pred_exp))
        
        
        scores.append(float(F.mse_loss(pred_exp, target_exp, reduction='none')))
        
        mae.append(float(torch.abs(torch.round(prediction) - target_ged)))
    
        scores_linear.append(float(F.mse_loss(pred_linear, target_linear, reduction='none')))

        #print(prediction, pre_ged, prediction_linear, g.num_nodes + g1.num_nodes, temp)

    # print(prediction_mat[i])
    if len(prediction_mat[i]) > 10:
        rho_list.append(calculate_ranking_correlation(spearmanr, np.array(prediction_mat[i]), np.array(ground_truth_ged[i])))
        tau_list.append(calculate_ranking_correlation(kendalltau, np.array(prediction_mat[i]), np.array(ground_truth_ged[i])))
    if len(prediction_mat[i]) > 20:
        prec_at_10_list.append(calculate_prec_at_k(10, np.array(prediction_exp[i]), np.array(ground_truth[i]), np.array(ground_truth_ged[i])))
    if len(prediction_mat[i]) > 30:
        prec_at_20_list.append(calculate_prec_at_k(20, np.array(prediction_exp[i]), np.array(ground_truth[i]), np.array(ground_truth_ged[i])))

    t.update(len(other_test_set))
    


rho = np.mean(rho_list).item()
tau = np.mean(tau_list).item()
prec_at_10 = np.mean(prec_at_10_list).item()
prec_at_20 = np.mean(prec_at_20_list).item()
model_error = np.mean(scores).item()
mae = np.mean(mae).item()
model_error_linear = np.mean(scores_linear).item()

print(model_error * 1000)
print(model_error_linear * 1000)
print(mae)
print(rho)
print(tau)
print(prec_at_10)
print(prec_at_20)
print(test_time)
