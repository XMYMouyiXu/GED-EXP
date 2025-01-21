import os
import pickle
import random
import time
import copy
import json
from tqdm import tqdm
import os.path as osp
from argparse import ArgumentParser
from utils.utils import get_config

# import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch_geometric as tg
from torch_geometric.data import Data
# from tqdm.auto import tqdm
from torch_geometric.data import DataLoader, Batch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from scipy.stats import spearmanr, kendalltau
import torch.nn.functional as F

from model.GSC import GSC

CUDA_INDEX = 0
DATASET = "pyg_IMDB"
NUM = 2
dataset_root = "./"
ged_flag = True

if DATASET == "AIDS700nef":
    LABELS = 29
    DEGREE = -1
    root = dataset_root + "datasets/AIDS700nef/"
elif DATASET == "LINUX":
    LABELS = 8
    DEGREE = 7
    root = dataset_root + "datasets/LINUX/"
elif DATASET == "IMDBMulti" or DATASET == "pyg_IMDB":
    LABELS = 89
    DEGREE = 88
    root = dataset_root + "datasets/IMDBMulti/"
    if DATASET == "pyg_IMDB":
        ged_flag = False
        DATASET = "IMDBMulti"
else:
    raise TypeError("Invalid Model")


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


# args = get_hyper("./hyperparameters.json")




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
    if DEGREE > 0:
        one_hot_degree = OneHotDegree(DEGREE, cat=False)
        graph = one_hot_degree(graph)     
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

def get_sim_score(training_graph, testing_graph, num_train_graphs, num_test_graphs, filename):
    print("Getting GED Values...")
    num_graphs = num_train_graphs + num_test_graphs
    ged = [[0.0] * num_graphs for _ in range(num_graphs)]
    with open(filename + ("ged" if ged_flag else "ged_pyg"), "r") as f:
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
    
    return ged

num_train_graphs = 0
train_set, num_train_set = get_dataset(True)
test_set, num_test_set = get_dataset(False)
gt_ged = get_sim_score(train_set, test_set, 
                       num_train_set, num_test_set,
                       root)
gt_ged = torch.tensor(gt_ged)

path = "./model_saved/{}/".format(DATASET)
temp = []
for names in os.listdir(path):
    if names.startswith("2025"):
        temp.append(names)

folder_name = sorted(temp)[NUM]
print(folder_name)

parser = ArgumentParser()
parser.add_argument('--num_workers',       type = int,            default = 8,                  choices=[0,8])
parser.add_argument('--seed',              type = int,            default = 1234,               choices=[0, 1, 1234])
parser.add_argument('--data_dir',          type = str,            default = 'datasets/GED/') 
parser.add_argument('--custom_data_dir',   type = str,            default = 'datasets/GED/')
parser.add_argument('--hyper_file',        type = str,            default = 'config/')
parser.add_argument('--recache',         action = "store_true",      help = "clean up the old adj data", default=True)   
parser.add_argument('--no_dev',          action = "store_true" ,  default = False)
parser.add_argument('--patience',          type = int  ,          default = -1)
parser.add_argument('--gpu_id',            type = int  ,          default = 0)
parser.add_argument('--model',             type = str,            default ='GSC_GNN')  # GCN, GAT or other
parser.add_argument('--train_first',       type = bool,           default = True)
parser.add_argument('--save_model',        type = bool,           default = True)
parser.add_argument('--run_pretrain',    action ='store_true',    default = False)
parser.add_argument('--pretrain_path',     type = str,            default = 'model_saved/LINUX/2022-03-20_03-01-57')
args = parser.parse_args()

#config_path                  = osp.join(args.hyper_file, DATASET + '.yml') if not args.run_pretrain else osp.join(args.pretrain_path, 'config' + '.yml')
config_path                  = osp.join(path + folder_name + "/config.yml")
config                       = get_config(config_path)
model_name                   = args.model
config                       = config[model_name] 
config['model_name']         = model_name
config['dataset_name']       = DATASET
custom                       = config.get('custom', False)

model = GSC(config, LABELS).cuda()

print("\n\nModel evaluation.\n")



model.load_state_dict(torch.load(path + folder_name + "/GSC_GNN_{}_checkpoint.pth".format(DATASET), map_location="cpu"))
model.eval()

def test(test_type="traintest"):
    if test_type == "traintest":
        other_test_set = train_set
    elif test_type == "testtest":
        other_test_set = test_set

    print(test_type)

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
        if i == 0:
            print(g.x.shape)
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
            
            temp_data = {}
            temp_data["g1"] = source_batch.cuda()
            temp_data["g2"] = target_batch.cuda()

            # target = gt_ged[]
            # ground_truth[i].append(float(target))
            target_ged = gt_ged[g.i][g1.i]
            target_exp = torch.exp(-2 * torch.tensor(target_ged) / (g.num_nodes + g1.num_nodes))
            target_linear = target_ged / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 

        
            start_time = time.time()    
            pred_exp,_ = model(temp_data)
            # print(prediction)
            test_time += time.time() - start_time
            

            
            pred_exp = pred_exp[0].detach().cpu()
            if pred_exp < 0 or pred_exp > 1:
                continue
            
            #pred_exp = torch.exp(-2 * torch.tensor(prediction) / (g.num_nodes + g1.num_nodes))
            prediction = torch.round(-torch.log(pred_exp) * 0.5 * (g.num_nodes + g1.num_nodes))
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

test("traintest")
if DATASET != "pyg_IMDB":
    test("testtest")