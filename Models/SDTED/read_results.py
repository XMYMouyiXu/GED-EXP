import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.stats import spearmanr, kendalltau
import torch.nn.functional as F

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

def get_sim_score(training_graph, testing_graph, num_train_graphs, num_test_graphs, filename):
    print("Getting GED Values...")
    num_graphs = num_train_graphs + num_test_graphs
    # sim_score_exp = [[0.0] * num_graphs for _ in range(num_graphs)]
    # sim_score_linear = [[0.0] * num_graphs for _ in range(num_graphs)]
    ged = [[0.0] * num_graphs for _ in range(num_graphs)]
    with open(filename + "ged_pyg", "r") as f:
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
            # s_exp = torch.exp(-2 * s / (num_n[x] + num_n[y]))
            # s_linear = s / (max(num_n[x], num_n[y]) + max(num_e[x], num_e[y]) // 2) 

            # sim_score_exp[x][y] = s_exp
            # sim_score_exp[y][x] = s_exp
            
            # sim_score_linear[x][y] = s_linear
            # sim_score_linear[y][x] = s_linear
            
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
                
                # s = test_scores[i][j]

                # s_exp = torch.exp(-2 * s / (num_n[x] + num_n[y]))
                # s_linear = s / (max(num_n[x], num_n[y]) + max(num_e[x], num_e[y]) // 2) 

                # sim_score_exp[g1.i][g2.i] = s_exp
                # sim_score_exp[g2.i][g1.i] = s_exp
                
                # sim_score_linear[g1.i][g2.i] = s_linear
                # sim_score_linear[g2.i][g1.i] = s_linear
                
    
    return ged #, sim_score_exp, sim_score_linear


def test(train_set, test_set, test_type, gt_ged, pred):
    
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

    if test_type == "traintest":
        other_test_set = train_set
    elif test_type == "testtest":
        other_test_set = test_set

    print(test_type)

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
            
            target_ged = gt_ged[g.i][g1.i]
            target_exp = torch.exp(-2 * torch.tensor(target_ged) / (g.num_nodes + g1.num_nodes))
            target_linear = target_ged / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 

            prediction = pred[g.i][g1.i]
            
            # if prediction == 0:
            #     continue
            
            
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

datasets = [
    # "AIDS700nef",
    # "LINUX",
    "IMDBMulti"
]

split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))

for dataset in datasets:
    print(dataset)
    root = "datasets/{}/".format(dataset)
    num_train_graphs = 0
    train_set, num_train_set = get_dataset(True)
    test_set, num_test_set = get_dataset(False)
    gt_ged = get_sim_score(train_set, test_set, 
                                                num_train_set, num_test_set,
                                                root)
    gt_ged = torch.tensor(gt_ged)
    # exp_ged = torch.tensor(exp_ged)
    # linear_ged = torch.tensor(linear_ged)

    origin_data_folder = "./datasets/{}/".format(dataset)
    # target_data_foler = "./data/TUDatasets/{}/".format(dataset)
    
    # if not os.path.exists(target_data_foler):
    #     os.system("mkdir {}".format(target_data_foler))
        
    num_graphs = num_train_set + num_test_set
    
    pred = [[0.0] * num_graphs for _ in range(num_graphs)]
    with open("my_results/{}.txt".format(dataset), "r") as f:
        for temp in f.readlines():
            x, y, value = map(float, temp.strip().split())
            x = int(x)
            y = int(y)
            pred[x][y] = value
            pred[y][x] = value
        

    # print(list(pred))
    # print(list(gt_ged))    

    pred = torch.tensor(pred)
    
    # print(pred)
    # print(gt_ged)
    
    
    test(train_set, test_set, "traintest", gt_ged, pred)
    test(train_set, test_set, "testtest", gt_ged, pred)
    



        
    