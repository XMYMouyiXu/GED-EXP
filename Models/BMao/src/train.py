import os
import copy
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from tqdm import tqdm, trange
import matplotlib as plt
from scipy.stats import spearmanr, kendalltau
import random
from src.utils import get_hyper, calculate_ranking_correlation, calculate_prec_at_k



class Trainer():

    def __init__(self, args, filename_best_model):
        random.seed(1)
        self.args = args
        self.filename = args["graph_folder"] + args["dataset"] + "/"
        self.num_graphs = 0
        self.graphs = []
        self.split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))
        self.num_features = 1
        self.training_graph, self.num_train_graphs = self.get_dataset(True)
        self.testing_graph, self.num_test_graphs = self.get_dataset(False)
        self.sim_score_exp, self.sim_score_linear, self.ged_matrix = self.get_sim_score()
        self.train_test_time = -1
        self.test_test_time = -1

    def read_file(self, filename):
        fea = []
        edge_index = [[], []]
        with open(filename, "r") as f:
            n, m, features = self.split_line(f.readline())
            self.num_features = features
            fea = [[] for _ in range(n)]
            for _ in range(n):
                x, y = self.split_line(f.readline())
                temp_y = [0.0] * self.num_features
                if self.num_features == 1:
                    temp_y = [1.0]
                else:
                    temp_y[y] = 1.0
                fea[x] = temp_y
            for _ in range(m):
                x, y = self.split_line(f.readline())
                edge_index[0].append(x)
                edge_index[1].append(y)
        fea = torch.Tensor(fea)
        edge_index = torch.LongTensor(edge_index)
        graph = Data(x=fea, edge_index=edge_index, num_nodes=n)

        return graph

    def get_dataset(self, train):
        print("Reading {} graphs...".format("train" if train else "test"))
        folder_name = "train/" if train else "test/"
        num_graphs = 0 if train else self.num_train_graphs
        graphs = []
        for i in range(len(os.listdir(self.filename + folder_name))):
            filename = str(i)
            temp_graph = self.read_file(self.filename + folder_name + filename)
            temp_graph.i = num_graphs
            graphs.append(temp_graph)
            num_graphs += 1

        num_graphs -= 0 if train else self.num_train_graphs
        return graphs, num_graphs
    
    def get_sim_score(self):
        print("Getting GED Values...")
        num_graphs = self.num_train_graphs + self.num_test_graphs
        sim_score_exp = [[0.0] * num_graphs for _ in range(num_graphs)]
        sim_score_linear = [[0.0] * num_graphs for _ in range(num_graphs)]
        ged = [[0.0] * num_graphs for _ in range(num_graphs)]
        with open(self.filename + "ged", "r") as f:
            all_scores = f.readlines()

        num_n = []
        num_e = []
        for graph in self.training_graph + self.testing_graph:
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

                s_exp = torch.exp(-2 * s / (num_n[x] + num_n[y]))
                s_linear = s / (max(num_n[x], num_n[y]) + max(num_e[x], num_e[y]) // 2) 

                sim_score_exp[x][y] = s_exp
                sim_score_exp[y][x] = s_exp
                sim_score_linear[x][y] = s_linear
                sim_score_linear[y][x] = s_linear
                
                y += 1
            x += 1

        if os.path.exists(self.filename + "test_ged"):
            with open(self.filename + "test_ged", "r") as f:
                test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]

            for i in range(self.num_test_graphs):
                for j in range(self.num_test_graphs):
                    g1 = self.testing_graph[i]
                    g2 = self.testing_graph[j]
                    ged[g1.i][g2.i] = test_scores[i][j]
                    ged[g2.i][g1.i] = test_scores[i][j]

                    s_exp = torch.exp(-2 * torch.tensor(test_scores[i][j]) / (g1.num_nodes + g2.num_nodes))
                    s_linear = test_scores[i][j] / (max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges) // 2) 
                    
                    sim_score_exp[g1.i][g2.i] = s_exp
                    sim_score_exp[g2.i][g1.i] = s_exp
                    sim_score_linear[g1.i][g2.i] = s_linear
                    sim_score_linear[g2.i][g1.i] = s_linear

        return torch.Tensor(sim_score_exp), torch.Tensor(sim_score_linear), torch.Tensor(ged)

    def write_graph(self, f, graph):
        f.write("t # {}\n".format(graph.i))
        for i in range(graph.num_nodes):
            if graph.x == None:
                feature = 1
            else:
                feature = np.argmax(graph.x[i])
            f.write("v {} {}\n".format(i, feature))
        
        for i in range(graph.num_edges):
            e1, e2 = graph.edge_index[0][i], graph.edge_index[1][i]
            if e1 > e2:
                continue
            f.write("e {} {} {}\n".format(e1, e2, 1))

    def test_BMao(self, type="traintest"):
        print("Testing Type: {}...".format(type))
        model_path = "./"
        source_graph_path = model_path + "source_graph.txt"
        target_graph_path = model_path + "target_graph.txt"
        result_path = model_path + "result_" + self.args["dataset"]
        num_all = self.num_train_graphs
        # print(num_all)
        if type == "traintest":
            num_graph_sets = self.num_train_graphs
            graph_sets = self.training_graph
        elif type == "testtest":
            num_graph_sets = self.num_test_graphs
            graph_sets = self.testing_graph
        else:
            raise TypeError("Invalid Test Type.")

        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        scores_exp = []
        scores_linear = []
        mae = []

        print("Creating Files...")
        source_graph_f = open(source_graph_path, "w")
        target_graph_f = open(target_graph_path, "w")

        for graph1 in self.testing_graph:
            self.write_graph(source_graph_f, graph1)
        
        for graph2 in graph_sets:
            self.write_graph(target_graph_f, graph2)

        source_graph_f.close()
        target_graph_f.close()

        start_time = time.time()
        print("Running BMao...")
        os.system("{}ged -d {} -q {} -m search -p astar -l BMao -g -k {} > {}result_{}".format(model_path,
                                                                                               source_graph_path, 
                                                                                               target_graph_path,
                                                                                               self.args["BMao_k"],
                                                                                               model_path,
                                                                                               self.args["dataset"]))
        
        end_time = time.time()
        
        print("Analyzing results...")
        result_f = open(result_path, "r")
        temp = result_f.readline()
        
        all_train_val_graphs = sorted(graph_sets, key=lambda x: x.i)
        all_test_graphs = sorted(self.testing_graph, key=lambda x: x.i)

        prediction_ged_list = [[] for _ in range(self.num_test_graphs)]
        prediction_list_exp = [[] for _ in range(self.num_test_graphs)]
        prediction_list_linear = [[] for _ in range(self.num_test_graphs)]
        target_list_exp = [[] for _ in range(self.num_test_graphs)]
        target_list_linear = [[] for _ in range(self.num_test_graphs)]
        target_ged_list = [[] for _ in range(self.num_test_graphs)]

        while temp:
            temp = temp.strip()
            temp = temp.split(" (")[1]
            g1, temp = temp.split(", ")
            g2, current_ged = temp.split("): ")
            g1, g2, current_ged = int(g1), int(g2), int(current_ged)

            gt_sim_score_exp = self.sim_score_exp[g1][g2]
            gt_sim_score_linear = self.sim_score_linear[g1][g2]
            gt_ged = self.ged_matrix[g1][g2]

            # print(g1,g2, current_ged, gt_ged)

            if self.ged_matrix[g1][g2] < 0:
                temp = result_f.readline()
                continue

            if type == "testtest":
                g2 -= num_all
                g1 -= num_all
            else:
                if g1 < g2:
                    g1, g2 = g2, g1
                
                g1 -= num_all

            # print(g1, len(prediction_ged_list[g1]), len(target_ged_list[g1]))

            pred_exp = torch.exp(-2 * torch.tensor(current_ged) / (all_test_graphs[g1].num_nodes 
                                                                + all_train_val_graphs[g2].num_nodes))
            pred_linear = current_ged / (max(all_test_graphs[g1].num_nodes, all_train_val_graphs[g2].num_nodes) 
                                         + max(all_test_graphs[g1].num_edges, all_train_val_graphs[g2].num_edges) // 2) 
            
            prediction_list_exp[g1].append(pred_exp)
            prediction_list_linear[g1].append(pred_linear)
            prediction_ged_list[g1].append(current_ged)
            target_list_exp[g1].append(gt_sim_score_exp)
            target_list_linear[g1].append(gt_sim_score_linear)
            target_ged_list[g1].append(gt_ged)

            temp = result_f.readline()
        result_f.close()
        
        # print(prediction_ged_list)
        # print(prediction_list)
        # print()

        for i in range(self.num_test_graphs):
            for j in range(len(prediction_ged_list[i])):
                # if prediction_ged_list[i][j] > target_ged_list[i][j]:
                #     # print(i, j)
                #     # print(prediction_ged_list[i][j], target_ged_list[i][j])
                mae.append(abs(prediction_ged_list[i][j] - target_ged_list[i][j]))
            scores_exp.append(F.mse_loss(torch.Tensor(prediction_list_exp[i]), torch.Tensor(target_list_exp[i]), reduction='mean').detach().numpy())
            scores_linear.append(F.mse_loss(torch.Tensor(prediction_list_linear[i]), torch.Tensor(target_list_linear[i]), reduction='mean').detach().numpy())
            # print(prediction_ged_list[i])
            # print(target_ged_list[i])
            
            
        # print(scores)

        # prediction_list = prediction_list.detach().numpy()
        # target_list = target_list.detach().numpy()
        # target_ged_list = target_ged_list.detach().numpy()

        for i in range(self.num_test_graphs):
            if len(prediction_list_exp[i]) > 10:
                rho_list.append(calculate_ranking_correlation(spearmanr, np.array(prediction_list_exp[i]), np.array(target_list_exp[i])))
                tau_list.append(calculate_ranking_correlation(kendalltau, np.array(prediction_list_exp[i]), np.array(target_list_exp[i])))
            if len(prediction_list_exp[i]) > 20:
                prec_at_10_list.append(calculate_prec_at_k(10, np.array(prediction_list_exp[i]), np.array(target_list_exp[i]), np.array(target_ged_list[i])))
            if len(prediction_list_exp[i]) > 30:
                prec_at_20_list.append(calculate_prec_at_k(20, np.array(prediction_list_exp[i]), np.array(target_list_exp[i]), np.array(target_ged_list[i])))

        if type == "traintest":
            self.train_rho = np.mean(rho_list).item()
            self.train_tau = np.mean(tau_list).item()
            self.train_prec_at_10 = np.mean(prec_at_10_list).item()
            self.train_prec_at_20 = np.mean(prec_at_20_list).item()
            self.train_mse_exp = np.mean(scores_exp).item()
            self.train_mse_linear = np.mean(scores_linear).item()
            self.train_model_mae = np.mean(mae).item()
            self.train_test_time = end_time - start_time
        else:
            self.test_rho = np.mean(rho_list).item()
            self.test_tau = np.mean(tau_list).item()
            self.test_prec_at_10 = np.mean(prec_at_10_list).item()
            self.test_prec_at_20 = np.mean(prec_at_20_list).item()
            self.test_mse_exp = np.mean(scores_exp).item()
            self.test_mse_linear = np.mean(scores_linear).item()
            self.test_model_mae = np.mean(mae).item()
            self.test_test_time = end_time - start_time

    def test(self):
        self.test_BMao("traintest")
        self.test_BMao("testtest")
            
   