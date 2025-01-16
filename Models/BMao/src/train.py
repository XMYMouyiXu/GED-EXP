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

from models.EGSC.model import EGSC
from models.EGSC.parser import parameter_parser as get_hyper_egsc

from models.GEDGNN.model import GedGNN
from models.GEDGNN.parser import parameter_parser as get_hyper_gedgnn
from models.GEDGNN.GedMatrix import fixed_mapping_loss

from models.TaGSim.model import TaGSim
from models.TaGSim.parser import parameter_parser as get_hyper_tagsim

from models.simGNN.model import SimGNN
from models.simGNN.parser import parameter_parser as get_hyper_simGNN

from models.GENN.model import GENN
from models.GENN.parser import parameter_parser as get_hyper_GENN

from models.GOTSim.model import GOTSim
from models.GOTSim.parser import parameter_parser as get_hyper_GOTSim

from models.GPN.model import GPN
from models.GPN.parser import parameter_parser as get_hyper_GPN

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
        self.sim_score, self.ged_matrix = self.get_sim_score()
        self.create_model()
        self.split_train_val()
        self.filename_best_model = filename_best_model
        self.batch_one = None
        
        if args["model"] == "GEDGNN":
            self.get_mapping_GEDGNN()

        self.train_pairs = self.generate_pairs(self.training_graph)
        print(len(self.train_pairs))
        self.val_pairs = self.generate_pairs(self.val_graph)

        self.total_training_time = -1
        self.total_validation_time = -1
        self.avg_train_time = -1
        self.avg_val_time = -1
        self.testing_time = -1
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
        sim_score = [[0.0] * num_graphs for _ in range(num_graphs)]
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
                if self.args["normalization"] == "exp":
                    s = torch.exp(-2 * s / (num_n[x] + num_n[y]))
                elif self.args["normalization"] == "linear":
                    s = s / (max(num_n[x], num_n[y]) + max(num_e[x], num_e[y]) // 2) 
                elif self.args["normalization"] != "none":
                    raise TypeError("Invalid normalization approach.")
                sim_score[x][y] = s
                sim_score[y][x] = s
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

                    if self.args["normalization"] == "exp":
                        s = torch.exp(-2 * torch.tensor(test_scores[i][j]) / (g1.num_nodes + g2.num_nodes))
                    elif self.args["normalization"] == "linear":
                        s = test_scores[i][j] / (max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges) // 2) 
                    elif self.args["normalization"] != "none":
                        raise TypeError("Invalid normalization approach.")
                    
                    sim_score[g1.i][g2.i] = s
                    sim_score[g2.i][g1.i] = s

        return torch.Tensor(sim_score), torch.Tensor(ged)

    def split_train_val(self):
        all = copy.deepcopy(self.training_graph)
        random.shuffle(all)
        self.training_graph = all[:int(self.num_train_graphs * (1 - self.args["val_ratio"]))]
        self.val_graph = all[int(self.num_train_graphs * (1 - self.args["val_ratio"])):]
        # self.training_graph = all[:self.num_train_graphs - self.num_test_graphs]
        # self.val_graph = all[self.num_train_graphs - self.num_test_graphs:]
        self.num_train_graphs = len(self.training_graph)
        self.num_val_graphs = len(self.val_graph)
        print(self.num_train_graphs)

    def generate_pairs(self, graphs):
        pairs = []
        for i in range(len(graphs)):
            for j in range(i, len(graphs)):
                # if self.args["model"] == "GEDGNN":
                # if graphs[i].num_nodes > 10 or graphs[j].num_nodes > 10:
                #     continue
                if self.ged_matrix[graphs[i].i][graphs[j].i] < 0:
                    continue
                # else:
                #     if graphs[i].num_nodes > 10 and graphs[j].num_nodes > 10:
                #         continue
                #     if self.sim_score[graphs[i].i][graphs[j].i] > self.args["threshold"]:
                #         continue
                pairs.append((graphs[i], graphs[j]))

        return pairs
    
    def create_model(self):
        print("Creating model...")
        if self.args["model"] in ["EGSC", "EGSC1"]:
            model_args = get_hyper_egsc()
            model = EGSC(model_args, self.num_features)
        elif self.args["model"] == "GEDGNN":
            model_args = get_hyper_gedgnn()
            print(self.num_features)
            model = GedGNN(model_args, self.num_features)
        elif self.args["model"] == "TaGSim":
            model_args = get_hyper_tagsim()
            model = TaGSim(model_args, self.num_features, self.args["device"])
        elif self.args["model"] == "simGNN":
            model_args = get_hyper_simGNN()
            model = SimGNN(model_args, self.num_features)
        elif self.args["model"] in ["GENN", "GENN-Astar"]:
            model_args = get_hyper_GENN()
            if self.args["model"] == "GENN-Astar":
                model_args.enable_astar = True
                self.args["batch_size"] = 1
            else:
                model_args.enable_astar = False
            model = GENN(model_args, self.num_features)
        elif self.args["model"] == "GOTSim":
            model_args = get_hyper_GOTSim()
            model = GOTSim(model_args, self.num_features, self.args["device"])
        elif self.args["model"] == "GPN":
            model_args = get_hyper_GPN()
            model = GPN(model_args, self.num_features)
        elif self.args["model"] in ["BMao", "VJ"]:
            return 
        else:
            raise TypeError("Invalid Model")

        model = model.to(self.args["device"])
        self.model = model
        self.model_args = model_args
    
    def get_mapping_GEDGNN(self):
        print("Getting mappings for GEDGNN...")
        file_path = "./datasets/{}/".format(self.args["dataset"])
        with open(file_path + "index_mapping", "r") as f:
            index_mapping = list(map(int, f.readline().split()))
        
        with open(file_path + "node_mapping", "r") as f:
            temp_list = f.readlines()
            node_mapping = []
            for temp_line in temp_list:
                temp_mapping = list(map(int, temp_line.strip().split()))
                real_mapping = {}
                for i in range(len(temp_mapping)):
                    real_mapping[temp_mapping[i]] = i
                node_mapping.append(real_mapping)
                

        ged_dict = {}
        TaGED = json.load(open(file_path + "TaGED.json", 'r'))
        # confirm_ged_value = {}
        for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
            ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
            # confirm_ged_value[(id_1, id_2)] = ged_value
            ged_dict[(id_1, id_2)] = (ta_ged, mappings)
        
        self.ged_dict = ged_dict
        n = self.num_train_graphs + self.num_test_graphs + self.num_val_graphs
        mapping = [[None] * n for _ in range(n)]
        graphs = self.training_graph + self.testing_graph + self.val_graph
        for g1 in graphs:
            for g2 in graphs:
                id_pair = (index_mapping[g1.i], index_mapping[g2.i])
                if not id_pair in ged_dict:
                    continue
                _, gt_mappings = ged_dict[id_pair]
                node_index_1 = node_mapping[g1.i]
                node_index_2 = node_mapping[g2.i]
                mapping_list = [[0] * g2.num_nodes for _ in range(g1.num_nodes)]
                for gt_mapping in gt_mappings:
                    for x, y in enumerate(gt_mapping):
                        mapping_list[node_index_1[x]][node_index_2[y]] = 1
                mapping_matrix = torch.tensor(mapping_list).float().to(self.args["device"])
                mapping[g1.i][g2.i] = mapping[g2.i][g1.i] = mapping_matrix

                # if self.ged_matrix[g1.i][g2.i] != confirm_ged_value[id_pair]:
                #     print(g1.i, g2.i)
                #     print(self.ged_matrix[g1.i][g2.i], confirm_ged_value[id_pair])
                #     print()
        self.mapping = mapping

    def create_batches_pair(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        random.shuffle(self.train_pairs)

        source = []
        target = []

        for g1, g2 in self.train_pairs[:self.num_train_graphs]:
            source.append(g1)
            target.append(g2)

        source_loader = DataLoader(source, batch_size=self.args["batch_size"])
        target_loader = DataLoader(target, batch_size=self.args["batch_size"])
        
        return list(zip(source_loader, target_loader))
    
    def create_batches_one(self):
        random.shuffle(self.train_pairs)
        source = []
        target = []
        for g1, g2 in self.train_pairs:
            source.append(g1)
            target.append(g2)
        source_loader = DataLoader(source, batch_size=self.args["batch_size"])
        target_loader = DataLoader(target, batch_size=self.args["batch_size"])

        return list(zip(source_loader, target_loader))
    
    def create_batches(self):
        if self.args["batch_type"] == "pair":
            return self.create_batches_pair()
        elif self.args["batch_type"] == "oneone":
            return self.create_batches_one()
        else:
            raise TypeError("Invalid Batch Type.")

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0].to(self.args["device"])
        new_data["g2"] = data[1].to(self.args["device"])
        
        normalized_ged = self.sim_score[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()]
        new_data["target"] = normalized_ged.to(self.args["device"])
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()]#.tolist()
        new_data["target_ged"] = ged.to(self.args["device"])

        return new_data
   
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()

        if self.args["model"] in ["EGSC"]:
            data = self.transform(batch)
            target = data["target"]

            prediction = self.model(data)
            loss = F.mse_loss(prediction, target, reduction='sum') 
        elif self.args["model"] == "GENN":
            data = self.transform(batch)
            if not self.model.args.enable_astar:
                target = data["target"]
                prediction = self.model(data)
                loss = F.mse_loss(prediction, target, reduction='sum') 
            else:
                prediction, target = self.model(data)
                loss = F.mse_loss(prediction, target, reduction='mean')
        elif self.args["model"] in ["TaGSim", "simGNN", "GOTSim", "GPN", "EGSC1"]:
            loss = 0
            for graph_pair in batch:
                data = self.transform(graph_pair)
                prediction = self.model(data)
                loss += torch.nn.functional.mse_loss(data["target"], prediction)

        elif self.args["model"] == "GEDGNN":
            loss = torch.tensor([0]).float().to(self.args["device"])
            weight = self.model_args.loss_weight
            if self.args["dataset"] == "IMDBMulti":
                weight = 1.0
            for graph_pair in batch:
                data = self.transform(graph_pair)
                if data["g1"].num_nodes > data["g2"].num_nodes:
                    data["g1"], data["g2"] = data["g2"], data["g1"]
                id_pair = (data["g1"].i, data["g2"].i)
                gt_mapping = self.mapping[id_pair[0]][id_pair[1]]
                if gt_mapping is None:
                    continue
                target = data["target"]
                prediction, mapping = self.model(data)
                loss = loss + fixed_mapping_loss(mapping.cpu(), gt_mapping.cpu()) + weight * F.mse_loss(target, prediction)
        else:
            assert False
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_args.learning_rate, 
                                          weight_decay=self.model_args.weight_decay)
        self.model.train()

        epochs = trange(self.args["epochs"], leave=True, desc = "Epoch")
        loss_list = []
        loss_list_test = []
        best_loss = 100000000
        loss_count = 0

        self.total_training_time = 0
        self.total_validation_time = 0
        self.avg_train_time = 0
        self.avg_val_time = 0

        for epoch in epochs:
            self.model.train()
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            total_epoch_time = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches)):
                start_time = time.time()
                loss_score = self.process_batch(batch_pair)
                main_index += batch_pair[0].num_graphs
                loss_sum += loss_score
                total_epoch_time += time.time() - start_time

            self.total_training_time += total_epoch_time

            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
            loss_list.append(loss)

            #validation
            if len(self.val_pairs) == 0:
                self.save_model()
                continue

            if epoch % 10 != 0:
                continue
            #validation
            self.model.eval()

            validation_time = time.time()
            t = tqdm(total=len(self.val_pairs), desc="Validation")
            target_list = torch.zeros(len(self.val_pairs))
            prediction_list = torch.zeros(len(self.val_pairs))

            for i, graph_pair in enumerate(self.val_pairs):
                g, g1 = graph_pair
                
                if g1.num_nodes > g.num_nodes:
                    source_batch = Batch.from_data_list([g]) 
                    target_batch = Batch.from_data_list([g1])
                else:
                    source_batch = Batch.from_data_list([g1]) 
                    target_batch = Batch.from_data_list([g])
                
                data = self.transform((source_batch, target_batch))
                target_list[i] = data["target"]

                if self.args["model"] == "GEDGNN":
                    prediction, _ = self.model(data)
                else:
                    prediction = self.model(data)
                
                prediction_list[i] = prediction
                t.update(1)

            validation_time = time.time() - validation_time
            self.total_validation_time += validation_time

            avg_val_loss = F.mse_loss(prediction_list, target_list, reduction='mean').detach().numpy()

            t.set_description("Validation_loss: {}".format(avg_val_loss))
            t.close()

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.save_model()
                loss_count = 0
            else:
                loss_count += 1
            
            if loss_count > self.args["early_stopping"]:
                print("Early Stopping...")
                self.avg_train_time = self.total_training_time / (epoch + 1)
                self.avg_val_time = self.total_validation_time / (epoch // 10 + 1)
                break
                
        if self.avg_train_time == 0 or self.avg_val_time == 0:
            self.avg_train_time = self.total_training_time / (self.args["epochs"] + 1)
            self.avg_val_time = self.total_validation_time / (self.args["epochs"] // 10 + 1) 
            
    # def test_batch(self):
    #     """
    #     Scoring.
    #     """
    #     print("\n\nModel evaluation.\n")
        
    #     self.load_model()
    #     self.model.eval()

    #     scores = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     mae = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     ground_truth = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     ground_truth_ged = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     prediction_mat = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
        
    #     rho_list = []
    #     tau_list = []
    #     prec_at_10_list = [] 
    #     prec_at_20_list = []
        
    #     t = tqdm(total=self.num_test_graphs*len(self.training_graph + self.val_graph))

    #     n1n2 = []

    #     for g1 in self.testing_graph:
    #         n1n2.append([])
    #         for g2 in self.training_graph + self.val_graph:
    #             if self.args["normalization"] == "exp":
    #                 temp = g1.num_nodes + g2.num_nodes
    #             elif self.args["normalization"] == "linear":
    #                 temp = max(g1.num_nodes, g2.num_nodes) + (max(g1.num_edges, g2.num_edges) // 2)
    #             elif self.args["normalization"] != "none":
    #                 raise TypeError("Invalid Normalization")
    #             n1n2[-1].append(temp)

    #     n1n2 = torch.Tensor(n1n2)

    #     for i, g in enumerate(self.testing_graph):
    #         source_batch = Batch.from_data_list([g]*(self.num_train_graphs + self.num_val_graphs))
    #         target_batch = Batch.from_data_list(self.training_graph + self.val_graph)
            
    #         data = self.transform((source_batch, target_batch))
    #         target = data["target"].cpu()
    #         ground_truth[i] = target
    #         target_ged = data["target_ged"].cpu()
    #         ground_truth_ged[i] = target_ged

    #         prediction = self.model(data)
    #         prediction = prediction.cpu().detach()

    #         if self.args["normalization"] == "exp":
    #             ged_pred = -0.5 * torch.log(prediction) * n1n2[i]
    #         elif self.args["normalization"] == "linear":
    #             ged_pred = prediction * n1n2[i]
    #         elif self.args["normalization"] == "none":
    #             ged_pred = prediction
    #         else:
    #             raise TypeError("Invalid Normalization")

    #         prediction_mat[i] = prediction.numpy()
    #         scores[i] = F.mse_loss(prediction, target, reduction='none').cpu().detach().numpy()
    #         mae[i] = torch.abs(torch.round(ged_pred) - target_ged)

    #         rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
    #         tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
    #         prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))
    #         prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))

    #         t.update((self.num_train_graphs + self.num_val_graphs))

    #     self.rho = np.mean(rho_list).item()
    #     self.tau = np.mean(tau_list).item()
    #     self.prec_at_10 = np.mean(prec_at_10_list).item()
    #     self.prec_at_20 = np.mean(prec_at_20_list).item()
    #     self.model_error = np.mean(scores).item()
    #     self.model_mae = np.mean(mae).item()
    #     self.print_evaluation()
    
    def test_no_batch(self, type="traintest"):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        
        # print(self.filename_best_model)
        self.load_model()
        self.model.eval()
        
        test_pairs = []
        total_pairs = 0
        
        if type == "traintest":
            compared_graph_set = self.training_graph + self.val_graph
        elif type == "testtest":
            compared_graph_set = self.testing_graph
        else:
            raise TypeError("Invalid Test Type")
            
        for graph1 in self.testing_graph:
            test_pairs.append([])
            for graph2 in compared_graph_set:
                # if graph1.num_nodes > 10 and graph2.num_nodes > 10:
                #     continue
                if self.ged_matrix[graph1.i][graph2.i] < 0:
                    continue
                test_pairs[-1].append(graph2)
                total_pairs += 1

        t = tqdm(total=total_pairs)
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        scores = []
        mae = []
        
        # print(self.testing_graph[0].i)
        # print(self.training_graph[0].i)
        
        # g = self.testing_graph[0]
        
        # for graph in self.training_graph:
        #     if graph.i == 0:
        #         g1 = graph
        
        # print(g.i)
        # print(g1.i)
        # print(g.edge_index)
        # print(g1.edge_index)
        
        # source_batch = Batch.from_data_list([g]) 
        # target_batch = Batch.from_data_list([g1])
        # data = self.transform((source_batch, target_batch))
        
        # prediction, _ = self.model(data)
        
        # temp = max(g.num_nodes, g1.num_nodes) + (max(g.num_edges, g1.num_edges) // 2)
        # ged_pred = prediction.detach() * temp

        # print(prediction, ged_pred)
        
        # exit()
        

        if type == "traintest":
            self.testing_time = 0
        elif type == "testtest":
            self.test_test_time = 0
        else:
            raise TypeError("Invalid Test Type")
        
        
        for i, g in enumerate(self.testing_graph):
            pair_size = len(test_pairs[i])
            if pair_size == 0:
                continue

            target_list = torch.zeros((pair_size))
            ged_list = torch.zeros((pair_size))
            prediction_list = torch.zeros((pair_size))
            ged_pred_list = torch.zeros((pair_size))

            for j, g1 in enumerate(test_pairs[i]):
                if g1.num_nodes > g.num_nodes:
                    source_batch = Batch.from_data_list([g]) 
                    target_batch = Batch.from_data_list([g1])
                else:
                    source_batch = Batch.from_data_list([g1]) 
                    target_batch = Batch.from_data_list([g])
                data = self.transform((source_batch, target_batch))
                target_list[j] = data["target"]
                ged_list[j] = self.ged_matrix[g.i][g1.i]

                temp_start_time = time.time()
                if self.args["model"] == "GEDGNN":
                    prediction, _ = self.model(data)
                else:
                    prediction = self.model(data)
                    

                if type == "traintest":
                    self.testing_time += time.time() - temp_start_time
                elif type == "testtest":
                    self.test_test_time += time.time() - temp_start_time
                else:
                    raise TypeError("Invalid Test Type")

                if prediction <= 0:
                    print("wow")
                    ged_pred = g.num_nodes + g1.num_nodes + min(g.num_edges, g1.num_edges)

                    if self.args["normalization"] == "exp":
                        s = torch.exp(-2 * torch.tensor(ged_pred) / (g.num_nodes + g1.num_nodes))
                    elif self.args["normalization"] == "linear":
                        s = ged_pred / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 
                    elif self.args["normalization"] != "none":
                        raise TypeError("Invalid normalization approach.")
                    
                    prediction = torch.Tensor([s])

                else:
                    if self.args["normalization"] == "exp":
                        temp = g.num_nodes + g1.num_nodes
                        ged_pred = -0.5 * torch.log(prediction.detach()) * temp
                    elif self.args["normalization"] == "linear":
                        temp = max(g.num_nodes, g1.num_nodes) + (max(g.num_edges, g1.num_edges) // 2)
                        ged_pred = prediction.detach() * temp
                    elif self.args["normalization"] == "none":
                        ged_pred = prediction
                    else:
                        raise TypeError("Invalid Normalization")
                
                prediction_list[j] = prediction
                ged_pred_list[j] = ged_pred
                scores.append(((prediction - data["target"]) ** 2).cpu().detach().numpy())

                t.update(1)

        

            #scores.append(F.mse_loss(prediction_list, target_list, reduction='mean').detach().numpy())
            # print(ged_list)
            # print(torch.round(ged_pred_list))
            # print(list(torch.abs(ged_list - torch.round(ged_pred_list))))
            mae += list(torch.abs(ged_list - torch.round(ged_pred_list)))
            prediction_list = prediction_list.detach().numpy()
            target_list = target_list.detach().numpy()
            ged_list = ged_list.detach().numpy()
            ged_pred_list = ged_pred_list.detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_list, target_list))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_list, target_list))
            
            # print(prediction_list.argsort()[::-1][:10])
            # print(ged_pred_list.argsort()[:10])
            # print(ged_list.argsort()[:10])
            if self.args["precision_type"] == "raw":
                prec_at_10_list.append(calculate_prec_at_k(10, ged_pred_list, target_list, ged_list, self.args["normalization"], type="raw"))
                prec_at_20_list.append(calculate_prec_at_k(20, ged_pred_list, target_list, ged_list, self.args["normalization"], type="raw"))
            elif self.args["precision_type"] == "normalized":
                prec_at_10_list.append(calculate_prec_at_k(10, prediction_list, target_list, ged_list, self.args["normalization"]))
                prec_at_20_list.append(calculate_prec_at_k(20, prediction_list, target_list, ged_list, self.args["normalization"]))
            else:
                raise TypeError("Invalid Precision Type!")                

        # print(prec_at_10_list)
        if type == "traintest":
            self.rho = np.mean(rho_list).item()
            self.tau = np.mean(tau_list).item()
            self.prec_at_10 = np.mean(prec_at_10_list).item()
            self.prec_at_20 = np.mean(prec_at_20_list).item()
            self.model_error = np.mean(scores).item()
            self.model_mae = np.mean(mae).item()
        elif type == "testtest":
            self.test_rho = np.mean(rho_list).item()
            self.test_tau = np.mean(tau_list).item()
            self.test_prec_at_10 = np.mean(prec_at_10_list).item()
            self.test_prec_at_20 = np.mean(prec_at_20_list).item()
            self.test_model_error = np.mean(scores).item()
            self.test_model_mae = np.mean(mae).item()
        else:
            raise TypeError("Invalid Test Type")
        
        self.print_evaluation()
        
        
    # def test_test(self):
    #     """
    #     Scoring.
    #     """
    #     print("\n\nModel evaluation on test set only.\n")
        
    #     # self.model.load_state_dict(torch.load("models/GEDGNN/best_model/AIDS_20"))
    #     self.load_model()
    #     self.model.eval()
        
    #     # scores = np.empty((self.num_test_graphs, (self.num_test_graphs)))
    #     # mae = np.empty((self.num_test_graphs, (self.num_test_graphs)))
    #     # ground_truth = np.empty((self.num_test_graphs, (self.num_test_graphs)))
    #     # ground_truth_ged = np.empty((self.num_test_graphs, (self.num_test_graphs)))
    #     # prediction_mat = np.empty((self.num_test_graphs, (self.num_test_graphs)))
        
    #     # rho_list = []
    #     # tau_list = []
    #     # prec_at_10_list = [] 
    #     # prec_at_20_list = []
        
    #     # t = tqdm(total=self.num_test_graphs*(self.num_test_graphs))

    #     test_pairs = []
    #     total_pairs = 0
    #     for graph1 in self.testing_graph:
    #         test_pairs.append([])
    #         for graph2 in self.testing_graph:
    #             # if graph1.num_nodes > 10 and graph2.num_nodes > 10:
    #             #     continue
    #             if self.ged_matrix[graph1.i][graph2.i] < 0:
    #                 continue
    #             test_pairs[-1].append(graph2)
    #             total_pairs += 1

    #     t = tqdm(total=total_pairs)
    #     rho_list = []
    #     tau_list = []
    #     prec_at_10_list = [] 
    #     prec_at_20_list = []
    #     scores = []
    #     mae = []

    #     start_time = time.time()

    #     for i, g in enumerate(self.testing_graph):
    #         # target_list = torch.zeros((self.num_test_graphs))
    #         # ged_list = torch.zeros((self.num_test_graphs))
    #         # prediction_list = torch.zeros((self.num_test_graphs))
    #         # ged_pred_list = torch.zeros((self.num_test_graphs))
    #         pair_size = len(test_pairs[i])
    #         if pair_size == 0:
    #             continue

    #         target_list = torch.zeros((pair_size))
    #         ged_list = torch.zeros((pair_size))
    #         prediction_list = torch.zeros((pair_size))
    #         ged_pred_list = torch.zeros((pair_size))

    #         for j, g1 in enumerate(test_pairs[i]):
    #             # if g.num_nodes > 10 or g1.num_nodes > 10:
    #             #     continue
    #             # if self.sim_score[g.i][g1.i] < 0:
    #             #     continue

    #             source_batch = Batch.from_data_list([g])
    #             target_batch = Batch.from_data_list([g1])
                
    #             data = self.transform((source_batch, target_batch))
    #             target_list[j] = data["target"]
    #             ged_list[j] = self.ged_matrix[g.i][g1.i]

    #             if self.args["model"] == "GEDGNN":
    #                 prediction, _ = self.model(data)
    #             else:
    #                 prediction = self.model(data)

    #             if prediction <= 0:
    #                 ged_pred = g.num_nodes + g1.num_nodes + min(g.num_edges, g1.num_edges)

    #                 if self.args["normalization"] == "exp":
    #                     s = torch.exp(-2 * torch.tensor(ged_pred) / (g.num_nodes + g1.num_nodes))
    #                 elif self.args["normalization"] == "linear":
    #                     s = ged_pred / (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2) 
    #                 elif self.args["normalization"] != "none":
    #                     raise TypeError("Invalid normalization approach.")
                    
    #                 prediction = torch.Tensor([s])

    #             else:
    #                 if self.args["normalization"] == "exp":
    #                     temp = g.num_nodes + g1.num_nodes
    #                     ged_pred = -0.5 * torch.log(prediction.detach()) * temp
    #                 elif self.args["normalization"] == "linear":
    #                     temp = max(g.num_nodes, g1.num_nodes) + (max(g.num_edges, g1.num_edges) // 2)
    #                     ged_pred = prediction.detach() * temp
    #                 elif self.args["normalization"] == "none":
    #                     ged_pred = prediction
    #                 else:
    #                     raise TypeError("Invalid Normalization")
                
    #             prediction_list[j] = prediction
    #             ged_pred_list[j] = ged_pred
                
    #             t.update(1)

    #         # prediction_mat[i] = prediction_list.detach().numpy()
    #         # ground_truth[i] = target_list.detach().numpy()
    #         # ground_truth_ged[i] = ged_list.detach().numpy()

    #         # scores[i] = F.mse_loss(prediction_list, target_list, reduction='none').detach().numpy()
    #         # mae[i] = torch.abs(ged_list - torch.round(ged_pred_list))

    #         # rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
    #         # tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
    #         # prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))
    #         # prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))
                
    #         scores.append(F.mse_loss(prediction_list, target_list, reduction='mean').detach().numpy())
    #         mae += list(torch.abs(ged_list - torch.round(ged_pred_list)))
    #         prediction_list = prediction_list.detach().numpy()
    #         target_list = target_list.detach().numpy()
    #         ged_list = ged_list.detach().numpy()

    #         rho_list.append(calculate_ranking_correlation(spearmanr, prediction_list, target_list))
    #         tau_list.append(calculate_ranking_correlation(kendalltau, prediction_list, target_list))
    #         prec_at_10_list.append(calculate_prec_at_k(10, prediction_list, target_list, ged_list, self.args["normalization"]))
    #         prec_at_20_list.append(calculate_prec_at_k(20, prediction_list, target_list, ged_list, self.args["normalization"]))

    #         # print(ged_list)
    #         # print(ged_pred_list)

    #     self.test_test_time = time.time() - start_time

    #     self.test_rho = np.mean(rho_list).item()
    #     self.test_tau = np.mean(tau_list).item()
    #     self.test_prec_at_10 = np.mean(prec_at_10_list).item()
    #     self.test_prec_at_20 = np.mean(prec_at_20_list).item()
    #     self.test_model_error = np.mean(scores).item()
    #     self.test_model_mae = np.mean(mae).item()

    #     print("\nmse(10^-3): " + str(round(self.test_model_error*1000, 5)))
    #     print("mae: " + str(round(self.test_model_mae, 5)))
    #     print("Spearman's rho: " + str(round(self.test_rho, 5)))
    #     print("Kendall's tau: " + str(round(self.test_tau, 5)))
    #     print("p@10: " + str(round(self.test_prec_at_10, 5)))
    #     print("p@20: " + str(round(self.test_prec_at_20, 5)))

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
        model_path = "./models/BMao/"
        source_graph_path = model_path + "source_graph.txt"
        target_graph_path = model_path + "target_graph.txt"
        result_path = model_path + "result"
        num_all = self.num_train_graphs + self.num_val_graphs
        # print(num_all)
        if type == "traintest":
            num_graph_sets = self.num_train_graphs + self.num_val_graphs
            graph_sets = self.val_graph + self.training_graph
        elif type == "testtest":
            num_graph_sets = self.num_test_graphs
            graph_sets = self.testing_graph
        else:
            raise TypeError("Invalid Test Type.")
        
        # prediction_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        # prediction_ged_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        # target_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        # target_ged_list = torch.zeros((self.num_test_graphs, num_graph_sets))

        # scores = np.empty((self.num_test_graphs, (num_graph_sets)))
        # mae = np.empty((self.num_test_graphs, (num_graph_sets)))

        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        scores = []
        mae = []

        print("Creating Files...")
        source_graph_f = open(source_graph_path, "w")
        target_graph_f = open(target_graph_path, "w")

        for graph1 in self.testing_graph:
            self.write_graph(source_graph_f, graph1)
            # for graph2 in graph_sets:
            #     # if graph1.num_nodes > 10 and graph2.num_nodes > 10:
            #     #     continue
            #     # if self.sim_score[graph1.i][graph2.i] > self.args["threshold"]:
            #     #     continue
            #     if self.ged_matrix[graph1.i][graph2.i] < 0:
            #         print("test")
            #         continue
            #     total_pairs += 1
            #     # target_list[graph1.i - num_all][graph2.i if type == "traintest" else graph2.i - num_all] = self.sim_score[graph1.i][graph2.i]
            #     # target_ged_list[graph1.i - num_all][graph2.i if type == "traintest" else graph2.i - num_all] = self.ged_matrix[graph1.i][graph2.i]
        
        
        
        for graph2 in graph_sets:
            self.write_graph(target_graph_f, graph2)

        source_graph_f.close()
        target_graph_f.close()

        start_time = time.time()
        print("Running BMao...")
        os.system("{}ged -d {} -q {} -m search -p astar -l BMao -g -k {} > {}result".format(model_path,
                                                                                          source_graph_path, 
                                                                                          target_graph_path,
                                                                                          self.args["BMao_k"],
                                                                                          model_path))
        
        end_time = time.time()
        
        print("Analyzing results...")
        result_f = open(result_path, "r")
        temp = result_f.readline()
        
        all_train_val_graphs = sorted(graph_sets, key=lambda x: x.i)
        all_test_graphs = sorted(self.testing_graph, key=lambda x: x.i)

        prediction_ged_list = [[] for _ in range(self.num_test_graphs)]
        prediction_list = [[] for _ in range(self.num_test_graphs)]
        target_list = [[] for _ in range(self.num_test_graphs)]
        target_ged_list = [[] for _ in range(self.num_test_graphs)]

        while temp:
            temp = temp.strip()
            temp = temp.split(" (")[1]
            g1, temp = temp.split(", ")
            g2, current_ged = temp.split("): ")
            g1, g2, current_ged = int(g1), int(g2), int(current_ged)

            gt_sim_score = self.sim_score[g1][g2]
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

            if self.args["normalization"] == "exp":
                s = torch.exp(-2 * torch.tensor(current_ged) / (all_test_graphs[g1].num_nodes 
                                                                + all_train_val_graphs[g2].num_nodes))
            elif self.args["normalization"] == "linear":
                s = current_ged / (max(all_test_graphs[g1].num_nodes, all_train_val_graphs[g2].num_nodes) 
                                   + max(all_test_graphs[g1].num_edges, all_train_val_graphs[g2].num_edges) // 2) 
            elif self.args["normalization"] != "none":
                raise TypeError("Invalid normalization approach.")
            
            prediction_list[g1].append(s)
            prediction_ged_list[g1].append(current_ged)
            target_list[g1].append(gt_sim_score)
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
            scores.append(F.mse_loss(torch.Tensor(prediction_list[i]), torch.Tensor(target_list[i]), reduction='mean').detach().numpy())
            # print(prediction_ged_list[i])
            # print(target_ged_list[i])
            
            
        # print(scores)

        # prediction_list = prediction_list.detach().numpy()
        # target_list = target_list.detach().numpy()
        # target_ged_list = target_ged_list.detach().numpy()

        for i in range(self.num_test_graphs):
            if len(prediction_list[i]) > 10:
                rho_list.append(calculate_ranking_correlation(spearmanr, np.array(prediction_list[i]), np.array(target_list[i])))
                tau_list.append(calculate_ranking_correlation(kendalltau, np.array(prediction_list[i]), np.array(target_list[i])))
            if len(prediction_list[i]) > 20:
                prec_at_10_list.append(calculate_prec_at_k(10, np.array(prediction_list[i]), np.array(target_list[i]), np.array(target_ged_list[i]), self.args["normalization"]))
            if len(prediction_list[i]) > 30:
                prec_at_20_list.append(calculate_prec_at_k(20, np.array(prediction_list[i]), np.array(target_list[i]), np.array(target_ged_list[i]), self.args["normalization"]))

        if type == "traintest":
            self.rho = np.mean(rho_list).item()
            self.tau = np.mean(tau_list).item()
            self.prec_at_10 = np.mean(prec_at_10_list).item()
            self.prec_at_20 = np.mean(prec_at_20_list).item()
            self.model_error = np.mean(scores).item()
            self.model_mae = np.mean(mae).item()
            self.testing_time = end_time - start_time
        else:
            self.test_rho = np.mean(rho_list).item()
            self.test_tau = np.mean(tau_list).item()
            self.test_prec_at_10 = np.mean(prec_at_10_list).item()
            self.test_prec_at_20 = np.mean(prec_at_20_list).item()
            self.test_model_error = np.mean(scores).item()
            self.test_model_mae = np.mean(mae).item()
            self.test_test_time = end_time - start_time

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)))
        print("mae: " + str(round(self.model_mae, 5)))
        print("Spearman's rho: " + str(round(self.rho, 5)))
        print("Kendall's tau: " + str(round(self.tau, 5)))
        print("p@10: " + str(round(self.prec_at_10, 5)))
        print("p@20: " + str(round(self.prec_at_20, 5)))

    def save_model(self):
        torch.save(self.model.state_dict(), self.filename_best_model)
    
    def load_model(self):
        #self.model.load_state_dict(torch.load("models/GEDGNN/best_model/AIDS_20"))
        self.model.load_state_dict(torch.load(self.filename_best_model))

    def test(self):
        if self.args["model"] in ["EGSC", "GENN", "GENN-Astar"]:
            self.test_no_batch("traintest")
        # elif self.args["model"] == "GEDGNN":
        #     self.test_GEDGNN()
        elif self.args["model"] in ["TaGSim", "GEDGNN", "simGNN", "GOTSim", "GPN"]:
            self.test_no_batch(type="traintest")
        elif self.args["model"] in ["BMao"]:
            self.test_BMao("traintest")
            self.test_BMao("testtest")
            return
        elif self.args["model"] in ["VJ"]:
            self.test_VJ("traintest")
            # self.test_VJ("testtest")
            return
        else:
            raise TypeError("Invalid Model Name")
        
        if self.args["dataset"] in ["AIDS700nef", "LINUX"]:
            self.test_no_batch(type="testtest")
            
            
            
    def create_cost_matrix_vj(self, graph1, graph2):
        n = graph1.num_nodes + graph2.num_nodes
        cost_matrix = [[0] * n for _ in range(n)]
        
        label_check = [[False] * graph2.num_nodes for _ in range(graph1.num_nodes)]
        for i in range(graph1.num_nodes):
            for j in range(graph2.num_nodes):
                label_check[i][j] = list(graph1.x[i]) == list(graph2.x[j])
                
        for i in range(graph1.num_nodes):
            for j in range(graph2.num_nodes):
                cost_matrix[i][j] = 0 if label_check[i][j] else 1

        for i in range(graph1.num_nodes):
            for j in range(graph2.num_nodes, n):
                cost_matrix[i][j] = 1 if j - graph2.num_nodes == i else 100000000
        
        for i in range(graph1.num_nodes, n):
            for j in range(graph2.num_nodes):
                cost_matrix[i][j] = 1 if i - graph1.num_nodes == j else 100000000
                
        # new_cost_matrix = []
        # for i in range(n):
        #     new_cost_matrix.append(np.array(cost_matrix[i]))
        # new_cost_matrix = np.array(new_cost_matrix)
        return cost_matrix
            
    def test_VJ(self, type="traintest"):
        from lapjv import lapjv
        
        num_all = self.num_train_graphs + self.num_val_graphs
        if type == "traintest":
            num_graph_sets = self.num_train_graphs + self.num_val_graphs
            num_temp = 0
            graph_sets = self.val_graph + self.training_graph
        elif type == "testtest":
            num_graph_sets = self.num_test_graphs
            num_temp = num_all
            graph_sets = self.testing_graph
        else:
            raise TypeError("Invalid Test Type.")
        
        prediction_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        prediction_ged_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        target_list = torch.zeros((self.num_test_graphs, num_graph_sets))
        target_ged_list = torch.zeros((self.num_test_graphs, num_graph_sets))

        scores = []
        mae = []
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []   
        
        current_time = 0
        for i, graph1 in enumerate(self.testing_graph):
            for j, graph2 in enumerate(graph_sets):
                target_list[graph1.i - num_all][graph2.i - num_temp] = self.sim_score[graph1.i][graph2.i]
                target_ged_list[graph1.i - num_all][graph2.i - num_temp] = self.ged_matrix[graph1.i][graph2.i]
                
                start_time = time.time()
                cost_matrix = self.create_cost_matrix_vj(graph1, graph2)
                x_assign, y_assign, _ = lapjv(cost_matrix)
                current_time += time.time() - start_time
                
                predicted_ged = 0
                for x, y in zip(x_assign, y_assign):
                    predicted_ged += cost_matrix[x][y]
                prediction_ged_list[graph1.i - num_all][graph2.i - num_temp] = predicted_ged
                
                if self.args["normalization"] == "exp":
                    s = torch.exp(-2 * torch.tensor(predicted_ged) / (graph1.num_nodes + graph2.num_nodes))
                elif self.args["normalization"] == "linear":
                    s = predicted_ged / (max(graph1.num_nodes, graph2.num_nodes) 
                                    + max(graph1.num_edges, graph2.num_edges) // 2) 
                elif self.args["normalization"] != "none":
                    raise TypeError("Invalid normalization approach.")
                prediction_list[graph1.i - num_all][graph2.i - num_temp] = s
                
            scores.append(F.mse_loss(prediction_list[i], target_list[i], reduction='mean').detach().numpy())
            mae += list(torch.abs(target_ged_list[i] - torch.round(prediction_ged_list[i])))
            # prediction_list = prediction_list.detach().numpy()
            # target_list = target_list.detach().numpy()
            # ged_list = ged_list.detach().numpy()

            # rho_list.append(calculate_ranking_correlation(spearmanr, prediction_list, target_list))
            # tau_list.append(calculate_ranking_correlation(kendalltau, prediction_list, target_list))
            # prec_at_10_list.append(calculate_prec_at_k(10, prediction_list, target_list, target_ged_list, self.args["normalization"]))
            # prec_at_20_list.append(calculate_prec_at_k(20, prediction_list, target_list, target_ged_list, self.args["normalization"]))

        prediction_list = prediction_list.detach().numpy()
        target_list = target_list.detach().numpy()
        target_ged_list = target_ged_list.detach().numpy()

        for i in range(self.num_test_graphs):
            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_list[i], target_list[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_list[i], target_list[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_list[i], target_list[i], target_ged_list[i], self.args["normalization"]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_list[i], target_list[i], target_ged_list[i], self.args["normalization"]))


        if type == "traintest":
            self.rho = np.mean(rho_list).item()
            self.tau = np.mean(tau_list).item()
            self.prec_at_10 = np.mean(prec_at_10_list).item()
            self.prec_at_20 = np.mean(prec_at_20_list).item()
            self.model_error = np.mean(scores).item()
            self.model_mae = np.mean(mae).item()
            self.testing_time = current_time
        else:
            self.test_rho = np.mean(rho_list).item()
            self.test_tau = np.mean(tau_list).item()
            self.test_prec_at_10 = np.mean(prec_at_10_list).item()
            self.test_prec_at_20 = np.mean(prec_at_20_list).item()
            self.test_model_error = np.mean(scores).item()
            self.test_model_mae = np.mean(mae).item()
            self.test_test_time = current_time

    # def init_graph_pairs(self):
    #     random.seed(1)

    #     testing_graphs = []

    #     graphs = self.training_graph + self.testing_graph

    #     train_num = self.num_train_graphs - self.num_test_graphs
    #     val_num = self.num_train_graphs
    #     test_num = len(graphs)

    #     # if self.args.demo:
    #     #     train_num = 30
    #     #     val_num = 40
    #     #     test_num = 50
    #     #     self.args.epochs = 1

    #     # assert self.args.graph_pair_mode == "combine"
    #     # dg = self.delta_graphs
    #     # for i in range(train_num):
    #     #     if self.gn[i] <= 10:
    #     #         for j in range(i, train_num):
    #     #             tmp = self.check_pair(i, j)
    #     #             if tmp is not None:
    #     #                 self.training_graphs.append(tmp)
    #     #     elif dg[i] is not None:
    #     #         k = len(dg[i])
    #     #         for j in range(k):
    #     #             self.training_graphs.append((1, i, j))

    #     li = []
    #     for i in range(train_num):
    #         if self.training_graph[i].num_nodes <= 10:
    #             li.append(i)

    #     # for i in range(train_num, val_num):
    #     #     if self.gn[i] <= 10:
    #     #         random.shuffle(li)
    #     #         self.val_graphs.append((0, i, li[:self.args.num_testing_graphs]))
    #     #         #self.val_graphs.append((0, i, li))
    #     #     # elif dg[i] is not None:
    #     #     #     k = len(dg[i])
    #     #     #     self.val_graphs.append((1, i, list(range(k))))

    #     for i in range(self.num_test_graphs):
    #         if self.testing_graph[i].num_nodes <= 10:
    #             random.shuffle(li)
    #             testing_graphs.append((0, i, li[:100]))
    #             #self.testing_graphs.append((0, i, li))
    #         # elif dg[i] is not None:
    #         #     k = len(dg[i])
    #         #     self.testing_graphs.append((1, i, list(range(k))))

    #     # li = []
    #     # for i in range(val_num, test_num):
    #     #     if self.gn[i] <= 10:
    #     #         li.append(i)

    #     # for i in range(val_num, test_num):
    #     #     if self.gn[i] <= 10:
    #     #         random.shuffle(li)
    #     #         self.testing2_graphs.append((0, i, li[:self.args.num_testing_graphs]))
    #     #         #self.testing2_graphs.append((0, i, li))
    #     #     elif dg[i] is not None:
    #     #         k = len(dg[i])
    #     #         self.testing2_graphs.append((1, i, list(range(k))))

    #     # print("Generate {} training graph pairs.".format(len(self.training_graphs)))
    #     # print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), self.args.num_testing_graphs))
    #     # print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), self.args.num_testing_graphs))
    #     # print("Generate {} * {} testing2 graph pairs.".format(len(self.testing2_graphs), self.args.num_testing_graphs))
    #     return testing_graphs
    
    # def cal_pk(self, num, pre, gt):
    #     tmp = list(zip(gt, pre))
    #     tmp.sort()
    #     beta = []
    #     for i, p in enumerate(tmp):
    #         beta.append((p[1], p[0], i))
    #     beta.sort()
    #     ans = 0
    #     for i in range(num):
    #         if beta[i][2] < num:
    #             ans += 1
    #     return ans / num

    # def test_GEDGNN(self):
    #     """
    #     Scoring on the test set.
    #     """
    #     testing_graph_set = "test"
    #     test_k = 0
    #     self.load_model()
    #     #self.model.load_state_dict(torch.load("models/GEDGNN/best_model/AIDS_20"))
    #     print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
    #     if testing_graph_set == 'test':
    #         testing_graphs = self.init_graph_pairs()
    #     # elif testing_graph_set == 'test2':
    #     #     testing_graphs = self.testing2_graphs
    #     # elif testing_graph_set == 'val':
    #     #     testing_graphs = self.val_graphs
    #     else:
    #         assert False

    #     self.model.eval()

    #     # g1 = self.testing_graph[1]
    #     # g = self.training_graph[0]

    #     # g = nx.Graph()
    #     # g1 = nx.Graph()
    #     # g.add_nodes_from(list(range(10)))
    #     # g1.add_nodes_from(list(range(10)))
    #     # g = from_networkx(g)
    #     # g1 = from_networkx(g1)

    #     # g.edge_index = torch.LongTensor([[7, 3, 3, 3, 5, 6, 1, 0, 4, 3, 1, 8, 9, 2, 2, 0, 2, 2],
    #     # [3, 1, 8, 9, 2, 2, 0, 2, 2, 7, 3, 3, 3, 5, 6, 1, 0, 4]])
    #     # g1.edge_index = torch.LongTensor([[7, 7, 3, 3, 3, 5, 8, 6, 1, 0, 4, 9, 0, 5, 4, 8, 9, 4, 0, 2],
    #     # [4, 9, 0, 5, 4, 8, 9, 4, 0, 2, 7, 7, 3, 3, 3, 5, 8, 6, 1, 0]])
    #     # g.x = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g1.x = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g.i = 0
    #     # g1.i = 560

    #     # g = nx.Graph()
    #     # g1 = nx.Graph()
    #     # g.add_nodes_from(list(range(10)))
    #     # g1.add_nodes_from(list(range(6)))
    #     # g = from_networkx(g)
    #     # g1 = from_networkx(g1)

    #     # g.edge_index = torch.LongTensor([[7, 3, 3, 3, 5, 6, 1, 0, 4, 3, 1, 8, 9, 2, 2, 0, 2, 2],
    #     # [3, 1, 8, 9, 2, 2, 0, 2, 2, 7, 3, 3, 3, 5, 6, 1, 0, 4]])
    #     # g1.edge_index = torch.LongTensor([[3, 3, 5, 1, 1, 4, 5, 1, 4, 0, 2, 2],
    #     # [5, 1, 4, 0, 2, 2, 3, 3, 5, 1, 1, 4]])
    #     # g.x = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g1.x = torch.Tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g.i = 0
    #     # g1.i = 561


    #     # g = nx.Graph()
    #     # g1 = nx.Graph()
    #     # g.add_nodes_from(list(range(9)))
    #     # g1.add_nodes_from(list(range(9)))
    #     # g = from_networkx(g)
    #     # g1 = from_networkx(g1)

    #     # g.edge_index = torch.LongTensor([[7, 3, 5, 8, 6, 1, 1, 0, 6, 1, 1, 6, 2, 0, 4, 2],
    #     # [6, 1, 1, 6, 2, 0, 4, 2, 7, 3, 5, 8, 6, 1, 1, 0]])
    #     # g1.edge_index = torch.LongTensor([[7, 3, 5, 8, 6, 1, 1, 0, 2, 0, 1, 2, 1, 0, 4, 2],
    #     # [2, 0, 1, 2, 1, 0, 4, 2, 7, 3, 5, 8, 6, 1, 1, 0]])
    #     # g.x = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g1.x = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     #  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    #     # g.i = 1
    #     # g1.i = 562

    #     # source_batch = Batch.from_data_list([g])
    #     # target_batch = Batch.from_data_list([g1])
    #     # data = self.transform((source_batch, target_batch ))
    #     # print(data["g1"].i, data["g2"].i)
    #     # print(data["g1"].num_nodes, data["g2"].num_nodes)
    #     # print(data["g1"].num_edges, data["g2"].num_edges)
    #     # print(data["g1"].edge_index)
    #     # print(data["g2"].edge_index)
    #     # for key in data:
    #     #     print(key, data[key])
    #     # target, gt_ged = data["target"].cpu().item(), data["target_ged"].cpu()
    #     # # print(data["id_1"], data["id_2"], gt_ged)
    #     # model_out = self.model(data) # if test_k == 0 else self.test_matching(data, test_k)
    #     # prediction, _ = model_out[0].cpu(), model_out[1]
    #     # pre_ged = prediction.detach().cpu() * (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2)
    #     # round_pre_ged = round(pre_ged.cpu().item())
    #     # prediction = prediction.cpu()
    #     # print(prediction, round_pre_ged)
    #     # exit()

    #     num = 0  # total testing number
    #     time_usage = []
    #     mse = []  # score mse
    #     mae = []  # ged mae
    #     num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
    #     num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
    #     rho = []
    #     tau = []
    #     pk10 = []
    #     pk20 = []
        
    #     self.results = []
    #     for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
    #         pre = []
    #         gt = []
    #         t1 = time.time()
    #         for j in j_list:
    #             # print(i, j)
    #             g = self.testing_graph[i]
    #             g1 = self.training_graph[j]
    #             source_batch = Batch.from_data_list([g])
    #             target_batch = Batch.from_data_list([g1])
    #             data = self.transform((source_batch, target_batch ))
    #             # print(data["g1"].edge_index)
    #             target, gt_ged = data["target"].cpu().item(), data["target_ged"].cpu()
    #             # print(data["id_1"], data["id_2"], gt_ged)
    #             # print(data["g2"].x)
    #             # print(data["g2"].i)
    #             # exit()
    #             model_out = self.model(data) # if test_k == 0 else self.test_matching(data, test_k)
    #             prediction = model_out[0].cpu()
    #             pre_ged = prediction.detach().cpu() * (max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2)
    #             round_pre_ged = round(pre_ged.cpu().item())
    #             prediction = prediction.cpu()

    #             num += 1
    #             if prediction is None:
    #                 print("test")
    #                 mse.append(-0.001)
    #             elif prediction.shape[0] == 1:
    #                 mse.append((prediction.item() - target) ** 2)
    #             else:  # TaGSim
    #                 mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
    #             pre.append(pre_ged.numpy())
    #             gt.append(gt_ged.numpy())

    #             mae.append(abs(round_pre_ged - gt_ged.numpy()))
    #             if round_pre_ged == gt_ged:
    #                 num_acc += 1
    #                 num_fea += 1
    #             elif round_pre_ged > gt_ged:
    #                 num_fea += 1
    #         t2 = time.time()
    #         time_usage.append(t2 - t1)
    #         rho.append(spearmanr(pre, gt)[0])
    #         tau.append(kendalltau(pre, gt)[0])
    #         pk10.append(self.cal_pk(10, pre, gt))
    #         pk20.append(self.cal_pk(20, pre, gt))

    #     time_usage = round(np.mean(time_usage), 3)
    #     # print(mae)
    #     mse = round(np.mean(mse) * 1000, 3)
    #     mae = round(np.mean(mae), 3)
    #     acc = round(num_acc / num, 3)
    #     fea = round(num_fea / num, 3)
    #     rho = round(np.mean(rho), 3)
    #     tau = round(np.mean(tau), 3)
    #     pk10 = round(np.mean(pk10), 3)
    #     pk20 = round(np.mean(pk20), 3)

    #     print(mae)
    #     print(acc)

    #     # self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mse', 'mae', 'acc',
    #     #                      'fea', 'rho', 'tau', 'pk10', 'pk20'))
    #     # self.results.append(("GEDGNN", self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
    #     #                      fea, rho, tau, pk10, pk20))

    #     # print(*self.results[-2], sep='\t')
    #     # print(*self.results[-1], sep='\t')
    #     # with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
    #     #     if test_k == 0:
    #     #         print("## Testing", file=f)
    #     #     else:
    #     #         print("## Post-processing", file=f)
    #     #     print("```", file=f)
    #     #     print(*self.results[-2], sep='\t', file=f)
    #     #     print(*self.results[-1], sep='\t', file=f)
    #     #     print("```\n", file=f)

    #     # """
    #     # Scoring.
    #     # """
    #     # print("\n\nModel evaluation.\n")
        
    #     # self.model.load_state_dict(torch.load("models/GEDGNN/best_model/AIDS_20"))
    #     # #self.load_model()
    #     # self.model.eval()
        
    #     # # scores = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     # # mae = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     # # ground_truth = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     # # ground_truth_ged = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
    #     # # prediction_mat = np.empty((self.num_test_graphs, (self.num_train_graphs + self.num_val_graphs)))
        
    #     # # rho_list = []
    #     # # tau_list = []
    #     # # prec_at_10_list = [] 
    #     # # prec_at_20_list = []

    #     # mse = []
    #     # mae = []
        
    #     # t = tqdm(total=self.num_test_graphs*100)

    #     # n1n2 = []

    #     # for g1 in self.testing_graph:
    #     #     n1n2.append([])
    #     #     for g2 in self.training_graph:
    #     #         if self.args["normalization"] == "exp":
    #     #             temp = g1.num_nodes + g2.num_nodes
    #     #         elif self.args["normalization"] == "linear":
    #     #             temp = max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges)
    #     #         elif self.args["normalization"] != "none":
    #     #             raise TypeError("Invalid Normalization")
    #     #         n1n2[-1].append(temp)

    #     # n1n2 = torch.Tensor(n1n2)

    #     # test_test_graph = []
    #     # for graph in self.training_graph:
    #     #     if graph.num_nodes > 10:
    #     #         continue
    #     #     test_test_graph.append(graph)

    #     # # print(self.num_train_graphs)

    #     # random.seed(1)
    #     # random.shuffle(test_test_graph)

    #     # for i, g in enumerate(self.testing_graph):
    #     #     target_list = []
    #     #     ged_list = []
    #     #     prediction_list = []
    #     #     ged_pred_list = []
    #     #     for j, g1 in enumerate(test_test_graph[:100]):
    #     #         # if g.num_nodes > 10 or g1.num_nodes > 10:
    #     #         #     print(g.i, g1.i)
    #     #         #     continue
    #     #         source_batch = Batch.from_data_list([g])
    #     #         target_batch = Batch.from_data_list([g1])

    #     #         # print(g.i, g1.i, self.ged_matrix[g.i][g1.i])
                
    #     #         data = self.transform((source_batch, target_batch))
    #     #         target_list.append(data["target"])
    #     #         # ground_truth[i] = target
    #     #         ged_list.append(self.ged_matrix[g.i][g1.i])
    #     #         # ground_truth_ged[i] = target_ged

    #     #         if self.args["model"] == "GEDGNN":
    #     #             prediction, _ = self.model(data)
    #     #         else:
    #     #             prediction = self.model(data)
    #     #         #print(prediction.shape, target.shape)
    #     #         if self.args["normalization"] == "exp":
    #     #             ged_pred = -0.5 * torch.log(prediction.detach()) * n1n2[i][j]
    #     #         elif self.args["normalization"] == "linear":
    #     #             ged_pred = prediction.detach() * n1n2[i][j]
    #     #         elif self.args["normalization"] == "none":
    #     #             ged_pred = prediction
    #     #         else:
    #     #             raise TypeError("Invalid Normalization")
                
    #     #         mse.append((prediction.item() - data["target"]) ** 2)
    #     #         mae.append(abs(round(float(ged_pred)) - int(self.ged_matrix[g.i][g1.i])))
                
    #     #         # prediction_list.append(prediction)
    #     #         # ged_pred_list.append(ged_pred)
                
    #     #         t.update(1)

    #     #     # prediction_mat[i] = prediction_list.detach().numpy()
    #     #     # ground_truth[i] = target_list.detach().numpy()
    #     #     # ground_truth_ged[i] = ged_list.detach().numpy()
    #     #     # print(ged_list)
    #     #     # print(torch.round(ged_pred_list))
    #     #     # print("-------------------------------------------------------------------------")
            
    #     #     # print(prediction_list, target_list)
    #     #     # mse.append(F.mse_loss(torch.Tensor(prediction_list), torch.Tensor(target_list), reduction='none').detach().numpy())
    #     #     # mae.append(torch.abs(torch.Tensor(ged_list) - torch.round(torch.Tensor(ged_pred_list))).numpy())

    #     #     # # print(prediction_mat[i], ground_truth[i])
    #     #     # rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
    #     #     # tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
    #     #     # prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))
    #     #     # prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i], self.args["normalization"]))
    #     #     # # print(rho_list)

    #     # # self.rho = np.mean(rho_list).item()
    #     # # self.tau = np.mean(tau_list).item()
    #     # # self.prec_at_10 = np.mean(prec_at_10_list).item()
    #     # # self.prec_at_20 = np.mean(prec_at_20_list).item()
    #     # # self.model_error = np.mean(scores).item()
    #     # # self.model_mae = np.mean(mae).item()
    #     # # self.print_evaluation()
    #     # print("MAE: ", np.mean(mae))
    #     # print("MSE: ", np.mean(mse.cpu()) * 1000)



