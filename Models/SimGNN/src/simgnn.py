"""SimGNN class and runner."""

import glob
import torch
import random
import numpy as np
import time
import os
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged, evaluation

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        # print(features_1.shape)
        # print(edge_index_1.shape)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.device = "cuda"
        self.initial_label_enumeration()
        self.setup_model()
        self.training_time = 0
        self.train_test_time = 0


    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels).to(self.device)
        
        
    def generate_one_set(self, set1, set2):
        graphs = []
        for graph_1 in set1:
            for graph_2 in set2:
                # if graph_1["i"] > graph_2["i"]:
                #     continue
                if self.ged[graph_1["i"], graph_2["i"]] < 0:
                    continue
                if graph_1.x is not None:
                    temp_label_1 = [temp.argmax() for temp in graph_1.x]
                    temp_label_2 = [temp.argmax() for temp in graph_2.x]
                else:
                    temp_label_1 = [[0] for _ in range(graph_1.num_nodes)]
                    temp_label_2 = [[0] for _ in range(graph_2.num_nodes)]
                    
                edges_1 = graph_1.edge_index
                edges_2 = graph_2.edge_index
                
                data = {
                    "graph_1": graph_1,
                    "graph_2": graph_2,
                    "labels_1": temp_label_1,
                    "labels_2": temp_label_2,
                    "features_1": graph_1.x if graph_1.x is not None else torch.tensor([[1.0] for _ in range(graph_1.num_nodes)]),
                    "features_2": graph_2.x if graph_2.x is not None else torch.tensor([[1.0] for _ in range(graph_2.num_nodes)]), 
                    "edges_1": graph_1.edge_index,
                    "edges_2": graph_2.edge_index,
                    "ged": self.ged[graph_1["i"], graph_2["i"]]
                }
                graphs.append(data)
        return graphs
        
    def generate_graph_pairs(self):
        from torch_geometric.datasets import GEDDataset
        file_path = "../../datasets/pygdatasets/"
        
        if self.args.dataset == "new_IMDB":
            self.training_set = GEDDataset(file_path, "IMDBMulti", train=True) 
            self.testing_set = GEDDataset(file_path, "IMDBMulti", train=False)             
        else:
            self.training_set = GEDDataset(file_path, self.args.dataset, train=True) 
            self.testing_set = GEDDataset(file_path, self.args.dataset, train=False) 
        # self.training_set = GEDDataset(file_path, self.args.dataset, train=True)
        # self.testing_set = GEDDataset(file_path, self.args.dataset, train=False)
        self.ged = self.training_set.ged
        
        
        if self.args.dataset == "new_IMDB":
            with open("../../datasets/datasets/IMDBMulti/ged", "r") as f:
                test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]
            self.ged = torch.tensor(test_scores)
            

        if os.path.exists("../../datasets/datasets/{}/test_ged".format(self.args.dataset)):
            with open("../../datasets/datasets/{}/test_ged".format(self.args.dataset), "r") as f:
                test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]


            num_test_graphs = len(self.testing_set)
            num_train_graphs = len(self.training_set)
            for i in range(num_test_graphs):
                for j in range(i, num_test_graphs):
                    g1 = self.testing_set[i]
                    g2 = self.testing_set[j]
                    # print(g1["i"] - i)
                    # print(g2["i"] - j)
                    self.ged[g1["i"], g2["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs]
                    self.ged[g2["i"], g1["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs]

                    # if self.args["normalization"] == "exp":
                    #     s = torch.exp(-2 * torch.tensor(test_scores[i][j]) / (g1.num_nodes + g2.num_nodes))
                    # elif self.args["normalization"] == "linear":
                    #     s = test_scores[i][j] / (max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges) // 2) 
                    # elif self.args["normalization"] != "none":
                    #     raise TypeError("Invalid normalization approach.")
                    
                    # temp = g1.num_nodes + g2.num_nodes
                    
                    # self.nged_matrix[g1["i"], g2["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs] / temp
                    # self.nged_matrix[g2["i"], g1["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs] / temp

        
        self.training_graphs = self.generate_one_set(self.training_set, self.training_set)
        self.train_test_graphs = self.generate_one_set(self.training_set, self.testing_set)
        self.test_test_graphs = self.generate_one_set(self.testing_set, self.testing_set)
        

    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        # self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        # self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        self.generate_graph_pairs()
        graph_pairs = self.training_graphs + self.train_test_graphs
        if self.args.dataset in ["AIDS700nef"]:
            self.global_labels = set(range(29))
        else:
            self.global_labels = set([0])
        # for data in tqdm(graph_pairs):
        #     # data = process_pair(graph_pair)
        #     self.global_labels = self.global_labels.union(set(data["labels_1"]))
        #     self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = sorted(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        # edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

        # edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        # edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        # edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        # features_1, features_2 = [], []

        # for n in data["labels_1"]:
        #     features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        # for n in data["labels_2"]:
        #     features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        # features_1 = torch.FloatTensor(np.array(features_1))
        # features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = data["edges_1"].to(self.device)
        new_data["edge_index_2"] = data["edges_2"].to(self.device)

        # new_data["features_1"] = data["graph_1"].x if data["graph_1"].x is not None else torch.tensor([[1.0] for _ in range(data["graph_1"].num_nodes)])
        # new_data["features_2"] = data["graph_2"].x if data["graph_2"].x is not None else torch.tensor([[1.0] for _ in range(data["graph_2"].num_nodes)])
        
        new_data["features_1"] = data["features_1"].to(self.device)
        new_data["features_2"] = data["features_2"].to(self.device)

        norm_ged = (data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))).to(self.device)

        new_data["target"] = (torch.exp(-norm_ged).reshape(1, 1).view(-1).float()).to(self.device)
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        start_time = time.time()
        for data in batch:
            # data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data).reshape(1)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        self.training_time += time.time() - start_time
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            # print(len(batches))
            # print(len(self.training_graphs))
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            
            self.save()
            print("Training Time: {}".format(self.training_time))

    def score(self, test_type="traintest"):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.mse = []
        self.mae = []
        self.mse_linear = []
        

        prediction_list = []
        ged_pred_list = []
        target_list = []
        target_ged_list = []
        
        if test_type == "traintest":
            other_testing_set = self.train_test_graphs
        elif test_type == "testtest":
            other_testing_set = self.test_test_graphs
        else:
            raise TypeError("Invalid test type.")
        
        print(len(other_testing_set))
        prediction_list_graph = [[] for _ in range(len(self.testing_set))]
        ged_pred_list_graph = [[] for _ in range(len(self.testing_set))]
        target_list_graph = [[] for _ in range(len(self.testing_set))]
        target_ged_list_graph = [[] for _ in range(len(self.testing_set))]
        
        
        for d in tqdm(other_testing_set):
            # data = process_pair(graph_pair)
            # self.ground_truth.append(calculate_normalized_ged(data).detach().numpy())
            # print(self.ground_truth)
            data = self.transfer_to_torch(d)
            target = data["target"].cpu()
            prediction = self.model(data).reshape(1).cpu()
            # print(prediction[0])
            if prediction[0] == 0:
                continue

            # print(data)

            graph_1 = d["graph_1"]
            graph_2 = d["graph_2"]

            current_idx = graph_2["i"] - len(self.training_set)

            temp = graph_1.num_nodes + graph_2.num_nodes
            ged_pred = -0.5 * torch.log(prediction.detach()) * temp
            
            temp = max(graph_1.num_nodes, graph_2.num_nodes) + max(graph_1.num_edges, graph_2.num_edges) // 2
            linear_pred = ged_pred / temp
            linear_target = d["ged"] / temp
            
            prediction_list.append(prediction)
            ged_pred_list.append(ged_pred)
            target_list.append(target)
            target_ged_list.append(d["ged"])
            
            prediction_list_graph[current_idx].append(prediction)
            ged_pred_list_graph[current_idx].append(ged_pred)
            target_list_graph[current_idx].append(target)
            target_ged_list_graph[current_idx].append(d["ged"])
            

            current_mae = torch.abs(d["ged"] - torch.round(ged_pred)).detach().numpy()
            current_error = torch.nn.functional.mse_loss(prediction, target).detach().numpy()
            current_mse_linear = torch.nn.functional.mse_loss(linear_pred, linear_target).detach().numpy()

            
            
            self.mse.append(current_error)
            self.mse_linear.append(current_mse_linear)
            self.mae.append(current_mae)
            # self.scores.append(calculate_loss(prediction, target))
        # self.print_evaluation()
        
        
        self.rho = []
        self.tau = []
        self.p10 = []
        self.p20 = []
        for i in range(len(self.testing_set)):
            rho, tau, p10, p20 = evaluation(prediction_list_graph[i], 
                                            ged_pred_list_graph[i], 
                                            target_list_graph[i], 
                                            target_ged_list_graph[i])
            if rho is not None:
                self.rho.append(rho)
                self.tau.append(tau)
            if p10 is not None:
                self.p10.append(p10)
            if p20 is not None:
                self.p20.append(p20)        
        
        
        
        
        
        
        print("MSE(*10^-3): {}".format(np.mean(np.array(self.mse)) * 1000))
        print("MSE_linear(*10^-3): {}".format(np.mean(np.array(self.mse_linear)) * 1000))
        print("MAE: {}".format(np.mean(np.array(self.mae))))
        print("Rho: {}".format(np.mean(np.array(self.rho))))
        print("Tau: {}".format(np.mean(np.array(self.tau))))
        print("P10: {}".format(np.mean(np.array(self.p10))))
        print("P20: {}".format(np.mean(np.array(self.p20))))
        

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error, 5))+".")
        print("\nModel test error: " +str(round(model_error, 5))+".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
