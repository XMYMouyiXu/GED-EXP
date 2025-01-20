import torch
import random
import time
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt

from model import EGSCT_generator, EGSCT_classifier

import pdb

class EGSCTrainer(object):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.setup_model()
        self.best_rho = 0
        self.best_tau = 0
        self.best_prec_at_10 = 0
        self.best_prec_at_20 = 0
        self.best_model_error = float('inf')
        self.training_time = 0
        self.testing_time = 0


    def setup_model(self):
        """
        Creating a EGSC.
        """
        self.model_g = EGSCT_generator(self.args, self.number_of_labels)
        self.model_c = EGSCT_classifier(self.args, self.number_of_labels)


    def save_model(self):
        """
        Saving a EGSC.
        """
        PATH_g = './model_saved/{}_EGSC_g_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator) + '.pth'
        torch.save(self.model_g.state_dict(), PATH_g)

        PATH_c = './model_saved/{}_EGSC_c_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+ '.pth'
        torch.save(self.model_c.state_dict(), PATH_c)
        print('Model Saved')

    def load_model(self):
        """
        Loading a EGSC.
        """
        # PATH_g = './model_saved/{}_EGSC_g_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        # + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        PATH_g = './model_saved/{}_EGSC_g_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator) + '.pth'
        self.model_g.load_state_dict(torch.load(PATH_g))

        # PATH_c = './model_saved/{}_EGSC_c_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        # + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        PATH_c = './model_saved/{}_EGSC_c_EarlyFusion_'.format(self.args.idx) +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+  '.pth'
        self.model_c.load_state_dict(torch.load(PATH_c))
        print('Model Loaded')
        
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../../datasets/pygdatasets/'

        if self.args.dataset == "new_IMDB":
            self.training_graphs = GEDDataset(self.args.data_dir, "IMDBMulti", train=True) 
            self.testing_graphs = GEDDataset(self.args.data_dir, "IMDBMulti", train=False)             
        else:
            self.training_graphs = GEDDataset(self.args.data_dir, self.args.dataset, train=True) 
            self.testing_graphs = GEDDataset(self.args.data_dir, self.args.dataset, train=False) 

        # self.testing_graphs.norm_ged
        self.nged_matrix = self.training_graphs.norm_ged
        self.ged_matrix = self.training_graphs.ged
        
        # print(list(self.nged_matrix[0][:20]))
        
        if self.args.dataset == "new_IMDB":
            with open("../../../datasets/datasets/IMDBMulti/ged", "r") as f:
                test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]
            self.ged_matrix = torch.tensor(test_scores)
            

            # exit()
            num_graphs = len(self.training_graphs) + len(self.testing_graphs)
            all_graphs = self.training_graphs + self.testing_graphs
            for i in range(num_graphs):
                for j in range(i, num_graphs):
                    g1 = all_graphs[i]
                    g2 = all_graphs[j]
                    if test_scores[g1["i"]][g2["i"]] == -1:
                        self.nged_matrix[g1["i"], g2["i"]] = torch.tensor(-1.0)
                        self.nged_matrix[g2["i"], g1["i"]] = torch.tensor(-1.0)
                        continue
                    # self.ged_matrix[g1["i"], g2["i"]] = test_scores[g1["i"]][g2["i"]]
                    # self.ged_matrix[g2["i"], g1["i"]] = test_scores[g1["i"]][g2["i"]]

                    temp = g1.num_nodes + g2.num_nodes
                    
                    self.nged_matrix[g1["i"], g2["i"]] = 2 * self.ged_matrix[g1["i"], g2["i"]] / temp
                    self.nged_matrix[g2["i"], g1["i"]] = 2 * self.ged_matrix[g1["i"], g2["i"]] / temp

        # print(list(self.nged_matrix[0][:20]))
        # exit()
        
        if os.path.exists("../../../datasets/datasets/{}/test_ged".format(self.args.dataset)):
            with open("../../../datasets/datasets/{}/test_ged".format(self.args.dataset), "r") as f:
                test_scores = [list(map(int, line.strip().split())) for line in f.readlines()]


            num_test_graphs = len(self.testing_graphs)
            num_train_graphs = len(self.training_graphs)
            for i in range(num_test_graphs):
                for j in range(i, num_test_graphs):
                    g1 = self.testing_graphs[i]
                    g2 = self.testing_graphs[j]
                    # print(g1["i"] - i)
                    # print(g2["i"] - j)
                    self.ged_matrix[g1["i"], g2["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs]
                    self.ged_matrix[g2["i"], g1["i"]] = test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs]

                    # if self.args["normalization"] == "exp":
                    #     s = torch.exp(-2 * torch.tensor(test_scores[i][j]) / (g1.num_nodes + g2.num_nodes))
                    # elif self.args["normalization"] == "linear":
                    #     s = test_scores[i][j] / (max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges) // 2) 
                    # elif self.args["normalization"] != "none":
                    #     raise TypeError("Invalid normalization approach.")
                    
                    temp = g1.num_nodes + g2.num_nodes
                    
                    self.nged_matrix[g1["i"], g2["i"]] = 2 * test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs] / temp
                    self.nged_matrix[g2["i"], g1["i"]] = 2 * test_scores[g1["i"] - num_train_graphs][g2["i"] - num_train_graphs] / temp

        # print(self.nged_matrix[560])
        # print(num_train_graphs)
        # print(self.ged_matrix[560, 560:])
        # print(test_scores[0])
        # exit()
        

        self.real_data_size = self.nged_matrix.size(0)
        
        if self.args.synth:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)  
            
            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat((self.nged_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat((torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))
        
        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs + (self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
        
        # labeling of synth data according to real data format    
            if self.args.synth:
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size
                    
        self.number_of_labels = self.training_graphs.num_features

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)

        temp_training_graphs_1 = self.training_graphs.shuffle()
        temp_training_graphs_2 = self.training_graphs.shuffle()

        training_graphs_1 = []
        training_graphs_2 = []

        for i in range(len(temp_training_graphs_1)):
            g1 = temp_training_graphs_1[i]
            g2 = temp_training_graphs_2[i]
            if self.ged_matrix[g1["i"], g2["i"]] == -1:
                continue
            training_graphs_1.append(g1)
            training_graphs_2.append(g2)

        source_loader = DataLoader(training_graphs_1, batch_size=self.args.batch_size)
        target_loader = DataLoader(training_graphs_2, batch_size=self.args.batch_size)
        
        
        # source_loader = DataLoader(self.training_graphs.shuffle() + 
        #     ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        # target_loader = DataLoader(self.training_graphs.shuffle() + 
        #     ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()

        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()

        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data. 
        """
        self.optimizer.zero_grad()
        
        data = self.transform(data)
        target = data["target"]

        # if int(data["target_ged"][0]) == -1:
        #     return None

        start_time = time.time()
        prediction = self.model_c(self.model_g(data))

        loss = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        loss.backward()
        self.optimizer.step()
        self.training_time += time.time() - start_time
        
        return loss.item()
        
    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam([{'params': self.model_g.parameters()}, {'params': self.model_c.parameters()}],\
         lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model_g.train()
        self.model_c.train()

        
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        loss_list = []
        loss_list_test = []
        for epoch in epochs:
            
            # if self.args.plot:
            #     if epoch % 10 == 0:
            #         self.model_g.train(False)
            #         self.model_c.train(False)
            #         cnt_test = 20
            #         cnt_train = 100
            #         t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc = "Validation")
            #         scores = torch.empty((cnt_test, cnt_train))
                    
            #         for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
            #             source_list = []
            #             target_list = []
            #             temp_list = self.training_graphs[:cnt_train].shuffle()
            #             for graph in temp_list:
            #                 if self.ged_matrix[g["i"], graph["i"]] != -1:
            #                     source_list.append(g)
            #                     target_list.append(graph)
            #             source_batch = Batch.from_data_list(source_list)
            #             target_batch = Batch.from_data_list(target_list)
            #             data = self.transform((source_batch, target_batch))
            #             target = data["target"]
            #             prediction = self.model_c(self.model_g(data))
                        
            #             scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
            #             t.update(cnt_train)
                    
            #         t.close()
            #         loss_list_test.append(scores.mean().item())
            #         self.model_g.train(True)
            #         self.model_c.train(True)
            
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.process_batch(batch_pair)
                if loss_score is None:
                    continue
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
            loss_list.append(loss)
            
        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")
            plt.ylim([0, 0.01])
            plt.legend()
            filename = self.args.dataset
            filename += '_' + self.args.gnn_operator 
            filename = filename + str(self.args.epochs) + '.pdf'
            plt.savefig(filename)

    def score(self, test_type="traintest"):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")

        self.load_model()

        self.model_g.eval()
        self.model_c.eval()
        
        
        if test_type == "traintest":
            other_test_set = self.training_graphs
        elif test_type == "testtest":
            other_test_set = self.testing_graphs
        
        
        # scores = np.empty((len(self.testing_graphs), len(other_test_set)))
        # scores_linear = np.empty((len(self.testing_graphs), len(other_test_set)))
        # mae = np.empty((len(self.testing_graphs), len(other_test_set)))
        # ground_truth = np.empty((len(self.testing_graphs), len(other_test_set)))
        # ground_truth_ged = np.empty((len(self.testing_graphs), len(other_test_set)))
        # prediction_mat = np.empty((len(self.testing_graphs), len(other_test_set)))
        
        scores = []
        scores_linear = []
        mae = []
        ground_truth = []
        ground_truth_ged = []
        prediction_mat = []
        
        # print(scores)
        
        n1s = np.empty((len(self.testing_graphs), len(other_test_set)))
        n2s = np.empty((len(self.testing_graphs), len(other_test_set)))
        
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        
        t = tqdm(total=len(self.testing_graphs)*len(other_test_set))

        
        # num_nodes = [g.num_nodes for g in other_test_set]
        
        # for i, g in enumerate(self.testing_graphs):
        #     for j, g1 in enumerate(other_test_set):
        #     source_batch = Batch.from_data_list([g]*len(other_test_set))
        #     target_batch = Batch.from_data_list(other_test_set)
            
        #     n1s[i] = torch.bincount(source_batch.batch).detach().numpy()
        #     n2s[i] = torch.bincount(target_batch.batch).detach().numpy()
            
        #     data = self.transform((source_batch, target_batch))
        #     target = data["target"]
        #     ground_truth[i] = target
        #     target_ged = data["target_ged"]
        #     ground_truth_ged[i] = target_ged


        #     start_time = time.time()
        #     prediction = self.model_c(self.model_g(data))
        #     self.testing_time += time.time() - start_time

        #     prediction_mat[i] = prediction.detach().numpy()
            
        #     scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()

        #     rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
        #     tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
        #     prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
        #     prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

        #     t.update(len(other_test_set))
        
        
        for i, g in enumerate(self.testing_graphs):
            # scores.append([])
            # scores_linear.append([])
            # mae.append([])
            ground_truth.append([])
            ground_truth_ged.append([])
            prediction_mat.append([])
            for j, g1 in enumerate(other_test_set):
                if int(self.ged_matrix[g["i"], g1["i"]]) == -1:
                    # scores[i][j] = np.nan
                    # scores_linear[i][j] = np.nan
                    # mae[i][j] = np.nan
                    # ground_truth = np.nan
                    # ground_truth_ged = np.nan
                    # prediction_mat = np.nan
                    continue
                # print(g["i"], g1["i"], int(self.ged_matrix[g["i"], g1["i"]]), int(self.ged_matrix[g1["i"], g["i"]]))
                source_batch = Batch.from_data_list([g])
                target_batch = Batch.from_data_list([g1])

                
                data = self.transform((source_batch, target_batch))
                target = data["target"]
                # ground_truth[i].append(float(target))
                target_ged = data["target_ged"]
                # print(target_ged, self.ged_matrix[g["i"], g1["i"]])
                # ground_truth_ged[i].append(int(target_ged))

                


                start_time = time.time()    
                prediction = self.model_c(self.model_g(data))

                if prediction == 0:
                    continue
                # print(prediction)
                self.testing_time += time.time() - start_time

                ground_truth_ged[i].append(int(target_ged))
                ground_truth[i].append(float(target))
                prediction_mat[i].append(float(prediction))
                
                scores.append(float(F.mse_loss(prediction, target, reduction='none')))

                temp = g.num_nodes + g1.num_nodes
                pre_ged = torch.round(-torch.log(prediction) * 0.5 * temp)
                
                mae.append(float(torch.abs(pre_ged - target_ged)))
                
                temp = max(g.num_nodes, g1.num_nodes) + max(g.num_edges, g1.num_edges) // 2
                prediction_linear = pre_ged / temp
                target_linear = target_ged / temp
                scores_linear.append(float(F.mse_loss(prediction_linear, target_linear, reduction='none')))

                #print(prediction, pre_ged, prediction_linear, g.num_nodes + g1.num_nodes, temp)

            if len(prediction_mat[i]) > 10:
                rho_list.append(calculate_ranking_correlation(spearmanr, np.array(prediction_mat[i]), np.array(ground_truth[i])))
                tau_list.append(calculate_ranking_correlation(kendalltau, np.array(prediction_mat[i]), np.array(ground_truth[i])))
            if len(prediction_mat[i]) > 20:
                prec_at_10_list.append(calculate_prec_at_k(10, np.array(prediction_mat[i]), np.array(ground_truth[i]), np.array(ground_truth_ged[i])))
            if len(prediction_mat[i]) > 30:
                prec_at_20_list.append(calculate_prec_at_k(20, np.array(prediction_mat[i]), np.array(ground_truth[i]), np.array(ground_truth_ged[i])))

            t.update(len(other_test_set))
            
        # prediction_ged_mat = -np.log(prediction_mat) * 0.5 * (n1s + n2s)
        
        # self.mae = np.mean(np.abs(np.round(prediction_ged_mat) - np.round(np.array(ground_truth_ged))))

        
        # scores = np.array(scores)
        # scores_linear = np.array(scores_linear)
        # mae = np.array(mae)

        # print(scores)
        # print(scores_linear)
        # print(mae)
        # exit()
        
        
        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        self.mae = np.mean(mae).item()
        self.model_error_linear = np.mean(scores_linear).item()
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")
        print("mse_linear(10^-3): " + str(round(self.model_error_linear*1000, 5)) + ".")
        print("mae: " + str(round(self.mae, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
        print("training time: " + str(self.training_time))
        print("testing time: " + str(self.testing_time))
