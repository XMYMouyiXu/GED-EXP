import glob
import os
import torch
import random
import pickle
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from layers import TensorNetworkModule, GraphAggregationLayer
from utils import load_graphs, load_generated_graphs, process_pair, evaluation

class TaGSim(torch.nn.Module):

    def __init__(self, args, number_of_labels):

        super(TaGSim, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()


    def setup_layers(self):

        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons


        self.tensor_network_in = TensorNetworkModule(self.args, 7)# 7 for linux; 11 for IMDB # the valus here can be set by the users
        self.tensor_network_ie = TensorNetworkModule(self.args, 21)# 21 for linux; 60 for IMDB # the valus here can be set by the users

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_in = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_in = torch.nn.Linear(8, 4)
        self.scoring_layer_in = torch.nn.Linear(4, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ie = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ie = torch.nn.Linear(8, 4)
        self.scoring_layer_ie = torch.nn.Linear(4, 1)


    def gal_pass(self, edge_index, features):

        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2

    def forward(self, data):

        adj_1 = torch.FloatTensor(np.array(data["edge_index_1"].todense()))
        adj_2 = torch.FloatTensor(np.array(data["edge_index_2"].todense()))
        features_1, features_2 = data["features_1"], data["features_2"]
        
        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)#
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)#
    
        Graph1_hidden1, Graph1_hidden2, Graph2_hidden1, Graph2_hidden2 = [], [], [], []
        for i in range(graph1_hidden1.size()[0]):
            if(graph1_hidden1[i][0] >= 6):# 10 for imdb; 6 for linux # the valus here can be set by the users
                Graph1_hidden1.append([0.0]*5 + [1.0])
            else:
                Graph1_hidden1.append([1.0 if graph1_hidden1[i][0] == j else 0.0 for j in range(6)])

            if(graph1_hidden2[i][0] >= 15):# 50 for imdb; 15 for linux # the valus here can be set by the users
                Graph1_hidden2.append([0.0]*14 + [1.0])
            else:
                Graph1_hidden2.append([1.0 if graph1_hidden2[i][0] == j else 0.0 for j in range(15)])

        for i in range(graph2_hidden1.size()[0]):
            if(graph2_hidden1[i][0] >= 6):# 10 for imdb; 6 for linux # the valus here can be set by the users
                Graph2_hidden1.append([0.0]*5 + [1.0])
            else:
                Graph2_hidden1.append([1.0 if graph2_hidden1[i][0] == j else 0.0 for j in range(6)])

            if(graph2_hidden2[i][0] >= 15):# 50 for imdb; 15 for linux # the valus here can be set by the users
                Graph2_hidden2.append([0.0]*14 + [1.0])
            else:
                Graph2_hidden2.append([1.0 if graph2_hidden2[i][0] == j else 0.0 for j in range(15)])
            
        Graph1_hidden1, Graph1_hidden2 = torch.FloatTensor(np.array(Graph1_hidden1)), torch.FloatTensor(np.array(Graph1_hidden2))
        Graph2_hidden1, Graph2_hidden2 = torch.FloatTensor(np.array(Graph2_hidden1)), torch.FloatTensor(np.array(Graph2_hidden2))

        graph1_01concat = torch.cat([features_1, Graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, Graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([Graph1_hidden1, Graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([Graph2_hidden1, Graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)# default: sum
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)


        scores_in = self.tensor_network_in(graph1_01pooled, graph2_01pooled)
        scores_in = torch.t(scores_in)

        scores_in = torch.nn.functional.relu(self.fully_connected_first_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_second_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_third_in(scores_in))
        score_in = torch.sigmoid(self.scoring_layer_in(scores_in))

        scores_ie = self.tensor_network_ie(graph1_12pooled, graph2_12pooled)
        scores_ie = torch.t(scores_ie)

        scores_ie = torch.nn.functional.relu(self.fully_connected_first_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_second_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_third_ie(scores_ie))
        score_ie = torch.sigmoid(self.scoring_layer_ie(scores_ie))

        return torch.cat([score_in, score_ie], dim=1)

class TaGSimTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial_label_enumeration()
        self.model = TaGSim(self.args, self.number_of_labels)
        self.training_time = 0

    def initial_label_enumeration(self):
        
        self.training_pairs = load_generated_graphs(self.args.dataset, file_name='generated_graph_pairs')
        self.training_graphs = load_graphs(self.args.dataset, train_or_test='train')
        self.testing_graphs = load_graphs(self.args.dataset, train_or_test='test')

        self.number_of_labels = 1


    def transfer_to_torch(self, data, type_specified=True):

        new_data = dict()
        graph1, graph2 = data['graph_pair'][0], data['graph_pair'][1]
        nodes1, nodes2 = list(graph1.nodes()), list(graph2.nodes())

        features_1, features_2 = [], []

        for n in graph1.nodes():
            features_1.append([1.0])

        for n in graph2.nodes():
            features_2.append([1.0])

        features_1, features_2 = torch.FloatTensor(np.array(features_1)), torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"], new_data["edge_index_2"] = nx.adjacency_matrix(graph1), nx.adjacency_matrix(graph2)
        new_data["features_1"], new_data["features_2"] = features_1, features_2

        if(type_specified):
            norm_ged = [data['ged'][key] / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes())) for key in ['in', 'ie']]
            norm_ged = np.array(norm_ged)
            new_data["target"] = torch.from_numpy(np.exp(-norm_ged)).view(1,-1).float()
            
            norm_gt_ged = (data['ged']['in'] + data['ged']['ie']) / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes()))
            new_data["gt_ged"] = torch.from_numpy(np.exp(-norm_gt_ged).reshape(1, 1)).view(1, -1).float()
        else:
            norm_gt_ged = data['ged'] / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes()))
            new_data["gt_ged"] = torch.from_numpy(np.exp(-norm_gt_ged).reshape(1, 1)).view(1, -1).float()

        return new_data
#----------------------------------------------------------------------------------------------------------
    def fit(self):
        print("\n-------Model training---------.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        iteration = 0

        for epoch in range(self.args.epochs):
            random.shuffle(self.training_pairs)
            batches = []
            for graph in range(0, len(self.training_pairs), self.args.batch_size):
                batches.append(self.training_pairs[graph:graph+self.args.batch_size])

            for batch in batches:
                self.model.train()
                self.optimizer.zero_grad()
                losses = 0
                start_time = time.time()
                for graph_pair in batch:
                    data = self.transfer_to_torch(graph_pair)
                    prediction = self.model(data)
                    losses += torch.nn.functional.mse_loss(data["target"], prediction)
                losses.backward(retain_graph=True)
                self.optimizer.step()
                loss = losses.item()
                self.training_time += time.time() - start_time
                print('Iteration', iteration, 'loss: ', loss/len(batch))
            
                iteration += 1
                
        torch.save(self.model.state_dict(), "./{}_model_{}_new".format(self.args.idx, self.args.dataset))
        print("training time:" + str(self.training_time))
                
#-------------------------------------------------------------------------------------------------------
    # def test(self):
    #     print("\n\nModel testing.\n")
       
       
    #     self.model.load_state_dict(torch.load("./model_{}".format(self.args.dataset)))
    #     self.model.eval()
    #     self.test_scores= []
    #     test_gt_ged = load_generated_graphs(self.args.dataset, file_name='ged_matrix_test')
    #     for graph_1 in self.testing_graphs:
    #         for graph_2 in self.training_graphs:
    #             if((graph_1.graph['gid'], graph_2.graph['gid']) in test_gt_ged):
    #                 curr_graph_pair = {'graph_pair': [graph_1, graph_2], 'ged':test_gt_ged[(graph_1.graph['gid'], graph_2.graph['gid'])]}
    #                 data = self.transfer_to_torch(curr_graph_pair, type_specified=False)
    #                 prediction = self.model(data)
    #                 prediction = torch.exp(torch.sum(torch.log(prediction))).view(1, -1)
    #                 current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])
    #                 self.test_scores.append(current_error.data.item())

    #         for graph_2 in self.testing_graphs:
    #             if((graph_1.graph['gid'], graph_2.graph['gid']) in test_gt_ged):
    #                 curr_graph_pair = {'graph_pair': [graph_1, graph_2], 'ged':test_gt_ged[(graph_1.graph['gid'], graph_2.graph['gid'])]}
    #                 data = self.transfer_to_torch(curr_graph_pair, type_specified=False)
    #                 prediction = self.model(data)
    #                 prediction = torch.exp(torch.sum(torch.log(prediction))).view(1, -1)
    #                 current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])
    #                 self.test_scores.append(current_error.data.item())


    #     model_error = sum(self.test_scores) / len(self.test_scores)

    #     print("\nModel test error: " + str(model_error))

    def test(self):
        print("\n\nModel testing.\n")

        self.model.load_state_dict(torch.load("./{}_model_{}_new".format(self.args.idx, self.args.dataset)))
        self.model.eval()
        self.train_test_scores = []
        self.test_test_scores = []
        self.train_test_scores_linear = []
        self.test_test_scores_linear = []
        self.train_test_mae = []
        self.test_test_mae = []
        self.train_test_rho = []
        self.test_test_rho = []
        self.train_test_tau = []
        self.test_test_tau = []
        self.train_test_p10 = []
        self.test_test_p10 = []
        self.train_test_p20 = []
        self.test_test_p20 = []
        
        self.train_test_time = 0
        self.test_test_time = 0
        
        
        test_gt_ged = load_generated_graphs(self.args.dataset, file_name='ged_matrix_test')
        for graph_1 in self.testing_graphs:
            prediction_list = []
            ged_pred_list = []
            target_list = []
            target_ged_list = []
            for graph_2 in self.training_graphs:
                if not ((graph_1.graph['gid'], graph_2.graph['gid']) in test_gt_ged):
                    if not ((graph_2.graph['gid'], graph_1.graph['gid']) in test_gt_ged):
                        continue
                    curr_graph_pair = {'graph_pair': [graph_2, graph_1], 'ged':test_gt_ged[(graph_2.graph['gid'], graph_1.graph['gid'])]}
                else:
                    curr_graph_pair = {'graph_pair': [graph_1, graph_2], 'ged':test_gt_ged[(graph_1.graph['gid'], graph_2.graph['gid'])]}
  
                data = self.transfer_to_torch(curr_graph_pair, type_specified=False)
                start_time = time.time()
                prediction = self.model(data)
                self.train_test_time += time.time() - start_time
                prediction = torch.exp(torch.sum(torch.log(prediction))).view(1, -1)
                
                
                #unnormalized
                temp = graph_1.number_of_nodes() + graph_2.number_of_nodes()
                ged_pred = -0.5 * torch.log(prediction.detach()) * temp
                
                
                temp = max(graph_1.number_of_nodes(), graph_2.number_of_nodes()) + max(graph_1.number_of_edges(), graph_2.number_of_edges())
                pred_linear = torch.round(ged_pred) / temp
                target_linear = curr_graph_pair["ged"] / temp
                
                
                prediction_list.append(prediction)
                ged_pred_list.append(ged_pred)
                target_list.append(data["gt_ged"])
                target_ged_list.append(curr_graph_pair["ged"])
                

                current_mae = torch.abs(curr_graph_pair["ged"] - torch.round(ged_pred)).detach().numpy()
                current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])
                current_error_linear = torch.nn.functional.mse_loss(torch.Tensor([pred_linear]), torch.Tensor([target_linear]))
                
                self.train_test_scores.append(current_error.data.item())
                self.train_test_scores_linear.append(current_error_linear.data.item())
                self.train_test_mae.append(current_mae)
            
            if len(prediction_list) > 0:
                rho, tau, p10, p20 = evaluation(prediction_list, ged_pred_list, target_list, target_ged_list)
                if rho is not None:
                    self.train_test_rho.append(rho)
                    self.train_test_tau.append(tau)
                if p10 is not None:
                    self.train_test_p10.append(p10)
                if p20 is not None:
                    self.train_test_p20.append(p20)

                    


            prediction_list = []
            ged_pred_list = []
            target_list = []
            target_ged_list = []
            for graph_2 in self.testing_graphs:
                if not ((graph_1.graph['gid'], graph_2.graph['gid']) in test_gt_ged):
                    if not ((graph_2.graph['gid'], graph_1.graph['gid']) in test_gt_ged):
                        continue
                    curr_graph_pair = {'graph_pair': [graph_2, graph_1], 'ged':test_gt_ged[(graph_2.graph['gid'], graph_1.graph['gid'])]}
                else:
                    curr_graph_pair = {'graph_pair': [graph_1, graph_2], 'ged':test_gt_ged[(graph_1.graph['gid'], graph_2.graph['gid'])]}
                data = self.transfer_to_torch(curr_graph_pair, type_specified=False)
                start_time = time.time()
                prediction = self.model(data)
                self.test_test_time = time.time() - start_time
                prediction = torch.exp(torch.sum(torch.log(prediction))).view(1, -1)
                # current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])
                # self.test_test_scores.append(current_error.data.item())

                temp = graph_1.number_of_nodes() + graph_2.number_of_nodes()
                ged_pred = -0.5 * torch.log(prediction.detach()) * temp
                
                temp = max(graph_1.number_of_nodes(), graph_2.number_of_nodes()) + max(graph_1.number_of_edges(), graph_2.number_of_edges())
                pred_linear = torch.round(ged_pred) / temp
                target_linear = curr_graph_pair["ged"] / temp
                
                prediction_list.append(prediction)
                ged_pred_list.append(ged_pred)
                target_list.append(data["gt_ged"])
                target_ged_list.append(curr_graph_pair["ged"])

                current_mae = torch.abs(curr_graph_pair["ged"] - torch.round(ged_pred)).detach().numpy()
                current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])
                current_error_linear = torch.nn.functional.mse_loss(torch.Tensor([pred_linear]), torch.Tensor([target_linear]))
                
                self.test_test_scores.append(current_error.data.item())
                self.test_test_mae.append(current_mae)
                self.test_test_scores_linear.append(current_error_linear.data.item())
            
            if len(prediction_list) > 0:
                rho, tau, p10, p20 = evaluation(prediction_list, ged_pred_list, target_list, target_ged_list)
                if rho is not None:
                    self.test_test_rho.append(rho)
                    self.test_test_tau.append(tau)
                if p10 is not None:
                    self.test_test_p10.append(p10)
                if p20 is not None:
                    self.test_test_p20.append(p20)

        # model_error = sum(self.test_scores) / len(self.test_scores)
        # print("\nModel test error: " + str(model_error))
        self.print_evaluation()

    def print_evaluation(self):
        print("=================================")
        print("Train-Test:")
        # print(self.train_test_mae)
        print("MSE: {}".format(np.mean(np.array(self.train_test_scores)) * 1000))
        print("MSE_Linear: {}".format(np.mean(np.array(self.train_test_scores_linear)) * 1000))
        print("MAE: {}".format(np.mean(np.array(self.train_test_mae))))
        print("Rho: {}".format(np.mean(np.array(self.train_test_rho))))
        print("Tau: {}".format(np.mean(np.array(self.train_test_tau))))
        print("P@10: {}".format(np.mean(np.array(self.train_test_p10))))
        print("P@20: {}".format(np.mean(np.array(self.train_test_p20))))
        print("training_time: {}".format(self.training_time))
        print("testing_time: {}".format(self.train_test_time))
        print("Test-Test:")
        print("MSE: {}".format(np.mean(np.array(self.test_test_scores)) * 1000))
        print("MSE_Linear: {}".format(np.mean(np.array(self.test_test_scores_linear)) * 1000))
        print("MAE: {}".format(np.mean(np.array(self.test_test_mae))))
        print("Rho: {}".format(np.mean(np.array(self.test_test_rho))))
        print("Tau: {}".format(np.mean(np.array(self.test_test_tau))))
        print("P@10: {}".format(np.mean(np.array(self.test_test_p10))))
        print("P@20: {}".format(np.mean(np.array(self.test_test_p20))))
        print("testing_time: {}".format(self.test_test_time))