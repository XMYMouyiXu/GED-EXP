import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau
from copy import deepcopy

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
        self.norm = "exp"


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
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_g.state_dict(), PATH_g)

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_c.state_dict(), PATH_c)
        print('Model Saved')

    def load_model(self):
        """
        Loading a EGSC.
        """
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_g.load_state_dict(torch.load(PATH_g))

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_c.load_state_dict(torch.load(PATH_c))
        print('Model Loaded')
        
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        
        # self.testing_graphs.norm_ged
        self.nged_matrix = self.training_graphs.norm_ged
        self.ged_matrix = self.training_graphs.ged
        self.lged_matrix = deepcopy(self.training_graphs.ged)

        for g1 in self.training_graphs:
            for g2 in self.training_graphs:
                temp = max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges)
                self.lged_matrix[g1.i, g2.i] /= temp

        for g1 in self.training_graphs:
            for g2 in self.testing_graphs:
                temp = max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges)
                self.lged_matrix[g1.i, g2.i] /= temp
                self.lged_matrix[g2.i, g1.i] /= temp
        

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
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
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

        if self.norm == "exp":
            normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
            
            new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        elif self.norm == "linear":
            linear_ged = self.lged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()]
            new_data["target"] = linear_ged.view(-1).float()

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

        
        prediction = self.model_c(self.model_g(data))
        loss = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        loss.backward()
        self.optimizer.step()
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
            
            if self.args.plot:
                if epoch % 10 == 0:
                    self.model_g.train(False)
                    self.model_c.train(False)
                    cnt_test = 20
                    cnt_train = 100
                    t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc = "Validation")
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g]*cnt_train)
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]
                        prediction = self.model_c(self.model_g(data))
                        
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
                        t.update(cnt_train)
                    
                    t.close()
                    loss_list_test.append(scores.mean().item())
                    self.model_g.train(True)
                    self.model_c.train(True)
            
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.process_batch(batch_pair)
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

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")

        self.model_g.eval()
        self.model_c.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))

        mae = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        
        t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs))

        n1n2 = []

        for g1 in self.testing_graphs:
            n1n2.append([])
            for g2 in self.training_graphs:
                if self.norm == "exp":
                    temp = g1.num_nodes + g2.num_nodes
                elif self.norm == "linear":
                    temp = max(g1.num_nodes, g2.num_nodes) + max(g1.num_edges, g2.num_edges)
                elif self.norm != "none":
                    raise TypeError("Invalid Normalization")
                n1n2[-1].append(temp)

        n1n2 = torch.Tensor(n1n2)

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g]*len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)
            
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged

            prediction = self.model_c(self.model_g(data))

            if self.norm == "exp":
                ged_pred = -0.5 * torch.log(prediction.detach()) * n1n2[i]
            elif self.norm == "linear":
                ged_pred = prediction.detach() * n1n2[i]
            elif self.norm == "none":
                ged_pred = prediction
            else:
                raise TypeError("Invalid Normalization")


            prediction_mat[i] = prediction.detach().numpy()
            mae[i] = torch.abs(torch.round(ged_pred) - target_ged)
                
            
            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

            t.update(len(self.training_graphs))

        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        self.model_mae = np.mean(mae).item()
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")
        print("mae: " + str(round(self.model_mae, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
