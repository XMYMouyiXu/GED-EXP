"""Data processing utilities."""

import json
import math
import torch
import numpy as np
from texttable import Texttable
from scipy.stats import spearmanr, kendalltau

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged

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


def calculate_prec_at_k(k, prediction, target, target_ged, normalization="exp", type="raw"):
    """
    Calculating precision at k.
    """
    if normalization == "exp" and type == "normalized":
        best_k_pred = prediction.argsort()[::-1][:k]
        best_k_target = _calculate_prec_at_k(k, -target)
    elif normalization == "linear" or type == "raw":
        best_k_pred = prediction.argsort()[:k]
        # print(best_k_pred)
        best_k_target = _calculate_prec_at_k(k, target)
    else:
        raise TypeError("Invalid normalization")

    # print(target_ged.argsort())
    best_k_target_ged = _calculate_prec_at_k(k, target_ged)
    # print(best_k_target_ged)
    # print(set(best_k_target_ged))
    # print(set(best_k_pred))
    # print(set(best_k_pred).intersection(set(best_k_target_ged)))
    
    return len(set(best_k_pred).intersection(set(best_k_target_ged))) / k


def evaluation(prediction_list, ged_pred_list, target_list, target_ged_list):
    prediction_list = torch.tensor(prediction_list).detach().numpy()
    ged_pred_list = torch.tensor(ged_pred_list).detach().numpy()
    # print(prediction_list)
    # print(ged_pred_list)
    target_list = torch.tensor(target_list).detach().numpy()
    target_ged_list = torch.tensor(target_ged_list).detach().numpy()
    
    if prediction_list.shape[0] > 10:
        rho = calculate_ranking_correlation(spearmanr, prediction_list, target_list)
        tau = calculate_ranking_correlation(kendalltau, prediction_list, target_list)
    else:
        rho = None
        tau = None
    if prediction_list.shape[0] > 20:
        p10 = calculate_prec_at_k(10,ged_pred_list, target_list, target_ged_list)
    else:
        p10 = None
    if prediction_list.shape[0] > 30:
        p20 = calculate_prec_at_k(20,ged_pred_list, target_list, target_ged_list)
    else:
        p20 = None
    # else:
    #     p10 = 
    #     p20 = 

    return rho, tau, p10, p20
