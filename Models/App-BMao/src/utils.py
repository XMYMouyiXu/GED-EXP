import json

import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy

import torch
from torch_geometric.utils import k_hop_subgraph, degree
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

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


def calculate_prec_at_k(k, prediction, target, target_ged, normalization="exp", type="normalized"):
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
    
    return len(set(best_k_pred).intersection(set(best_k_target))) / k
