import os
import pickle
import random
import time
import copy
import json

# import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch_geometric as tg
from torch_geometric.data import Data
# from tqdm.auto import tqdm

from neuro import config, datasets, metrics, models, train, utils, viz
import pyged

from importlib import reload
reload(config)
reload(datasets)
reload(metrics)
reload(models)
reload(pyged)
reload(train)
reload(utils)
reload(viz)

def get_hyper(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    return data

args = get_hyper("./hyperparameters.json")

CUDA_INDEX = 0
NAME = args["dataset"]

if NAME == "GED_AIDS700nef":
    CLASSES = 29
elif NAME in ["GED_IMDBMulti", "GED_LINUX"]:
    CLASSES = 1
else:
    raise TypeError("Invalid Dataset")


root = "datasets/AIDS700nef/"
val_ratio = 0.1

torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))


def transfer_data_object(temp_set):
    for i in range(len(temp_set[0])):
        temp_set[0][i] = Data.from_dict(temp_set[0][i].__dict__)
        
    for i in range(len(temp_set[1])):
        temp_set[1][i] = Data.from_dict(temp_set[1][i].__dict__)
    
    return temp_set


train_set, train_meta = torch.load(f'../../datasets/GREEDDataset/{NAME}/train.pt', map_location='cpu')
train_set = transfer_data_object(train_set)

print(len(train_set))
print(len(train_set[0]))
# print(type(train_set[0][0]))

# Assuming 'data' is the object you loaded

    
# print(type(train_set[1]))
# print(type(train_set[2]))
# print(type(train_set[3]))

# for i in range(len(train_set)):
#     for j in range(len(train_set[i])):
#         print(Data.from_dict(train_set[i][j].__dict__), j)
#         train_set[i][j] = Data.from_dict(train_set[i][j].__dict__)
    
# print(train_set)
# train_meta = Data.from_dict(train_meta.__dict__)

# print(train_set[1])
# print(train_meta)

nodes = [h.num_nodes for h in train_set[1]]
edges = [h.num_edges for h in train_set[1]]
print(f'avg target nodes: {sum(nodes)/len(nodes):.3f}')
print(f'avg target edges: {sum(edges)/len(edges):.3f}')

# viz.plot_inner_dataset_plus(train_set, train_meta, n_items=5, random=True)
val_set, _ = torch.load(f'./greed-expts/data/{NAME}/val.pt', map_location='cpu')
val_set = transfer_data_object(val_set)



loader = tg.data.DataLoader(list(zip(*train_set)), batch_size=200, shuffle=True)
val_loader = tg.data.DataLoader(list(zip(*val_set)), batch_size=1000, shuffle=True)

if not os.path.exists(f'results/{NAME}-{args["model"]}'):
    os.system("mkdir " + f'results/{NAME}-{args["model"]}')
dump_path = os.path.join(f'results/{NAME}-{args["model"]}', str(time.time()))
os.mkdir(dump_path)

start_time = time.time()
if args["model"] == "origin":
    model = models.NormGEDModel(8, CLASSES, 64, 64).to(config.device)
    train.train_full(model, loader, val_loader, lr=1e-3, weight_decay=1e-3, cycle_patience=5, step_size_up=2000, step_size_down=2000, dump_path=dump_path)

elif args["model"] == "Dual":
    model = models.DualNormGEDModel(8, CLASSES, 64, 64).to(config.device)
    train.train_full(model, loader, val_loader, lr=1e-3, weight_decay=1e-3, cycle_patience=5, step_size_up=2000, step_size_down=2000, dump_path=dump_path)
    
elif args["model"] == "NN":
    model = models.NeuralSiameseModel(8, CLASSES, 64, 64).to(config.device)
    train.train_full(model, loader, val_loader, lr=1e-3, weight_decay=1e-3, cycle_patience=5, step_size_up=2000, step_size_down=2000, dump_path=dump_path)
    
elif args["model"] == "Norm":
    model = models.NormGEDModel(8, CLASSES, 64, 64).to(config.device)
    model.weighted = True
    train.train_full(model, loader, val_loader, lr=1e-3, weight_decay=1e-3, cycle_patience=5, step_size_up=2000, step_size_down=2000, dump_path=dump_path)
    
else:
    raise TypeError("Invalid Model Type")

print(time.time() - start_time)

