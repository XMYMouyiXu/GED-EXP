from src.train import Trainer
from src.utils import get_hyper
import torch
import numpy as np

from torch_geometric.datasets import GEDDataset

def main():
    args = get_hyper("./hyperparameters.json")
    # filename_best_model = "./models/{}/best_model/0_model_{}_{}_{}.pt".format(args["model"], args["dataset"], args["normalization"], args["batch_type"])
    # if args["model"] == "GENN-Astar":
    #     args["model"] = "GENN"
    #     filename_best_model = "./models/GENN/best_model/0_model_Astar_{}_{}_{}.pt".format(args["dataset"], args["normalization"], args["batch_type"])

    trainer = Trainer(args, "")
    # train_graphs = GEDDataset("./geddataset", name="AIDS700nef", train=True) 
    # ged = train_graphs.ged
    # norm_ged = train_graphs.norm_ged

    # l = len(train_graphs)
    # for i in range(l):
    #     for j in range(l):
    #         if trainer.sim_score[i][j] != norm_ged[i][j]:
    #             print(i, j, trainer.training_graph[j].num_nodes, train_graphs[j].num_nodes, trainer.sim_score[i][j], torch.exp(-norm_ged[i][j]))
    # return 
    all = {
        "mse(10^-3)": [],
        "mae": [],
        "Spearman's rho": [],
        "Kendall's tau": [],
        "p@10": [],
        "p@20": [],
        "test_mse(10^-3)": [],
        "test_mae": [],
        "test_Spearman's rho": [],
        "test_Kendall's tau": [],
        "test_p@10": [],
        "test_p@20": [],
        "total_train_time" : [],
        "avg_train_time": [],
        "total_val_time": [],
        "avg_val_time": [],
        "total_test_time": [],
        "total_test_test_time": [],
    }
    for idx in range(max(1, args["num_test"])):
        filename_best_model = "./models/{}/best_model/{}_model_{}_{}_{}_original.pt".format(args["model"], idx, args["dataset"], args["normalization"], args["batch_type"])
        if args["model"] == "GENN-Astar":
            # args["model"] = "GENN-Astar"
            filename_best_model = "./models/GENN/best_model/{}_model_{}_{}_{}.pt".format(idx, args["dataset"], args["normalization"], args["batch_type"])
            args["model"] = "GENN-Astar"
        elif args["model"] == "GENN":
            filename_best_model = "./models/GENN/best_model/{}_model_{}_{}_{}_no_astar.pt".format(idx, args["dataset"], args["normalization"], args["batch_type"])

        trainer.filename_best_model = filename_best_model
        #trainer.filename_best_model = "./models/{}/best_model/AIDS_20_linear".format(args["model"])

        if args["train"]:
            trainer.create_model()
            trainer.train()
        trainer.test()
        all["mse(10^-3)"].append(trainer.model_error * 1000)
        all["mae"].append(trainer.model_mae) 
        all["Spearman's rho"].append(trainer.rho)
        all["Kendall's tau"].append(trainer.tau)
        all["p@10"].append(trainer.prec_at_10)
        all["p@20"].append(trainer.prec_at_20)
        all["total_train_time"].append(trainer.total_training_time)
        all["total_val_time"].append(trainer.total_validation_time)
        all["total_test_time"].append(trainer.testing_time)
        all["avg_train_time"].append(trainer.avg_train_time)
        all["avg_val_time"].append(trainer.avg_val_time)
        # print(trainer.avg_val_time)
        # if args["dataset"] != "IMDBMulti" and args["model"] != "BMao":
        all["test_mse(10^-3)"].append(trainer.test_model_error * 1000)
        all["test_mae"].append(trainer.test_model_mae)
        all["test_Spearman's rho"].append(trainer.test_rho)
        all["test_Kendall's tau"].append(trainer.test_tau)
        all["test_p@10"].append(trainer.test_prec_at_10)
        all["test_p@20"].append(trainer.test_prec_at_20)
        all["total_test_test_time"].append(trainer.test_test_time)

    print("=======================================================")
    print("Metric\tMean\tstd")
    print("train-test set")
    print("mse(10^-3): {} \n{}".format(round(np.mean(all["mse(10^-3)"]), 5), round(np.std(all["mse(10^-3)"]), 5)))
    print("mae: {} \n{}".format(round(np.mean(all["mae"]), 5), round(np.std(all["mae"]), 5)))
    print("Spearman's rho: {} \n{}".format(round(np.mean(all["Spearman's rho"]), 5), round(np.std(all["Spearman's rho"]), 5)))
    print("Kendall's tau: {} \n{}".format(round(np.mean(all["Kendall's tau"]), 5), round(np.std(all["Kendall's tau"]), 5)))
    print("p@10: {} \n{}".format(round(np.mean(all["p@10"]), 5), round(np.std(all["p@10"]), 5)))
    print("p@20: {} \n{}".format(round(np.mean(all["p@20"]), 5), round(np.std(all["p@20"]), 5)))
    print("Total training time: {}s\nTraining time per epoch: {}s".format(np.mean(all["total_train_time"]), np.mean(all["avg_train_time"])))
    print("Total validation time: {}s\nvalidation time per 10 epochs: {}s".format(np.mean(all["total_val_time"]), np.mean(all["avg_val_time"])))
    print("Train-test time: {}s\nTest-test time: {}s".format(np.mean(all["total_test_time"]), np.mean(all["total_test_test_time"])))

    # if args["dataset"] != "IMDBMulti" and args["model"] != "BMao":
    print("test-test set")
    print("mse(10^-3): {} \n{}".format(round(np.mean(all["test_mse(10^-3)"]), 5), round(np.std(all["test_mse(10^-3)"]), 5)))
    print("mae: {} \n{}".format(round(np.mean(all["test_mae"]), 5), round(np.std(all["test_mae"]), 5)))
    print("Spearman's rho: {} \n{}".format(round(np.mean(all["test_Spearman's rho"]), 5), round(np.std(all["test_Spearman's rho"]), 5)))
    print("Kendall's tau: {} \n{}".format(round(np.mean(all["test_Kendall's tau"]), 5), round(np.std(all["test_Kendall's tau"]), 5)))
    print("p@10: {} \n{}".format(round(np.mean(all["test_p@10"]), 5), round(np.std(all["test_p@10"]), 5)))
    print("p@20: {} \n{}".format(round(np.mean(all["test_p@20"]), 5), round(np.std(all["test_p@20"]), 5)))



if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()