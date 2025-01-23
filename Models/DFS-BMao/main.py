from src.train import Trainer
from src.utils import get_hyper
import torch

def main():
    args = get_hyper("./hyperparameters.json")
    trainer = Trainer(args)
    trainer.test()
    print("mse_exp(10^-3): {:.3f}".format(trainer.train_mse_exp * 1000))
    print("mse_linear(10^-3): {:.3f}".format(trainer.train_mse_linear * 1000))
    print("mae: {:.3f}".format(trainer.train_model_mae))
    print("Spearman's rho: {:.3f}".format(trainer.train_rho))
    print("Kendall's tau: {:.3f}".format(trainer.train_tau))
    print("p@10: {:.3f}".format(trainer.train_prec_at_10))
    print("p@20: {:.3f}".format(trainer.train_prec_at_20))
    if trainer.ged_flag:
        print("test_mse_exp(10^-3): {:.3f}".format(trainer.test_mse_exp * 1000))
        print("test_mse_linear(10^-3): {:.3f}".format(trainer.test_mse_linear * 1000))
        print("test_mae: {:.3f}".format(trainer.test_model_mae))
        print("test_Spearman's rho: {:.3f}".format(trainer.test_rho))
        print("test_Kendall's tau: {:.3f}".format(trainer.test_tau))
        print("test_p@10: {:.3f}".format(trainer.test_prec_at_10))
        print("test_p@20: {:.3f}".format(trainer.test_prec_at_20))
    print("total_train_test_time: {:.3f}".format(trainer.train_test_time))
    print("total_test_test_time: {:.3f}".format(trainer.test_test_time))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()