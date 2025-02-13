import argparse
import pickle
import tagsim
import tagsim_nodeUnlabel
# # from tagsim import TaGSimTrainer
# from tagsim_nodeUnlabel import TaGSimTrainer

def main():

    parser = argparse.ArgumentParser(description="Run TaGSim.")
    parser.add_argument("--dataset", nargs="?", default="AIDS700nef", help="Dataset name")
    parser.add_argument("--training-graphs", nargs="?", default="./dataset/train/", help="Folder with training graph pair jsons.")
    parser.add_argument("--testing-graphs", nargs="?", default="./dataset/test/", help="Folder with testing graph pair jsons.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs. Default is 5.")
    parser.add_argument("--tensor-neurons", type=int, default=16, help="Neurons in tensor network layer. Default is 16.")
    parser.add_argument("--bottle-neck-neurons", type=int, default=16, help="Bottle neck layer neurons. Default is 16.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability. Default is 0.0.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay. Default is 0.")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    if args.dataset in ["AIDS700nef"]:
        trainer = tagsim.TaGSimTrainer(args)
    elif args.dataset in ["LINUX", "IMDB"]:
        trainer = tagsim_nodeUnlabel.TaGSimTrainer(args)
        
    for i in range(5):
        args.idx = i
        if not args.test:
            trainer.fit()
        trainer.test()



if __name__ == "__main__":
    main()
