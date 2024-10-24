"""SimGNN runner."""

from utils import tab_printer
from simgnn import SimGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    args.training_graphs = "./dataset/{}".format(args.dataset)
    args.testing_graphs = "./dataset/{}/test".format(args.dataset)
    args.save_path = "./saved_models/2_{}".format(args.dataset)
    
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
        trainer.save()
    trainer.score()
    if args.dataset in ["AIDS700nef", "LINUX", "new_IMDB"]:
        trainer.score(test_type="testtest")


if __name__ == "__main__":
    main()
