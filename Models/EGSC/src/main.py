from utils import tab_printer
from egsc import EGSCTrainer
from parser import parameter_parser


def main():
    args = parameter_parser()
    tab_printer(args)
    
    for i in range(10):
        args.idx = i
        trainer = EGSCTrainer(args)
        
        count = 0
        for g in trainer.testing_graphs:
            for g1 in trainer.training_graphs:
                if int(trainer.ged_matrix[g["i"], g1["i"]]) == -1:
                    continue
                count += 1
        print(count)
        exit()
        # trainer.fit()
        # trainer.save_model()
        # trainer.score()
        # exit()
        if args.dataset in ["AIDS700nef", "LINUX", "new_IMDB"]:
            trainer.score(test_type="testtest")
        # trainer.save_model()
        
    
    if args.notify:
        import os
        import sys
        if sys.platform == 'linux':
            os.system('notify-send EGSC "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "EGSC" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No notification support for this OS.')


if __name__ == "__main__":
    main()
