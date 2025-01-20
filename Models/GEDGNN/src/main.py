from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
from utils import load_ged, load_all_graphs
import time

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    
    
    # train_num, val_num, test_num, graphs = load_all_graphs(args.abs_path, args.dataset)

    # gid = [g['gid'] for g in graphs]
    # ged_dict = dict()
    # load_ged(ged_dict, args.abs_path, args.dataset, 'TaGED.json')
    # n = len(graphs)
    # ans = []
    # for i in range(n):
    #     ans.append([])
    #     for j in range(n):
    #         id_pair = (gid[i], gid[j])
    #         if not id_pair in ged_dict:
    #             id_pair = (gid[j], gid[i])
    #             if not id_pair in ged_dict:
    #                 ans[-1].append(None)
    #                 continue
    #         ans[-1].append(ged_dict[id_pair][0][0])
        
    # for i in ans:
    #     print(i)
    # exit()

    for i in range(3):
        args.idx = i
        print("=========================================")
        trainer = Trainer(args)

        if args.model_epoch_start > 0:
            trainer.load(args.model_epoch_start)

        if args.model_train == 1:
            for epoch in range(args.model_epoch_start, args.model_epoch_end):
                trainer.cur_epoch = epoch
                start_time = time.time()
                trainer.fit()
                print(time.time() - start_time)
                # trainer.save(epoch + 1)
                #trainer.score('val')
                # trainer.score('test')
                #if not args.demo:
                #   trainer.score('test2')
        else:
            trainer.cur_epoch = args.model_epoch_start
            trainer.score("test")
            # trainer.score("test2")
            # trainer.score('test', test_k=100)
            # trainer.score('test2', test_k=100)
            # trainer.batch_score('test', test_k=100)
            """
            test_matching = True
            trainer.cur_epoch = args.model_epoch_start
            #trainer.score('val', test_matching=test_matching)
            trainer.score('test', test_matching=test_matching)
            #if not args.demo:
            #   trainer.score('test2')
            """

if __name__ == "__main__":
    main()
