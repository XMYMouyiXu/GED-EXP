import os

datasets = [
    "AIDS700nef",
    "LINUX",
    "IMDBMulti"
]

split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))

for dataset in datasets:
    origin_data_folder = "./datasets/{}/".format(dataset)
    target_data_foler = "./my_results/{}_pairs.txt".format(dataset)
    
    # if not os.path.exists(target_data_foler):
    #     os.system("mkdir {}".format(target_data_foler))
        
    num_train_graphs = len(os.listdir(origin_data_folder + "train/"))
    num_test_graphs = len(os.listdir(origin_data_folder + "test/"))
    
    num_graphs = num_train_graphs + num_test_graphs
    
    f = open(target_data_foler, "w")
    
    for i in range(num_graphs):
        for j in range(i, num_graphs):
            f.write("{} {}\n".format(i, j))
    
    f.close()
    