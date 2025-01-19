import os

datasets = [
    "AIDS700nef",
    "LINUX",
    "IMDBMulti"
]

split_line = lambda x: list(map(lambda y: int(y.strip()), x.split()))

for dataset in datasets:
    origin_data_folder = "./datasets/{}/".format(dataset)
    target_data_foler = "./data/TUDatasets/{}/".format(dataset)
    
    if not os.path.exists(target_data_foler):
        os.system("mkdir {}".format(target_data_foler))
        
    num_train_graphs = len(os.listdir(origin_data_folder + "train/"))
    num_test_graphs = len(os.listdir(origin_data_folder + "test/"))
    
    filenames = [
        "{}_A.txt".format(dataset),
        "{}_graph_indicator.txt".format(dataset),
        "{}_graph_labels.txt".format(dataset),
        "{}_node_labels.txt".format(dataset),
    ]
    files = []
    
    for filename in filenames:
        files.append(open(target_data_foler + filename, "w"))
        
    current_node_id = 0
    for i in range(num_train_graphs + num_test_graphs):
        files[2].write("1\n")
        if i < num_train_graphs:    
            f = open(origin_data_folder + "train/" + str(i), "r")
        else:
            f = open(origin_data_folder + "test/" + str(i - num_train_graphs), "r")
        n, m, features = split_line(f.readline())

        for _ in range(n):
            x, y = split_line(f.readline())
            files[1].write("{}\n".format(i + 1))
            files[3].write("{}\n".format(y))

        for _ in range(m):
            x, y = split_line(f.readline())
            files[0].write("{}, {}\n".format(x + current_node_id + 1,
                                           y + current_node_id + 1))
            

        current_node_id += n
        f.close()
        

        
        
    for i in range(len(filenames)):
        files[i].close()
    
    