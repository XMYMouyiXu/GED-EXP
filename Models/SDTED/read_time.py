import os

datasets = [
    "AIDS700nef",
    "LINUX",
    "IMDBMulti"
]

for dataset in datasets:
    print(dataset)
    origin_data_folder = "./datasets/{}/".format(dataset)
    # target_data_foler = "./my_results/{}_pairs.txt".format(dataset)
    
    num_train_graphs = len(os.listdir(origin_data_folder + "train/"))
    num_test_graphs = len(os.listdir(origin_data_folder + "test/"))
    
    num_graphs = num_train_graphs + num_test_graphs
    
    all_time = [[0] * num_graphs for _ in range(num_graphs)]

    with open("my_results/{}_time.txt".format(dataset), "r") as f:
        for temp in f.readlines():
            x, y, value = map(int, temp.strip().split())
            all_time[x][y] = value
            all_time[y][x] = value
            
    all = 0.0
    count = 0
    for i in range(num_train_graphs, num_graphs):

        for j in range(num_train_graphs):
            all += all_time[i][j]
            count += 1
        
    
    avg = all / count
    print("traintest")
    print("all: {:.5f}".format(all / 1000000000))
    print("avg: {:.5f}".format(avg / 1000000000))
        
    all = 0.0
    count = 0
    for i in range(num_train_graphs, num_graphs):
        for j in range(num_train_graphs, num_graphs):
            all += all_time[i][j]
            count += 1
        
    
    avg = all / count
    print("testtest")
    print("all: {:.5f}".format(all / 1000000000))
    print("avg: {:.5f}".format(avg / 1000000000))