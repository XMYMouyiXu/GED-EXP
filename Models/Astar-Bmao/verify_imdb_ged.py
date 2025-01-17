import os



dataset = "AIDS700nef"
root_folder = "geddataset/{}/{}/"

num_train_graphs = len(os.listdir(root_folder.format(dataset, "train")))
num_test_graphs = len(os.listdir(root_folder.format(dataset, "test")))

f = open("geddataset/IMDBMulti/ged", "r")

all = [list(map(int, temp.strip().split())) for temp in f.readlines()]

f.close()


all_count = 0
count = 0
for i in range(num_train_graphs):
    for j in range(num_test_graphs):
        if all[i][num_train_graphs + j] < 0:
            count += 1
        all_count += 1
        
print(count, all_count, count / all_count)

