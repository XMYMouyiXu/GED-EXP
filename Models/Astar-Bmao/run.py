import os
import time

dataset = "AIDS700nef"
root_folder = "geddataset/{}/{}/"
command = "timeout 100s ./ged -d graph1.txt -q graph2.txt -m pair -p astar -l BMao -g"

num_train_graphs = len(os.listdir(root_folder.format(dataset, "train")))
num_test_graphs = len(os.listdir(root_folder.format(dataset, "test")))


def rewrite_file(idx, filename, traintest, threshold):
    f_target = open(filename, "w")
    f_origin = open(root_folder.format(dataset, traintest) + str(idx), "r")
    
    temp = f_origin.readlines()
    r_line = lambda x : map(int, x.split())
    x, y, _ = r_line(temp[0])
    f_target.write("t # {}\n".format(idx + threshold))
    
    for i in range(1, x + 1):
        n, f = r_line(temp[i])
        f_target.write("v {} {}\n".format(n, f))
        
    for i in range(x + 1, x + y + 1):
        u, v = r_line(temp[i])
        if u > v:
            continue
        f_target.write("e {} {} 1\n".format(u, v))
    
    f_target.close()
    f_origin.close()

# num_train_graphs = len(os.listdir(root_folder.format(dataset, "train")))
# num_test_graphs = len(os.listdir(root_folder.format(dataset, "test")))

f = open("geddataset/{}/ged".format(dataset), "r")

all = [list(map(int, temp.strip().split())) for temp in f.readlines()]

f.close()


total_time = 0
count = 0
total_count = 0
for i in range(num_train_graphs):
    for j in range(num_test_graphs):
        if all[i][num_train_graphs + j] < 0:
            continue
        total_count += 1
        rewrite_file(i, "graph1.txt", "train", 0)
        rewrite_file(j, "graph2.txt", "test", num_train_graphs)
        start_time = time.time()
        os.system(command)
        temp_time = time.time() - start_time
        if temp_time > 100:
            count += 1
        total_time += temp_time
        

print(total_time)
print(count, total_count, count / total_count)