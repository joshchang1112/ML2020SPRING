import sys
import pandas as pd
import numpy as np

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

total_feature = 20
prev_day = 7
feature = [ 1, 2, 3,4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 18, 19]
column = len(feature)

test = pd.read_csv(sys.argv[1], header=None, encoding='big5')
test_data = test.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, column*prev_day], dtype = float)

# 插入新feature
tmp = np.zeros([240, 9])
for i in range(240):
    for j in range(9):
        if j == 0:
            tmp[i, j] = 0
        else:
            tmp[i, j] = float(test_data[i * 18 + 9, j]) - float(test_data[i * 18 + 9, j-1])

for i in range(240):
   # print(i)
    test_data = np.insert(test_data, 18 * (i+1) + i, tmp[i, :], 0)

for i in range(240):
    for j in range(9):
        if j == 0:
            tmp[i, j] = 0
        else:
            tmp[i, j] = float(test_data[i * 19 + 7, j]) - float(test_data[i * 19 + 7, j-1])


for i in range(240):
    test_data = np.insert(test_data, 19 * (i+1) + i, tmp[i, :], 0)


for i in range(240):
    for j in range(9):
        test_data[i * 20 + 6, j] = float(test_data[i * 20 + 4, j]) + float(test_data[i * 20 + 5, j])


for i in range(240):
    test_x[i, :] = test_data[list_add([total_feature * i] * column, feature), (9-prev_day):].reshape(1, -1)

mean_x = np.load('hw1_best_meanx.npy')
std_x = np.load('hw1_best_stdx.npy')
# Normalize
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j//prev_day] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j//prev_day]) / std_x[j//prev_day]

test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
w = np.load('hw1_best_weight.npy')
ans_y = np.dot(test_x, w)
import csv
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i] < 0:
            ans_y[i] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
