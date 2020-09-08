import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

train = pd.read_csv(sys.argv[1], encoding='big5')

data = train.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# prev_day: 取前幾小時的data, feature: 選取的feature, column: feature的數目
prev_day = 9
#feature = [9]
feature = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
column = len(feature)
x = np.empty([12 * (480-prev_day), column * prev_day], dtype = float)
y = np.empty([12 * (480-prev_day), 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 23-prev_day:
                continue
            x[month * (480-prev_day) + day * 24 + hour, :] = month_data[month][feature,day * 24 + hour : day * 24 + hour + prev_day].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * (480-prev_day) + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + prev_day] #value

mean_x = np.mean(x, axis = 0) #column * prev_day 
std_x = np.std(x, axis = 0) #column * prev_day
for i in range(len(x)): #12 * (480 - prev_day)
    for j in range(len(x[0])): #column * prev_day 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

np.save('hw1_meanx.npy', mean_x)
np.save('hw1_stdx.npy', std_x)
x = np.concatenate((np.ones([12 * (480-prev_day), 1]), x), axis = 1).astype(float)

import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

train_loss = []
val_loss = []
step = []
dim = column * prev_day + 1
w = np.zeros([dim, 1])
learning_rate = 1
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/len(x_train_set))#rmse
    val_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/len(x_validation))#rmse
    if(t%100==0):
        print('Train '+ str(t) + ":" + str(loss))
        print('Val '+ str(t) + ":" + str(val_loss))
        train_loss.append(loss)
        step.append(t)
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

np.save('hw1_weight.npy', w)
