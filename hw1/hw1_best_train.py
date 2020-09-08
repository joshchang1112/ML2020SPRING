import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import math

train = pd.read_csv(sys.argv[1], encoding='big5')
test = pd.read_csv(sys.argv[2], header=None, encoding='big5')

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

train = train.iloc[:, 3:]
train[train == 'NR'] = 0
raw_data = train.to_numpy()

# 插入新feature(和前一天pm2.5的差值)
tmp = np.zeros([240, 24])
for i in range(240):
    for j in range(24):
        if i % 20 ==0 and j == 0:
            tmp[i, j] = 0
        else:
            if j == 0:
                tmp[i, j] = float(raw_data[i * 18 + 9, j]) - float(raw_data[(i-1) * 18 + 9, 23])
            else:
                tmp[i, j] = float(raw_data[i * 18 + 9, j]) - float(raw_data[i * 18 + 9, j-1])

for i in range(240):
   raw_data = np.insert(raw_data, 18 * (i+1) + i, tmp[i, :],0)
print(raw_data.shape)

# 插入新feature(和前一天O^3的差值)
tmp = np.zeros([240, 24])
for i in range(240):
    for j in range(24):
        if i % 20 ==0 and j == 0:
            tmp[i, j] = 0
        else:
            if j == 0:
                tmp[i, j] = float(raw_data[i * 19 + 7, j]) - float(raw_data[(i-1) * 19 + 7, 23])
            else:
                tmp[i, j] = float(raw_data[i * 19 + 7, j]) - float(raw_data[i * 19 + 7, j-1])

for i in range(240):
    raw_data = np.insert(raw_data, 19 * (i+1) + i, tmp[i, :],0)

# 將NOx = NO + NO2
for i in range(240):
    for j in range(24):
        raw_data[i * 20 + 6, j] = float(raw_data[i * 20 + 4, j]) + float(raw_data[i * 20 + 5, j])

total_feature = 20
month_data = {}
for month in range(12):
    sample = np.empty([total_feature, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[total_feature * (20 * month + day) : total_feature * (20 * month + day + 1), :]
    month_data[month] = sample
#print(month_data[0])

prev_day = 7
feature = [ 1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12,13, 18, 19]
column = len(feature)

x = np.empty([12 * (480-prev_day), column * prev_day], dtype = float)
y = np.empty([12 * (480-prev_day), 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 23-prev_day:
                continue
            x[month * (480-prev_day) + day * 24 + hour, :] = month_data[month][feature,day * 24 + hour : day * 24 + hour + prev_day].reshape(1, -1)
            y[month * (480-prev_day) + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + prev_day]

### Clean Data(pm2.5 = -1)###
check = 0
while check != 1:
    for i in range(len(y)):
        if y[i] == -1:
            y = np.delete(y, i, 0)
            x = np.delete(x, i, 0)
            break

        if i == len(y) - 1:
            check = 1
print(len(y))

check = 0
while check != 1:
    for i in range(len(y)):
        if x[i][56] == -1 or x[i][57] == -1 or x[i][58] == -1 or x[i][59] == -1 or x[i][60] == -1 or x[i][61] == -1 or x[i][62] == -1:
            y = np.delete(y, i, 0)
            x = np.delete(x, i, 0)
            break

        if i == len(y) - 1:
            check = 1

### Clean Data(pm10 = 0)###
check = 0
while check != 1:
    for i in range(len(y)):
        if x[i][49] == 0 or x[i][50] == 0 or x[i][51] == 0 or x[i][52] == 0 or x[i][53] == 0 or x[i][54] == 0 or x[i][55] == 0:
            y = np.delete(y, i, 0)
            x = np.delete(x, i, 0)
            break

        if i == len(y) - 1:
            check = 1
print(len(y))

### Delete repeat data ###
repeat_data = []
for i in range(len(y)):
    tmp2 = np.zeros([prev_day, column])
    for j in range(len(x[0])):
       tmp2[j%prev_day, j//prev_day] = x[i][j]
    if np.array_equal(tmp2[0], tmp2[1]):
        if len(repeat_data) == 0:
             repeat_data.append(tmp2[0])
        else:
            for j in range(len(repeat_data)):              
                if np.array_equal(tmp2[0], repeat_data[j]):
                    break
                if j == len(repeat_data)-1:
                    repeat_data.append(tmp2[0])
#print(repeat_data)

delete_index = []
        
for i in range(len(y)):
    tmp2 = np.zeros([prev_day, column])
    for j in range(len(x[0])):
        tmp2[j%prev_day, j//prev_day] = x[i][j]
    for j in range(len(tmp2)):
        for k in range(len(repeat_data)):
            if np.array_equal(tmp2[j], repeat_data[k]):
                delete_index.append(i)
                print(i)

y = np.delete(y, delete_index, 0)
x = np.delete(x, delete_index, 0)
print(len(y))


test_data = test.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, column*prev_day], dtype = float)
train_test_x = np.empty([240*2, column*prev_day], dtype = float)
train_test_y = np.empty([240*2, 1], dtype = float)

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

for i in range(240):
    train_test_x[i*2, :] = test_data[list_add([total_feature * i] * column, feature), :prev_day].reshape(1, -1)
    train_test_y[i*2, 0] = test_data[i*total_feature+9, 7]
    train_test_x[i*2+1, :] = test_data[list_add([total_feature * i] * column, feature), 1:prev_day+1].reshape(1, -1)
    train_test_y[i*2+1, 0] = test_data[i*total_feature+9, 8]
    

# Normalize
#mean_x = np.mean(x, axis = 0) #18 * 9 
#std_x = np.std(x, axis = 0) #18 * 9 
mean_x = np.zeros([column])
for i in range(len(x)):
    for j in range(len(x[0])):
        mean_x[j//prev_day] += x[i][j] / (prev_day*(len(x)+len(train_test_x)))
                                          
for i in range(len(train_test_x)):
    for j in range(len(train_test_x[0])):
        mean_x[j//prev_day] += train_test_x[i][j] / (prev_day*(len(x)+len(train_test_x)))
                                          
std_x = np.zeros([column])
for i in range(len(x)):
    for j in range(len(x[0])):
        std_x[j//prev_day] += (x[i][j] - mean_x[j//prev_day])**2 / (prev_day*(len(x)+len(train_test_x)))
for i in range(len(train_test_x)):
    for j in range(len(train_test_x[0])):
        std_x[j//prev_day] += (train_test_x[i][j] - mean_x[j//prev_day])**2 / (prev_day*(len(x)+len(train_test_x)))

std_x = np.sqrt(std_x)

for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j//prev_day] != 0:
            x[i][j] = (x[i][j] - mean_x[j//prev_day]) / std_x[j//prev_day]

for i in range(len(train_test_x)): #12 * 471
    for j in range(len(train_test_x[0])): #18 * 9 
        if std_x[j//prev_day] != 0:
            train_test_x[i][j] = (train_test_x[i][j] - mean_x[j//prev_day]) / std_x[j//prev_day]

np.save('hw1_best_meanx.npy', mean_x)
np.save('hw1_best_stdx.npy', std_x)

# initialize
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
train_test_x = np.concatenate((np.ones([240*2, 1]), train_test_x), axis = 1).astype(float)
dim = column * prev_day + 1
x = np.concatenate((np.ones([len(y), 1]), x), axis = 1).astype(float)
learning_rate = 0.9
iter_time = 10001
eps = 0.0000000001
lamba = 0
avg_train_loss = 0
avg_val_loss = 0
split=5
kf = KFold(n_splits=split)
count = 0
model_num = 0

for train_index, test_index in kf.split(x):
    w = np.zeros([dim, 1])
    
    #print(w.shape)
    adagrad = np.zeros([dim, 1])
    min_loss = 100
    early_stop = 0
    x_train, x_valid = x[train_index], x[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    x_train = np.concatenate((x_train, train_test_x), axis = 0).astype(float)
    y_train = np.concatenate((y_train, train_test_y), axis = 0).astype(float)
    print(x_train.shape)
    #w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y_train)
    for t in range(iter_time):
    #for t in range(step[count]+1):
        #loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2) ) / len(x))#rmse
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2) ) / len(x_train))#rmse
        val_loss = np.sqrt(np.sum(np.power(np.dot(x_valid, w) - y_valid, 2))/ len(x_valid))
        if(t%1000==0):
            print("Train: " + str(t) + ":" + str(loss))
            print("Valid: " + str(t) + ":" + str(val_loss))
        #gradient1 = 2 * np.dot(x.transpose(), np.dot(x, w) - y) + 2 * lamba * w
        gradient1 = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train) + 2 * lamba * w
        adagrad += gradient1 ** 2
        w = w - learning_rate * gradient1 / np.sqrt(adagrad + eps)
        
        if val_loss < min_loss:
            min_loss = val_loss
            early_stop = 0
        else:
            early_stop += 1
        

        if t == iter_time - 1:
        #if t == step[count] - 1:
            print("Step " + str(t) + ':' + str(val_loss))
            avg_train_loss += loss/split
            avg_val_loss += val_loss/split
            best_step = t
        
        
    ans_y = np.dot(test_x, w)
    count += 1

print(avg_train_loss)
print(avg_val_loss)
np.save('hw1_best_weight.npy', w)

