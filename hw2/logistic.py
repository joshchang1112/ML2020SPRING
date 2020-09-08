import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

np.random.seed(0)
def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25, split = 5):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    if split == 5:
        return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
    elif split == 1:
        return X[len(X)-train_size:], Y[len(X)-train_size:], X[:len(X)-train_size], Y[:len(X)-train_size]
    elif split == 2:
        return np.concatenate([X[:len(X)-train_size], X[(len(X)-train_size)*2:]], axis=0), np.concatenate([Y[:len(X)-train_size], Y[(len(X)-train_size)*2:]], axis=0), X[len(X)-train_size:(len(X)-train_size)*2], Y[len(X)-train_size:(len(X)-train_size)*2]
    elif split == 3:
        return np.concatenate([X[:(len(X)-train_size)*2], X[(len(X)-train_size)*3:]], axis=0), np.concatenate([Y[:(len(X)-train_size)*2], Y[(len(X)-train_size)*3:]], axis=0), X[(len(X)-train_size)*2:(len(X)-train_size)*3], Y[(len(X)-train_size)*2:(len(X)-train_size)*3]
    elif split == 4:
        return np.concatenate([X[:(len(X)-train_size)*3], X[(len(X)-train_size)*4:]], axis=0), np.concatenate([Y[:(len(X)-train_size)*3], Y[(len(X)-train_size)*4:]], axis=0), X[(len(X)-train_size)*3:(len(X)-train_size)*4], Y[(len(X)-train_size)*3:(len(X)-train_size)*4]


def Map(name, key = ['N', 'Y'], value = [0, 1], test=True):
    if test == True:
        test_data[name] = test_data[name].map(dict(zip(key, value)))
    train_data[name] = train_data[name].map(dict(zip(key, value)))

def Drop(df1,df2,name, test=True):
    df1 = df1.drop(name, axis=1)
    if test==True:
        df2 = df2.drop(name, axis=1)
    return df1, df2

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1) + 2 * w * 0.0001 
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

# Numerical to Categorical data
Map('y', [train_data.iloc[1, -1], ' 50000+.'], [0.0, 1.0], test=False)
Map('year', [94, 95], ['young', 'old'])
Map('veterans benefits', [0, 1, 2], ['low', 'medium', 'high'])
Map('own business or self employed', [0, 1, 2], ['low', 'medium', 'high'])

# Drop feature
train_data, test_data = Drop(train_data, test_data, 'country of birth mother')
train_data, test_data = Drop(train_data, test_data, 'country of birth father')
train_data, test_data = Drop(train_data, test_data, 'id')
train_data, test_data = Drop(train_data, test_data, 'state of previous residence')

train_data['detailed industry recode'] = train_data['detailed industry recode'].astype(str) 
train_data['detailed occupation recode'] = train_data['detailed occupation recode'].astype(str) 
test_data['detailed industry recode'] = test_data['detailed industry recode'].astype(str) 
test_data['detailed occupation recode'] = test_data['detailed occupation recode'].astype(str)

# One hot encoding
X_train, X_test = Drop(train_data, test_data, 'y', test=False)
X = pd.concat([X_train, X_test])
X = pd.get_dummies(X)
X_train = X.iloc[:len(X_train), :]
X_test = X.iloc[len(X_train):, :]

# Drop one hot feature
X_train, X_test = Drop(X_train, X_test, 'country of birth self_ ?')
X_train, X_test = Drop(X_train, X_test, 'migration code-change in msa_ ?')
X_train, X_test = Drop(X_train, X_test, 'migration prev res in sunbelt_ ?')
X_train, X_test = Drop(X_train, X_test, 'migration code-change in reg_ ?')
X_train, X_test = Drop(X_train, X_test, 'migration code-move within reg_ ?')

X_train, X_test = Drop(X_train, X_test, 'detailed occupation recode_0')
X_train, X_test = Drop(X_train, X_test, 'detailed industry recode_0')
X_train, X_test = Drop(X_train, X_test, 'hispanic origin_ Do not know')
X_train, X_test = Drop(X_train, X_test, 'education_ Children')
X_train, X_test = Drop(X_train, X_test, 'class of worker_ Not in universe')

X_train, X_test = Drop(X_train, X_test, 'veterans benefits_medium')
X_train, X_test = Drop(X_train, X_test, "fill inc questionnaire for veteran's admin_ Not in universe")
X_train, X_test = Drop(X_train, X_test, 'country of birth self_ United-States')

# Feature engineering
X_train['capital gains'] = X_train['capital gains'] // 3000
X_test['capital gains'] = X_test['capital gains'] // 3000
X_train['wage per hour_2'] = X_train['wage per hour'] //1000
X_test['wage per hour_2'] = X_test['wage per hour'] //1000
X_train['age_2'] = X_train['age'] //5
X_test['age_2'] = X_test['age'] //5

X_train = X_train.astype(float)
X_train = X_train.values
X_test = X_test.astype(float)
X_test = X_test.values

X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

Y_train = train_data.iloc[:, -1]
Y_train = Y_train.values

dev_ratio = 0.2
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio, split=5)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# training
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

max_iter = 50
batch_size = 8
learning_rate = 0.18
step = 1
best_acc = -1
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]
        #print(Y)

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad 
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1

    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    #Y_train_pred = np.zeros([y_train_pred.shape[0]])
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
    if dev_acc[-1] > best_acc:
        print('EPOCH: {}'.format(epoch))
        print('Training loss: {}'.format(train_loss[-1]))
        print('Development loss: {}'.format(dev_loss[-1]))
        print('Training accuracy: {}'.format(train_acc[-1]))
        print('Development accuracy: {}'.format(dev_acc[-1]))
        best_acc = dev_acc[-1]
        best_w = w
        best_b = b

# Predict testing labels
predictions = _predict(X_test, best_w, best_b)
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
