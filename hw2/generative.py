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

def Drop(df1,df2,name, test=True):
    df1 = df1.drop(name, axis=1)
    if test==True:
        df2 = df2.drop(name, axis=1)
    return df1, df2

def Map(name, key = ['N', 'Y'], value = [0, 1], test=True):
    if test == True:
        test_data[name] = test_data[name].map(dict(zip(key, value)))
    train_data[name] = train_data[name].map(dict(zip(key, value)))


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

train_data = pd.read_csv(sys.argv[1])
test_data = pd.read_csv(sys.argv[2])

Map('y', [train_data.iloc[1, -1], ' 50000+.'], [0.0, 1.0], test=False)
Map('year', [94, 95], ['young', 'old'])
Map('veterans benefits', [0, 1, 2], ['low', 'medium', 'high'])
Map('own business or self employed', [0, 1, 2], ['low', 'medium', 'high'])
train_data, test_data = Drop(train_data, test_data, 'country of birth mother')
train_data, test_data = Drop(train_data, test_data, 'country of birth father')
train_data, test_data = Drop(train_data, test_data, 'id')

train_data['detailed industry recode'] = train_data['detailed industry recode'].astype(str) 
train_data['detailed occupation recode'] = train_data['detailed occupation recode'].astype(str) 
test_data['detailed industry recode'] = test_data['detailed industry recode'].astype(str) 
test_data['detailed occupation recode'] = test_data['detailed occupation recode'].astype(str)

X_train, X_test = Drop(train_data, test_data, 'y', test=False)
X = pd.concat([X_train, X_test])
X = pd.get_dummies(X)
X_train = X.iloc[:len(X_train), :]
X_test = X.iloc[len(X_train):, :]

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

X_train['capital gains'] = X_train['capital gains'] // 3000
X_test['capital gains'] = X_test['capital gains'] // 3000

# classification_6
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

data_dim = X_train.shape[1]
# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)  

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

# Predict testing labels
predictions = 1 - _predict(X_test, w, b)
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))


