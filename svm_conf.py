import numpy as np

import DataSet

TRAIN_PATH = './assets/heart.csv'
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
# featureCols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal']
featureCols = ['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal']
# featureCols = ['age', 'cp', 'trestbps', 'chol',  'restecg', 'thalach', 'oldpeak', 'slope', 'ca','thal']
# featureCols = ['age', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'thal']
# featureCols = ['age', 'oldpeak', 'slope', 'thal']
# featureCols = ['trestbps', 'chol', 'thalach', 'oldpeak']
labelCol = ['target']
# alpha = 0.001  #too slow
# alpha = 0.1 # too much fluctuation
alpha = 0.01 # good
MAX_ITERATIONS = 500
preProc = True
addbias = True


def h(x, w):
    return np.dot(x, w)


def t(y):
    '''
    :param y: label
    :return: transformed label, instead of 0, 1. values will be -1, and 1
    '''
    if y == 0:
        return -1
    return 1


def tInv(y):
    '''
    :param y: label
    :return: predicted class. 1 for positive examples and 0 for negative examples
    '''
    if y >= 0:
        return 1
    return 0


def cost(x, y, w, lamda):
    ret = 0
    onlyWeight = w[1:]
    ret = lamda * np.linalg.norm(onlyWeight)/2
    ret += max(0, 1 - t(y) * h(x, w))
    return ret


def predict(x, w):
    return tInv(h(x, w))


def isCorrectStrict(x, y, w):
    return t(y) * h(x, w) >= 1


def deriv(x, y, w, lamda):
    if (isCorrectStrict(x, y, w)):
        return 2 * lamda * w
    return 2 * lamda * w - t(y) * x


def get_accuracy(ds: DataSet, w):
    correct = 0
    for [x, y] in ds:
        correct += predict(x, w) == y
    return correct / ds.m

def totalCost(ds, w, lamda):
    ret = 0
    for [x, y] in ds:
        ret += cost(x, y, w, lamda)
    ret /= ds.m
    return ret
