import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import DataSet

TRAIN_PATH = './assets/heart.csv'
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
featureCols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal']
labelCol = ['target']
alpha = 0.0001
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


def cost(x, y, w):
    return max(0, 1 - t(y) * h(x, w))


def predict(x, w):
    return tInv(h(x, w))


def isCorrect(x, y, w):
    return predict(x, w) == y


def deriv(x, y, w, lamda):
    if (isCorrect(x, y, w)):
        return 2 * lamda * w
    return 2 * lamda * w - t(y) * x


def get_accuracy(ds: DataSet, w):
    correct = 0
    for [x, y] in ds:
        correct += isCorrect(x, y, w)
    return correct / ds.m

def totalCost(ds, w):
    ret = 0
    for [x, y] in ds:
        ret += cost(x, y, w)
    ret /= ds.m
    return ret

def benchmark(data, withKernel: bool):
    # TODO: Remove this and remove includes for sklearn too
    x_train = data.training.features
    y_train = data.training.labels
    x_test = data.testing.features
    y_test = data.testing.labels

    model = None
    if (withKernel):
        model = SVC(kernel='linear')
    else:
        model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_pred, y_test)
