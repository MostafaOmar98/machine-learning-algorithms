import numpy as np

import DataSet

TRAIN_PATH = './assets/heart.csv'
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
featureCols = ['oldpeak']
labelCol = ['target']
alpha = 0.1
MAX_ITERATIONS = 2000
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
