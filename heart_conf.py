import math
import house_conf
import numpy as np

TRAIN_PATH = './assets/heart.csv'
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
featureCols = ['trestbps', 'chol', 'thalach', 'oldpeak']
labelCol = ['target']
alpha = 0.01
MAX_ITERATIONS = 100


def h(x, theta):
    """

    :param x: is the features vector
    :param theta: coefficient vector
    :return: value between 0-1
    """
    return 1 / (1 + math.exp(-1 * house_conf.h(x, theta)))


def cost(ds, h, c, m):
    ret = 0
    for [x, y] in ds:
        # y*(-log(h(x)))+(1-y)*(-log(1-h(h(x))))
        if (y == 1):
            ret += -1 * y * (np.log(h(x, c)))
        else:
            ret += -1 * (1 - y) * (np.log(1 - h(x, c)))
    ret /= (2 * m)
    return ret
