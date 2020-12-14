TRAIN_PATH= 'assets/house_data.csv'  # relative path to the project directory /assets directory has the datasets
# featureCols=['sqft_living', 'grade', 'lat', 'view']
featureCols=['sqft_living']
labelCol=['price']
alpha=0.1
MAX_ITERATIONS = 100

import numpy as np

def h(x, c):
    '''
    hypothesis function
    :param x: numpy array representing feature vector
    :param c: numpy array representing weights
    :return: float value representing the prediction
    '''
    return c.dot(x)