import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path, featureCols, labelCol):
        '''
        path is absolute
        path is a csv file
        '''
        self.features = pd.read_csv(filepath_or_buffer=path, usecols=featureCols).to_numpy(dtype='float64')
        self.labels = pd.read_csv(filepath_or_buffer=path, usecols=labelCol).to_numpy(dtype='float64')
        self.labels = self.labels.flatten()
        self.n = self.features.shape[1] + 1 # Number of features with bias included
        self.m = self.features.size

        self.preProcess()

    def __getitem__(self, i):
        '''
        returns list [x, y]
        x is nparray, features of example i with bias included
        y is float value, the label of example i
        '''
        return [self.features[i], self.labels[i]]

    def preProcess(self):
        for i in range(self.n - 1):
            sum = np.sum(self.features, 0)[i]
            mean = sum/self.m
            r = np.amax(self.features, 0)[i] - np.amin(self.features, 0)[i]
            self.features[:,i] -= mean
            self.features[:,i] /= r
        self.features = np.insert(self.features, 0, 1, 1) # Adding bias

        mean = np.sum(self.labels)/self.m
        r = np.amax(self.labels) - np.amin(self.labels)
        self.labels -= mean
        self.labels /= r
