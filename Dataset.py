import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path, featureCols, labelCol, doPre, fromBegin, examplesPercentage):
        '''
        path is absolute
        path is a csv file
        '''
        self.features = pd.read_csv(filepath_or_buffer=path, usecols=featureCols).to_numpy(dtype='float64')
        self.labels = pd.read_csv(filepath_or_buffer=path, usecols=labelCol).to_numpy(dtype='float64')
        self.labels = self.labels.flatten()
        self.n = self.features.shape[1] + 1  # Number of features with bias included
        self.m = self.features.shape[0]

        if doPre:
            self.preProcess()

        if fromBegin == True:
            self.features = self.features[:int(examplesPercentage * self.m)]
            self.labels = self.labels[:int(examplesPercentage * self.m)]
        else:
            self.features = self.features[int((1-examplesPercentage)*self.m):]
            self.labels = self.labels[int((1-examplesPercentage)*self.m):]

        self.m = int(examplesPercentage * self.m)

    def __getitem__(self, i):
        '''
        returns list [x, y]
        x is nparray, features of example i with bias included
        y is float value, the label of example i
        '''
        return [self.features[i], self.labels[i]]

    def preProcess(self):
        for i in range(self.n - 1):
            r = np.amax(self.features, 0)[i] - np.amin(self.features, 0)[i]
            self.features[:, i] -= np.amin(self.features, 0)[i]
            self.features[:, i] /= r
        self.features = np.insert(self.features, 0, 1, 1)  # Adding bias

        r = np.amax(self.labels) - np.amin(self.labels)
        self.labels -= np.amin(self.labels)
        self.labels /= r
