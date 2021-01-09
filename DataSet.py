import numpy as np


class DataSet:
    def __init__(self, features: np.ndarray, labels: np.ndarray, preprocess: bool, bias=False):
        self.features = features
        self.labels = labels
        if bias:
            self.addBias()
        self.n = self.features.shape[1]  # Number of features with bias included
        self.m = self.features.shape[0]
        self.normFeatures = None
        self.normLabels = None
        if preprocess:
            self.preProcess()
            self.normFeatures = np.array(self.normFeatures)
            self.normLabels = np.array(self.normLabels)
            self.applyNormalization(self.normFeatures, self.normLabels)

    def addBias(self):
        self.features = np.insert(self.features, 0, 1, 1)  # Adding bias

    def __getitem__(self, i: int):
        '''
        returns list [x, y]
        x is nparray, features of example i with bias included
        y is float value, the label of example i
        '''
        return [self.features[i], self.labels[i]]

    def preProcess(self):
        self.normFeatures = []
        self.normFeatures.append([0, 1])  # Normalization for bias

        for i in range(1, self.n):
            mx = np.amax(self.features, 0)[i]
            mn = np.amin(self.features, 0)[i]
            r = mx - mn

            self.normFeatures.append([mn, r])

        mx = np.amax(self.labels)
        mn = np.amin(self.labels)
        r = mx - mn

        self.normLabels = [mn, r]

    def mergeFeatureAndLabel(self):
        newLabel = np.reshape(self.labels, (self.labels.shape[0], 1))

        wholeData = np.hstack((self.features, newLabel))
        self.wholeData = wholeData

    def applyNormalization(self, F, L):
        for i in range(1, self.n):
            if (F[i][1] > 0):
                self.features[:, i] -= F[i][0]
                self.features[:, i] /= F[i][1]

        if (L[1] > 0):
            self.labels -= L[0]
            self.labels /= L[1]
