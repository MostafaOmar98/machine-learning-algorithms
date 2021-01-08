import pandas as pd
from DataSet import DataSet


class Data:
    training = None
    testing = None

    def __init__(self, path, featureCols, labelCol, trainSize,preProc=False, addbias =False):
        '''
        path is absolute
        path is a csv file
        '''
        self.data = pd.read_csv(filepath_or_buffer=path, usecols=featureCols + labelCol)
        self.data = self.data.sample(frac=1).reset_index(drop=True)                            # frac=1 is taking all the dataset
        if preProc:
            self.features = self.data[featureCols].to_numpy(dtype='float64')
            self.labels = self.data[labelCol].to_numpy(dtype='float64')
        else:
            self.features = self.data[featureCols].to_numpy()
            self.labels = self.data[labelCol].to_numpy()
        self.labels = self.labels.flatten()
        self.n = self.features.shape[1]  # Number of features with bias included
        self.m = self.features.shape[0]
        self.training = DataSet(self.features[:int(trainSize * self.m)], self.labels[:int(trainSize * self.m)], preProc,addbias)
        self.testing = DataSet(self.features[int(trainSize * self.m):],self.labels[int(trainSize * self.m):], False, addbias)
        self.testing.applyNormalization(self.training.normFeatures, self.training.normLabels)
