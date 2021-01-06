import pandas as pd
from DataSet import DataSet


class Data:
    training = None
    testing = None

    def __init__(self, path, featureCols, labelCol, trainSize):
        '''
        path is absolute
        path is a csv file
        '''
        self.data = pd.read_csv(filepath_or_buffer=path,usecols=featureCols+labelCol)
        self.data.sample(frac=1)                                                # frac=1 is taking all the dataset
        self.features = self.data[featureCols].to_numpy(dtype='float64')
        self.labels = self.data[labelCol].to_numpy(dtype='float64')
        self.labels = self.labels.flatten()
        self.n = self.features.shape[1]  # Number of features with bias included
        self.m = self.features.shape[0]
        self.training = DataSet(self.features[:int(trainSize * self.m)], self.labels[:int(trainSize * self.m)], True,
                                True)
        self.testing = DataSet(self.features[int((1 - trainSize) * self.m):],
                               self.labels[int((1 - trainSize) * self.m):], True, True)
