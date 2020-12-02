import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, path, featureCols, labelCol):
        '''
        path is absolute
        path is a csv file
        '''
        self.features = pd.read_csv(filepath_or_buffer=path, usecols=featureCols)
        self.n = len(self.features.columns) + 1 # Number of features with bias included
        self.m = self.features.size
        self.labels = pd.read_csv(filepath_or_buffer=path, usecols=labelCol)

    def __getitem__(self, i):
        '''
        returns list [x, y]
        x is features of example i with bias included
        y is the label of example i
        '''
        x = self.features.iloc[i].to_numpy()
        np.insert(x, 0, 1) # insert bias in front
        y = self.labels.iloc[i]
        return [x, y]