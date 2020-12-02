import pandas as pd

class Dataset:
    def __init__(self, path, usecols):
        '''
        path is absolute
        path is a csv file
        '''
        self.df = pd.read_csv(filepath_or_buffer=path, usecols=usecols)
    def __getitem__(self, i):
        '''
        returns feature vector of example i in numpy format
        '''
        return self.df.iloc[i].to_numpy()