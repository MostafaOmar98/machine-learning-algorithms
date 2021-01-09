import sys
from typing import Callable

import numpy as np

from DataSet import DataSet
from Node import Node


class DecisionTree:
    root: Node
    dataset: DataSet
    preProcess: Callable

    def __init__(self, dataset: DataSet, preprocess: Callable):
        self.dataset = dataset
        self.preProcess = preprocess
        self.runPre()
        root = Node()

    def train(self, dataset: np.ndarray, taken=[]):
        # base case
        ent = self.entropy(dataset)
        if ent == 0:
            return
        arr = dict()
        mn = sys.maxint
        mnindx = 0
        for featureIndx in range(0, dataset.shape[1]):
            if not (featureIndx in taken):
                gain = ent - ((splitted.shape[0] / dataset.shape[0]) * self.entropy(splitted)
                              for splitted in (self.filterOnFeature(dataset, featureIndx)))
                if mn > gain:
                    mn = gain
                    mnindx = featureIndx
        taken.append()

    def runPre(self):
        self.preProcess(self.dataset.features)
        self.dataset.mergeFeatureAndLabel()

    def entropy(self, dataset: np.ndarray):
        votes = self.countVotes(dataset)
        cnt = dataset.shape[0]
        ret = 0
        fun = lambda x: (-x / cnt) * np.log2(x / cnt)
        for result in votes:
            ret += fun(votes[result])
        return ret

    #     it takes dataset and feature column index and filter to yes and no
    def filterOnFeature(self, wholeData: np.ndarray, feature: int):
        uniqueValues = np.unique(wholeData[:, feature])
        return [wholeData[wholeData[:, feature] != value] for value in uniqueValues]

    def countVotes(self, wholeData: np.ndarray):
        unique, counts = np.unique(wholeData[:, -1], return_counts=True)  # get votes from last column
        return dict(zip(unique, counts))
    def test(self,dataset: np.ndarray):
        if self.root is None:
            print("can't test before training")
            return
        else:
            #todo
            return
            # todo