import sys
from typing import Callable

import numpy as np

from DataSet import DataSet
from .Node import Node


class DecisionTree:

    def __init__(self, dataset: DataSet, preprocess: Callable):
        self.dataset = dataset
        self.preProcess = preprocess
        self.runPre()
        self.root = Node("root")
        self.train(self.root, self.dataset.wholeData, [])

    def train(self, currentNode: Node, dataset: np.ndarray, taken):
        # base case
        ent = self.entropy(dataset)
        if ent == 0:
            currentNode.result = dataset[:, -1][0]  # we make the result the
            return
        mx = 0
        mxindx = -1
        for featureIndx in range(0, dataset.shape[1] - 1):
            if not featureIndx in taken:
                if mxindx == -1:
                    mxindx = featureIndx
                gain = ent
                for splitted in self.filterOnFeature(dataset, featureIndx):
                    gain = gain - ((1.0 * splitted.shape[0] / dataset.shape[0]) * self.entropy(splitted))
                if mx < gain:
                    mx = gain
                    mxindx = featureIndx
        if mxindx == -1:
            votes = self.countVotes(dataset)
            mx = 0
            result = None
            for vote in votes:
                if votes[vote] > mx:
                    mx = votes[vote]
                    result = vote
            currentNode.result = result
            return

        currentNode.featureIndex = mxindx
        taken.append(mxindx)
        for (ds) in self.filterOnFeature(wholeData=dataset, feature=mxindx):
            if ds.shape[0] > 0:  # todo check removal
                currentNode.addToChildren(Node(featureName=ds[:, mxindx][0]))
                self.train(currentNode.children[-1], ds, taken=taken.copy())

    def runPre(self):
        self.preProcess(self.dataset.features)
        self.dataset.mergeFeatureAndLabel()

    def entropy(self, dataset: np.ndarray):
        votes = self.countVotes(dataset)
        if len(votes) == 1:  # if we have pure set
            return 0
        cnt = dataset.shape[0]
        ret = 0
        fun = lambda x: (-x / cnt) * np.log2(x / cnt)
        for result in votes:
            ret += fun(votes[result])
        return ret

    #     it takes dataset and feature column index and filter to yes and no
    def filterOnFeature(self, wholeData: np.ndarray, feature: int):
        uniqueValues = np.unique(wholeData[:, feature])
        return [wholeData[wholeData[:, feature] == value] for value in uniqueValues]

    def countVotes(self, wholeData: np.ndarray):
        unique, counts = np.unique(wholeData[:, -1], return_counts=True)  # get votes from last column
        return dict(zip(unique, counts))

    # it take Dataset and test it
    def testDataSet(self, testDataSet: DataSet):
        if self.root is None:
            print("model is not trained yet, please traing first")
            return
        cntRight = 0
        self.preProcess(testDataSet.features)
        total = len(testDataSet)
        for [x, y] in testDataSet:
            result = self.test(x, self.root)
            if result == y:
                cntRight += 1

        return (1.0 * cntRight) / total

    # it takes one examples and returns it's result
    def test(self, example: np.ndarray, node: Node):
        if node.result is None:
            featureIndx = node.featureIndex
            for child in node.children:
                if child.featureName == example[featureIndx]:
                    return self.test(example, child)
        return node.result

    def getHeight(self):
        return self.root.getHeight(self.root, 1)

    def getSize(self):
        return self.root.getSize(self.root)
