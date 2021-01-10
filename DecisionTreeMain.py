import votes_conf as conf
from Data_Wrapper import Data
from DecisionTree import Decision_Tree
import numpy as np
if __name__ == "__main__":

    # we run 5 times with 0.25 training size random from dataset
    for i in range(5):
        data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
        # we pass the dataset and the preprocess function we want it to be exucted on data
        # in votes case we need to populate absence
        dTree = Decision_Tree.DecisionTree(data.training, conf.preprocess)
        print("iteration {} accuracy = ".format(i + 1) + str(dTree.testDataSet(data.testing)))
        print("iteration {} tree height = ".format(i + 1) + str(dTree.getHeight()))
        print("iteration {} tree size = ".format(i + 1) + str(dTree.getSize()))

    Accuracy = []
    Height = []
    Size = []
    for trainSize in range(30, 80, 10):
        data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, trainSize / 100, conf.preProc, conf.addbias)
        # we pass the dataset and the preprocess function we want it to be exucted on data
        # in votes case we need to populate absence
        dTree = Decision_Tree.DecisionTree(data.training, conf.preprocess)
        accuracy = dTree.testDataSet(data.testing)
        height = dTree.getHeight()
        size = dTree.getSize()
        Accuracy.append(accuracy)
        Height.append(height)
        Size.append(size)
        # minHeight = min(minHeight, height)
        # minSize = min(minSize, size)
        # minAccuracy = min(minAccuracy, accuracy)
        print("training size = {}% accuracy = ".format(trainSize) + str(accuracy))
        print("training size = {}% tree height = ".format(trainSize) + str(height))
        print("training size = {}% tree size = ".format(trainSize) + str(size))
    nAccuracy = np.array(Accuracy)
    nHeight = np.array(Height)
    nSize = np.array(Size)
    print("min accuracy " + str(np.amin(Accuracy)))
    print("max accuracy " + str(np.amax(Accuracy)))
    print("mean accuracy " + str(np.mean(Accuracy)))

    print("min height " + str(np.amin(Height)))
    print("max height " + str(np.amax(Height)))
    print("mean height " + str(np.mean(Height)))

    print("min Size " + str(np.amin(Size)))
    print("max Size " + str(np.amax(Size)))
    print("mean Size " + str(np.mean(Size)))