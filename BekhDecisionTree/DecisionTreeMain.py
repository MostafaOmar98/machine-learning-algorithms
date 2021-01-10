import votes_conf as conf
from Data_Wrapper import Data
from DecisionTree import Decision_Tree

if __name__ == "__main__":

    # we run 5 times with 0.25 training size random from dataset
    for i in range(5):
        data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
        # we pass the dataset and the preprocess function we want it to be exucted on data
        # in votes case we need to populate absence
        dTree = Decision_Tree.DecisionTree(data.training, conf.preprocess)
        print("iteration {} ".format(i + 1) + str(dTree.testDataSet(data.training)))
