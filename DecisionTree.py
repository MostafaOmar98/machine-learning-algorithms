import numpy as np
from DataSet import DataSet


# we add the absence by the giving the major vote to missing votes in dataset
def populate_absence(features: np.ndarray):
    yes = 0
    no = 0
    for column in features.T:
        unique,count = np.unique(column, return_counts=True)
        ans = dict(zip(unique, count))
        winner = 'y' if ans['y'] > ans['n'] else 'n'
        for i in range(column.shape[0]):
            if column[i] == '?':
                column[i] = winner


#     it takes dataset and feature column index and filter to yes and no
def filterOnFeature(wholeData: np.ndarray, feature: int):
    uniqueValues = np.unique(wholeData[:, feature])
    return [wholeData[wholeData[:, feature] != value] for value in uniqueValues]


def run(dataset: DataSet):
    populate_absence(dataset.features)
    dataset.mergeFeatureAndLabel()
    gain = entropy(dataset.wholeData)
    # countVotes(dataset.wholeData)

def countVotes(wholeData: np.ndarray):
    unique, counts = np.unique(wholeData[:, -1], return_counts=True) # get votes from last column
    return dict(zip(unique, counts))

def entropy(dataset :np.ndarray ):
    votes = countVotes(dataset)
    cnt = dataset.shape[0]
    ret = 0
    fun = lambda x : (-x/cnt)*np.log2(x/cnt)
    for result in votes:
        ret+=fun(votes[result])
    return ret