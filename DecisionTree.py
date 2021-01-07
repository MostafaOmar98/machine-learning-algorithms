import numpy as np
from DataSet import DataSet


# we add the absence by the giving the major vote to missing votes in dataset
def populate_absence(features: np.ndarray):
    yes = 0
    no = 0
    for column in features.T:
        for vote in column:
            yes += (1 if vote == 'y' else 0)
            no += (1 if vote == 'n' else 0)
        winner = 'y' if yes > no else 'n'
        for i in range(column.shape[0]):
            if column[i] == '?':
                column[i] = winner


#     it takes dataset and feature column index and filter to yes and no
def filterOnFeature(wholeData: np.ndarray, feature: int):
    uniqueValues = np.unique(wholeData[:, feature])
    # filteredNo = wholeData[wholeData[:, feature] != 'y']
    # filteredYes = wholeData[wholeData[:, feature] != 'n']
    # return [filteredYes, filteredNo]
    # return [map( lambda value : wholeData[:,feature]!=value,uniqueValues)]
    return [wholeData[wholeData[:, feature] != value] for value in uniqueValues]


def run(dataset: DataSet):
    populate_absence(dataset.features)
    dataset.mergeFeatureAndLabel()
    for arr in filterOnFeature(dataset.wholeData, 0):
        for row in arr:
            print(row)
            break
