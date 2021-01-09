import numpy as np

TRAIN_PATH = './assets/house-votes-84.data.csv'
TRAIN_SIZE = 0.5
TEST_SIZE = 1 - TRAIN_SIZE
featureCols = []
for i in range(1, 17):
    featureCols.append("c{0}".format(str(i)))
labelCol = ["target"]
preProc = False
addbias = False
alpha = 0.01
MAX_ITERATIONS = 500



# we add the absence by the giving the major vote to missing votes in dataset
def preprocess(features: np.ndarray):
    yes = 0
    no = 0
    for column in features.T:
        unique, count = np.unique(column, return_counts=True)
        ans = dict(zip(unique, count))
        winner = 'y' if ans['y'] > ans['n'] else 'n'
        for i in range(column.shape[0]):
            if column[i] == '?':
                column[i] = winner

