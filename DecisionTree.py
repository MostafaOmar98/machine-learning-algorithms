import numpy as np


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
