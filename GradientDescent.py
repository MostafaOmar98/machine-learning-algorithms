import numpy as np


class GradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, h, cost, deriv):
        self.alpha = alpha  # Learning Rate
        self.ds = ds  # Dataset
        self.n = ds.n  # Number of features WITH bias included
        self.m = ds.m  # Number of examples
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.h = h  # hypothesis function
        self.cost = cost
        self.deriv = deriv
        # c is theta
        self.c = np.random.default_rng(1).random(self.n)  # create c as 1xn randomly initialized
        self.errors = []

    def run(self):
        self.errors.append(self.cost(self.ds, self.h, self.c, self.m))
        for i in range(self.MAX_ITERATIONS):
            self.updateParameters()
            self.errors.append(self.cost(self.ds, self.h, self.c, self.m))
            print('Iteration ' + str(i + 1) + ' ' + str(self.errors[-1]))

    def updateParameters(self):
        newC = np.zeros(self.n)
        # to update simtuanesly
        for j in range(self.n):
            newC[j] = self.c[j] - self.alpha * self.deriv(self.ds, self.h, self.c, j)
        self.c = newC


