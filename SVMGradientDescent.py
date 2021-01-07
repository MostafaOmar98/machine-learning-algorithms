import numpy as np


class SVMGradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, deriv, cost):
        self.alpha = alpha  # Learning Rate
        self.ds = ds  # Dataset
        self.n = ds.n  # Number of features WITH bias included
        self.m = ds.m  # Number of examples
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.deriv = deriv
        # c is theta
        self.lamda = 1.0/MAX_ITERATIONS
        self.w = np.random.default_rng(1).random(self.n)  # create c as 1xn randomly initialized
        self.cost = cost
        self.errors = []

    def run(self):
        self.errors.append(self.cost(self.ds, self.w))
        for i in range(self.MAX_ITERATIONS):
            self.updateParameters()
            self.errors.append(self.cost(self.ds, self.w))
            # print("Error on iteration " + str(i) + " = " + str(self.errors[-1]))

    def updateParameters(self):
        for [x, y] in self.ds:
            self.w = self.w - self.alpha * self.deriv(x, y, self.w, self.lamda)


