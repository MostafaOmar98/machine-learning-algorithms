import numpy as np


class SVMGradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, deriv):
        self.alpha = alpha  # Learning Rate
        self.ds = ds  # Dataset
        self.n = ds.n  # Number of features WITH bias included
        self.m = ds.m  # Number of examples
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.deriv = deriv
        # c is theta
        self.lamda = 1.0/MAX_ITERATIONS
        self.w = np.random.default_rng(1).random(self.n)  # create c as 1xn randomly initialized

    def run(self):
        for i in range(self.MAX_ITERATIONS):
            self.updateParameters()

    def updateParameters(self):
        for [x, y] in self.ds:
            self.w = self.w - self.alpha * self.deriv(x, y, self.w, self.lamda)


