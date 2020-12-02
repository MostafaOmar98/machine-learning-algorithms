import pandas as pd
import numpy as np

class GradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, h):
        self.alpha = alpha # Learning Rate
        self.ds = ds # Dataset
        self.n = ds.n # Number of features WITH bias included
        self.m = ds.m # Number of examples
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.h = h # hypothesis function

        self.c = np.random.default_rng(1).random(self.n) # create c as 1xn randomly initialized
        self.errors = []

    def run(self):
        print(self.cost())
        for i in range (self.MAX_ITERATIONS):
            self.updateParameters()
            self.errors.append(self.cost())
            print('Iteration ' + str(i) + ' ' + str(self.errors[-1]))

    def updateParameters(self):
        newC = np.zeros(self.n)
        for j in range (self.n):
                newC[j] = self.c[j] - self.alpha * self.cost(j)
        self.c = newC

    def cost(self, factorIndex=0):
        ret = 0
        for [x, y] in self.ds:
            ret += x[factorIndex] * ((y - self.h(self.c, x))**2)
        ret /= (2 * self.m)
        return ret