import pandas as pd
import numpy as np

class GradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, h,cost):
        self.alpha = alpha # Learning Rate
        self.ds = ds # Dataset
        self.n = ds.n # Number of features WITH bias included
        self.m = ds.m # Number of examples
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.h = h # hypothesis function
        self.cost = cost
        # c is theta
        self.c = np.random.default_rng(1).random(self.n) # create c as 1xn randomly initialized
        self.errors = []

    def run(self):
        self.errors.append(self.cost(self.ds,self.h,self.c,self.m))
        for i in range (self.MAX_ITERATIONS):
            self.updateParameters()
            self.errors.append(self.cost(self.ds,self.h,self.c,self.m))
            print('Iteration ' + str(i + 1) + ' ' + str(self.errors[-1]))

    def updateParameters(self):
        newC = np.zeros(self.n)
        # to update simtuanesly
        for j in range (self.n):
                newC[j] = self.c[j] - self.alpha * self.deriv(j)
        self.c = newC
    # both logistic regression and linear regression have the same derivative with different hypothesis functoin
    def deriv(self, factorIndex=0):
        # todo  shouldn't we increment the factorIndex ??? @Bekh
        ret = 0
        for [x, y] in self.ds:
            ret -= x[factorIndex] * (self.h(self.c, x) - y)
        ret /= self.m
        return ret

