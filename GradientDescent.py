import pandas as pd
import numpy as np

class GradientDescent:
    def __init__(self, alpha, ds, MAX_ITERATIONS, h):
        self.alpha = alpha # Learning Rate
        self.ds = ds # Dataset
        self.n = ds.n # Number of features WITH bias included
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.h = h # hypothesis function

        self.c = np.random.default_rng(1).random(self.n) # create c as 1xn randomly initialized
        self.errors = []

    def run(self):
        for i in range (self.MAX_ITERATIONS):
            self.updateParameters()
            self.errors.append(self.getError())

    def updateParameters(self):
        newC = np.zeros(self.n)

        for j in range (self.n):
            for [x, y] in self.ds:
                pass


    def getError(self):
        pass