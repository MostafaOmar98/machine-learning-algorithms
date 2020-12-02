import pandas as pd
import numpy as np

class GradientDescent:
    def __init__(self, alpha, ds):
        self.alpha = alpha
        self.ds = ds
        self.n = len(ds.columns)
        # self.c = init

    def run(self):
        pass