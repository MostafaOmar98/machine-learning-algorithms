from Dataset import Dataset
import conf
import numpy as np
from GradientDescent import GradientDescent

if __name__ == "__main__":
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol)
    g = GradientDescent(0.1, ds, 10, conf.h)
    pass