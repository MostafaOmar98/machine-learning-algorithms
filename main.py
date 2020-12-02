from Dataset import Dataset
import conf
from GradientDescent import GradientDescent
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol)
    g = GradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.h)
    g.run()
    errors = g.errors
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

    dsPlot = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, False)
    plt.scatter([x[0] for [x, y] in dsPlot], [y for [x, y] in dsPlot])
    plt.xlabel("Square Foot Living")
    plt.ylabel("Price")
    plt.show()
