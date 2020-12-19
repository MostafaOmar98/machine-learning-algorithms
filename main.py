from Dataset import Dataset
# import conf
from GradientDescent import GradientDescent
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    ans = input("if you want to test <House price mode> press -H-\nelse if you want to test <Heart disease model> press -D-")
    if(ans == 'h'):
        import  house_conf
    else :
        import heart_conf as conf
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol)
    g = GradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.h,conf.cost)
    g.run()
    errors = g.errors
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    # todo update the plotting axis based on which algorithm
    dsPlot = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, False)
    plt.scatter([x[0] for [x, y] in dsPlot], [y for [x, y] in dsPlot])
    plt.xlabel("Square Foot Living")
    plt.ylabel("Price")
    plt.show()
