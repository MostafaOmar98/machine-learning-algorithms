from Dataset import Dataset
import conf
from GradientDescent import GradientDescent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol)
    g = GradientDescent(0.1, ds, conf.MAX_ITERATIONS, conf.h)
    g.run()
    errors = g.errors
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()