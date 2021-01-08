from SVMGradientDescent import SVMGradientDescent
import matplotlib.pyplot as plt
import numpy as np
from Data_Wrapper import Data

if __name__ == "__main__":
    import svm_conf as conf

    for i in range(10):
        data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
        g = SVMGradientDescent(conf.alpha, data.training, conf.MAX_ITERATIONS, conf.deriv, conf.totalCost)
        g.run()
        errors = g.errors
        plt.plot(errors)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.show()
        print("Train Acurracy = " + str(conf.get_accuracy(data.training, g.w)))
        print("Test Acurracy = " + str(conf.get_accuracy(data.testing, g.w)))
        print("final training error = " + str(g.errors[-1]))
        print("Benchmark Without Kernel = " + str(conf.benchmark(data, False)))
        print("Benchmark With Kernel = " + str(conf.benchmark(data, True)))
        print()


    # if (ds.n == 2):
    #     plt.plot([x[1] for [x, y] in ds], [y for [x, y] in ds], 'og', [0, 1],
    #              [conf.h([1, 0], g.w), conf.h([1, 1], g.w)], 'k')
    #     plt.xlabel("oldpeak")
    #     plt.ylabel("hasCancer")
    #     plt.show()
    # while (False):
    #     ans = str(input("Do you want test one more example? [y/n]\n"))
    #     if (ans == 'n'):
    #         break
    #     x = [1]
    #     for f in conf.featureCols:
    #         ans = float(input("Enter feature " + f + ": "))
    #         x.append(ans)
    #
    #     x = np.array(x)
    #     x = x - ds.normFeatures[:, 0]
    #     x /= ds.normFeatures[:, 1]
    #
    #     y = conf.h(x, g.c)
    #     y *= ds.normLabels[1]
    #     y += ds.normLabels[0]
    #
    #     print("Y = " + str(y))
