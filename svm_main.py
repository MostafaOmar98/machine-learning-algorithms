from SVMGradientDescent import SVMGradientDescent
import matplotlib.pyplot as plt
import numpy as np
from Data_Wrapper import Data

if __name__ == "__main__":
    import svm_conf as conf

    bestAccuracy = -1e30
    ds = None
    w = None
    for i in range(10):
        data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
        g = SVMGradientDescent(conf.alpha, data.training, conf.MAX_ITERATIONS, conf.deriv, conf.totalCost)
        g.run()
        errors = g.errors
        plt.plot(errors)
        plt.xlabel('Iterations With Learning Rate = ' + str(conf.alpha))
        plt.ylabel('Error')
        plt.show()
        print("Train Acurracy = " + str(conf.get_accuracy(data.training, g.w)))
        test_accuracy = conf.get_accuracy(data.testing, g.w)
        print("Test Acurracy = " + str(test_accuracy))
        if (test_accuracy > bestAccuracy):
            bestAccuracy = test_accuracy
            ds = data.training
            w = g.w
        print("final training error = " + str(g.errors[-1]))
        print()

    print("Best Test Accuracy = " + str(bestAccuracy))

    while (True):
        ans = str(input("Do you want test one more example? [y/n]\n"))
        if (ans == 'n'):
            break
        x = [1]
        for f in conf.featureCols:
            ans = float(input("Enter feature " + f + ": "))
            x.append(ans)

        x = np.array(x)
        x = x - ds.normFeatures[:, 0]
        x /= ds.normFeatures[:, 1]

        print("Y = " + str(conf.predict(x, w)))
