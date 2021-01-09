from GradientDescent import GradientDescent
import matplotlib.pyplot as plt
import numpy as np
from Data_Wrapper import Data
from DataSet import DataSet

if __name__ == "__main__":
    ans = str(input(
        "if you want to test <House price mode> enter H\nelse if you want to test <Heart disease model> enter D\n"))
    if (ans == 'H'):
        import house_conf as conf
    elif ans == 'v':
        import votes_conf as conf
    else:
        import heart_conf as conf

    data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
    # ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, True, True, conf.TRAIN_SIZE)
    ds = data.training
    g = GradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.h, conf.cost, conf.deriv, conf.addbias)
    g.run()
    errors = g.errors
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    dsTest = data.testing
    # dsTest = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, True, False, conf.TEST_SIZE)
    print("Error on test data: " + str(conf.cost(dsTest, conf.h, g.c, dsTest.m)))
    if (ds.n == 2):
        plt.plot([x[1] for [x, y] in ds], [y for [x, y] in ds], 'og', [0, 1],
                 [conf.h([1, 0], g.c), conf.h([1, 1], g.c)], 'k')
        plt.xlabel("Square Foot Living")
        plt.ylabel("Price")
        plt.show()

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

        y = conf.h(x, g.c)
        y *= ds.normLabels[1]
        y += ds.normLabels[0]

        print("Y = " + str(y))
