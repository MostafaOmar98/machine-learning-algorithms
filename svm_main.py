from SVMGradientDescent import SVMGradientDescent
import matplotlib.pyplot as plt
import numpy as np
from Data_Wrapper import Data

if __name__ == "__main__":
    import svm_conf as conf

    data = Data(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, conf.TRAIN_SIZE, conf.preProc, conf.addbias)
    ds = data.training
    g = SVMGradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.deriv)
    g.run()
    dsTest = data.testing

    if (ds.n == 2):
        plt.plot([x[1] for [x, y] in ds], [y for [x, y] in ds], 'og', [0, 1],
                 [conf.h([1, 0], g.w), conf.h([1, 1], g.w)], 'k')
        plt.xlabel("oldpeak")
        plt.ylabel("hasCancer")
        plt.show()

    print("Acurracy = " + str(conf.get_accuracy(data.testing, g.w)))
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
