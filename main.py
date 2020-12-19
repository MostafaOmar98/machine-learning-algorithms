from Dataset import Dataset
from GradientDescent import GradientDescent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ans = str(input(
        "if you want to test <House price mode> enter H\nelse if you want to test <Heart disease model> enter D\n"))
    if (ans == 'H'):
        import house_conf as conf
    else:
        import heart_conf as conf
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, True, True, conf.TRAIN_SIZE)
    g = GradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.h, conf.cost, conf.deriv)
    g.run()
    errors = g.errors
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

    dsTest = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol, True, True, conf.TEST_SIZE)
    print("Error on test data: " + str(conf.cost(dsTest, conf.h, g.c, dsTest.m)))
    if (ds.n == 2):
        plt.plot([x[1] for [x, y] in ds], [y for [x, y] in ds], 'og', [0, 1], [conf.h([1, 0], g.c), conf.h([1, 1], g.c)], 'k')
        plt.xlabel("Square Foot Living")
        plt.ylabel("Price")
        plt.show()
