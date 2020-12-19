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
    ds = Dataset(conf.TRAIN_PATH, conf.featureCols, conf.labelCol)
    g = GradientDescent(conf.alpha, ds, conf.MAX_ITERATIONS, conf.h, conf.cost)
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
