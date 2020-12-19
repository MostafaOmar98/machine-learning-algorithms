TRAIN_PATH = 'assets/house_data.csv'  # relative path to the project directory /assets directory has the datasets
# featureCols=['sqft_living', 'grade', 'lat', 'view']
featureCols = ['sqft_living']
labelCol = ['price']
alpha = 0.1
MAX_ITERATIONS = 100
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE


def h(x, c):
    '''
    hypothesis function
    :param x: numpy array representing feature vector
    :param c: numpy array representing weights
    :return: float value representing the prediction
    '''
    return c.dot(x)


def cost(ds, h, c, m):
    '''

    :param ds: dataset
    :param h: hypothesis function
    :param c: thetas
    :param m: number of examples
    :return:
    '''
    ret = 0
    # looping on examples only, no need to loop on features because dot product
    for [x, y] in ds:
        ret += (h(c, x) - y) ** 2
    ret /= (2 * m)
    return ret
