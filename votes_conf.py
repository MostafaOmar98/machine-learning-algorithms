
TRAIN_PATH = './assets/house-votes-84.data.csv'
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
featureCols = []
for i in range(1, 17):
    featureCols.append("c{0}".format(str(i)))
for i in featureCols:
    print(i)
labelCol = ["target"]
print(featureCols)
preProc = False
addbias = False
alpha = 0.01
MAX_ITERATIONS = 500
