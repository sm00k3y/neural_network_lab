from perceptron import Perceptron
from adaline import Adaline

# DATA = [x,y]
# [pair([values], label)]
# x = [x1, x2], y = y1
data_and = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
data_or = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
data_nor = [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)]

data_and_bi = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]

EPOCHS = 100


def perceptron_test():
    '''
    Perceptron Test
    '''
    print("\nPERCEPTRON:")
    perceptron = Perceptron()
    perceptron.init(data_nor)
    for _ in range(EPOCHS):
        perceptron.train()
    perceptron.print_wages_and_bias()
    perceptron.classify(data_nor)


def adaline_test():
    '''
    Adaline Test
    '''
    print("\nAdaline:")
    adaline = Adaline()
    adaline.init(data_and_bi)
    # for _ in range(EPOCHS):
    adaline.train()
    adaline.print_wages_and_bias()
    adaline.classify(data_and_bi)


if __name__ == "__main__":
    # perceptron_test()
    adaline_test()
