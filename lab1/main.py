from perceptron import Perceptron
from adaline import Adaline
import tools

# DATA = [x,y]
# [pair([values], label)]
# x = [x1, x2], y = y1
data_and = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
data_or = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
data_nor = [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)]

data_and_bi = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
data_or_bi = [([-1, -1], -1), ([-1, 1], 1), ([1, -1], 1), ([1, 1], 1)]
data_xor_bi = [([-1, -1], -1), ([-1, 1], 1), ([1, -1], 1), ([1, 1], -1)]


EPOCHS = 100
DATA_SIZE = 100


def perceptron_test():
    '''
    Perceptron Test
    '''
    print("\nPERCEPTRON:")
    perceptron = Perceptron()
    data = tools.gen_and_data(DATA_SIZE)
    perceptron.init(data)
    perceptron.train()
    perceptron.print_wages_and_bias()
    perceptron.classify(data_and)


def adaline_test():
    '''
    Adaline Test
    '''
    print("\nAdaline:")
    adaline = Adaline()
    # data = tools.gen_and_data_bi(DATA_SIZE)
    adaline.init(data_xor_bi)
    adaline.train()
    adaline.print_wages_and_bias()
    adaline.classify(data_and_bi)


if __name__ == "__main__":
    # perceptron_test()
    adaline_test()
