from perceptron import Perceptron

# DATA = [x,y]
# [pair([values], label)]
# x = [x1, x2], y = y1
data_and = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
data_or = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
data_nor = [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)]

if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.init(data_nor)
    for i in range(100):
        perceptron.train()
    perceptron.print()


    perceptron.classify(data_nor)
    # res = excitation(values, wages, bias)
    # # print(res)
    # print(unipolar_function(res, threshold))
