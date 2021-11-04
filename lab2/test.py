import numpy as np
from math import floor


def fun1():
    arrayka = np.random.normal(size=(5))
    print(arrayka)
    print(arrayka.shape)
    print(arrayka.reshape(5, 1).shape)
    print(arrayka.shape[0])

    # print([0] * 5)
    # print(np.zeros((4)))
    # print(floor(np.average([1, 9])))


def fun2():
    tab1 = [1, 2]
    tab2 = [3, 4]
    for i, (x, y) in enumerate(zip(tab1, tab2)):
        print(i, x, y)


def fun3():
    arr = np.array([1,2,3])
    print(np.expand_dims(arr, 0).T)


def fun4():
    arr = np.empty([5])
    print(arr)


def fun5():
    arr = np.array([1, 2, 3, 4])
    print(arr[-4])


if __name__ == "__main__":
    fun5()
