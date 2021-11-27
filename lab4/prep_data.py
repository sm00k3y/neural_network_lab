from keras.datasets import mnist
from keras.utils import np_utils

from consts import SAMPLE


def prep_data_mlp():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')[:SAMPLE]
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')[:SAMPLE]

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)[:SAMPLE]
    y_test = np_utils.to_categorical(y_test)[:SAMPLE]
    num_classes = y_test.shape[1]

    return X_train, X_test, y_train, y_test, num_pixels, num_classes


def prep_data_cnn():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')[:SAMPLE]
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')[:SAMPLE]

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)[:SAMPLE]
    y_test = np_utils.to_categorical(y_test)[:SAMPLE]
    num_classes = y_test.shape[1]

    return X_train, X_test, y_train, y_test, 0, num_classes
