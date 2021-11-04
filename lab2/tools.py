import idx2numpy
# import numpy as np

FILE_NAMES = {
    "train_data": "data/train_images",
    "train_labels": "data/train_labels",

    "test_data": "data/test_images",
    "test_labels": "data/test_labels",
}


def load_train_data():
    return idx2numpy.convert_from_file(FILE_NAMES["train_data"])


def load_train_labels():
    return idx2numpy.convert_from_file(FILE_NAMES["train_labels"])


def load_test_data():
    return idx2numpy.convert_from_file(FILE_NAMES["test_data"])


def load_test_labels():
    return idx2numpy.convert_from_file(FILE_NAMES["test_labels"])


def normalize_data(data):
    return data / 255


def get_data(sample=0):
    if (sample != 0):
        training_data = load_train_data()[:sample]
        training_labels = load_train_labels()[:sample]
        test_data = load_test_data()[:sample]
        test_labels = load_test_labels()[:sample]
    else:
        training_data = load_train_data()
        training_labels = load_train_labels()
        test_data = load_test_data()
        test_labels = load_test_labels()

    prep_training_data = prep_data(normalize_data(training_data), training_labels)
    prep_test_data = prep_data(normalize_data(test_data), test_labels)

    return prep_training_data, prep_test_data


def prep_data(x, y):
    good_data = []
    for data_x, data_y in zip(x, y):
        # arrayka = np.zeros(10)
        # arrayka[data_y] = 1
        good_data.append((data_x.flatten(), data_y))
    # print(good_data[0])
    return good_data
