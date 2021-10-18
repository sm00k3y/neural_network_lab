import numpy as np


class Perceptron:
    def __init__(self):
        self.wages = np.array([0.1, 1])
        self.bias = 0
        self.threshold = 0
        self.learning_rate = 0.0001
        self.use_unipolar = True
        self.data = []
        self.epoch_number = 0

    def init(self, data):
        self.data = data

    def unipolar_function(self, value):
        return 1 if value > self.threshold else 0

    def bipolar_function(self, value):
        return 1 if value > self.threshold else -1

    def excitation(self, values):
        return np.dot(values, self.wages) + self.bias

    def train(self):
        if self.data == []:
            return False

        error_flag = True
        while error_flag:
            error_flag = False
            for pair in self.data:
                values, y = pair
                exct_val = self.excitation(values)
                test_y = self.bipolar_function(exct_val)
                err = self.calculate_error(y, test_y)
                self.update_wages_and_bias(values, err)
                if err != 0:
                    error_flag = True
            self.epoch_number += 1
            print("EPOCH", self.epoch_number, "  Error: True")

    def calculate_error(self, y, test_y):
        return y - test_y

    def update_wages_and_bias(self, x, error):
        self.wages = np.array(self.wages) + ((2 * self.learning_rate * error) * np.array(x))
        self.bias = self.bias + self.learning_rate * error

    def classify(self, data):
        correct_classes = []
        for pair in data:
            values, y = pair
            exct_val = self.excitation(values)
            class_y = self.unipolar_function(exct_val)
            print("Values:", values, "Class", class_y)
            correct_classes.append(1 if class_y == y else 0)
        print("Correctness:", np.sum(correct_classes) / len(correct_classes) * 100, "%")

    def print_wages_and_bias(self):
        print("Wages:", self.wages)
        print("Bias:", self.bias)
