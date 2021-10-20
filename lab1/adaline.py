import numpy as np


class Adaline:
    def __init__(self):
        self.wages = np.array([10, 10])
        self.min_accurracy = 200
        self.threshold = 0
        self.bias = 0
        self.learning_rate = 0.01
        self.data = []
        self.epoch_number = 0

    def init(self, data):
        self.data = data

    def bipolar_function(self, value):
        return 1 if value > self.threshold else -1

    def excitation(self, values):
        return np.dot(values, self.wages) + self.bias

    def train(self):
        sum_err = self.min_accurracy + 1

        while sum_err > self.min_accurracy:
            if self.data == []:
                return False

            errors_arr = []

            for pair in self.data:
                values, y = pair
                exct_val = self.excitation(values)
                error = self.calculate_error(y, exct_val)
                self.update_wages_and_bias(values, error)
                errors_arr.append(error**2)

            sum_err = np.sum(errors_arr) / len(self.data)
            self.epoch_number += 1
            print("EPOCH", self.epoch_number, "  Mean Square Error:", sum_err)

    def calculate_error(self, y, exct_val):
        return y - exct_val

    def update_wages_and_bias(self, x, error):
        self.wages = np.array(self.wages) + ((2 * self.learning_rate * error) * np.array(x))
        self.bias = self.bias + (2 * self.learning_rate * error)

    def classify(self, data):
        correct_classes = []
        for pair in data:
            values, y = pair
            exct_val = self.excitation(values)
            class_y = self.bipolar_function(exct_val)
            print("Values:", values, "Class", class_y)
            correct_classes.append(1 if class_y == y else 0)
        print("Correctness:", np.sum(correct_classes) / len(correct_classes) * 100, "%")

    def print_wages_and_bias(self):
        print("\nWages:", self.wages)
        print("Bias:", self.bias)
