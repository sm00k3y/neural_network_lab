import numpy as np


class Adaline:
    def __init__(self):
        self.wages = [0.1, 0.05]
        self.min_accurracy = 0.1
        self.threshold = 0
        self.bias = 0
        self.learning_rate = 0.05
        self.data = []

    def init(self, data):
        self.data = data

    def unipolar_function(self, value):
        return 1 if value > self.threshold else 0

    def bipolar_function(self, value):
        return 1 if value > self.threshold else -1

    def excitation(self, values):
        return np.dot(values, self.wages) + self.bias

    def train(self):
        sum_err = self.min_accurracy + 1
        while sum_err > self.min_accurracy:
            if self.data == []:
                return False

            errors = []

            for pair in self.data:
                values, y = pair
                exct_val = self.excitation(values)
                # test_y = self.bipolar_function(exct_val)
                square_err = self.calculate_square_error(y, exct_val)
                self.update_wages_and_bias(values, square_err)
                errors.append(square_err)

            sum_err = np.sum(errors)

    def calculate_square_error(self, y, exct_val):
        print("Wages", self.wages)
        print(exct_val)
        square_err = (y - exct_val)**2
        # square_err = pow(y - exct_val, 2)
        return square_err
        # return y - test_y

    def update_wages_and_bias(self, x, error):
        for i in range(0, len(x)):
            self.wages[i] = self.wages[i] + (2 * self.learning_rate * error * x[i])
        self.bias = self.bias + (2 * self.learning_rate * error)

    def classify(self, data):
        correct_classes = 0
        incorrect_classes = 0
        for pair in data:
            values, y = pair
            exct_val = self.excitation(values)
            class_y = self.unipolar_function(exct_val)
            print("Values:", values, "Class", class_y)
            if class_y == y:
                correct_classes += 1
            else:
                incorrect_classes += 1
        print("Correctness:", correct_classes / (correct_classes + incorrect_classes) * 100, "%")

    def print_wages_and_bias(self):
        print("Wages:", self.wages)
        print("Bias:", self.bias)
