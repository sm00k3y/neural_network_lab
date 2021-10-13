import numpy as np

class Perceptron:
    def __init__(self):
        self.wages = [0.1, 0.05]
        self.bias = 0
        self.threshold = 0
        self.learning_rate = 0.01
        self.use_unipolar = True
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
        if self.data == []: return False

        for pair in self.data:
            y = pair[1]
            values = pair[0]
            exct_val = self.excitation(values)
            test_y = self.unipolar_function(exct_val)
            err = self.calculate_error(y, test_y)
            self.update_wages_and_bias(values, err)
            
    def calculate_error(self, y, test_y):
        return y - test_y 

    def update_wages_and_bias(self, x, error):
        for i in range(0, len(x)):
            self.wages[i] = self.wages[i] + self.learning_rate * ( error * x[i] )
        self.bias = self.bias + self.learning_rate * error

    def classify(self, data):
        correct_classes = 0
        incorrect_classes = 0
        for pair in data:
            values, y = pair
            exct_val = self.excitation(values)
            class_y = self.unipolar_function(exct_val)
            print("Values:", values, "Class", class_y)
            if (class_y == y):
                correct_classes += 1
            else:
                incorrect_classes += 1
        print("Correctness:", correct_classes / (correct_classes + incorrect_classes))

    def print(self):
        print("Wages:", self.wages)
        print("Bias:", self.bias)

    

