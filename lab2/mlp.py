import numpy as np
import math
import random


class MLP:
    # INIT FUNCTIONS
    def __init__(self, data_count, layers, labels):
        self.neurons_by_layers = []
        self.weights_by_layers = []
        self.first_layer_neuron_count = data_count
        self.layers_count = layers
        self.labels_count = labels

        self.init_layers()
        self.init_weights()
        self.biases = np.zeros(layers + 1)

    def init_layers(self):
        self.neurons_by_layers.append(self.first_layer_neuron_count)
        for _ in range(self.layers_count):
            self.neurons_by_layers.append(math.floor(np.average([self.neurons_by_layers[-1], self.labels_count])))
        self.neurons_by_layers.append(self.labels_count)

    def init_weights(self):
        for layer in range(self.layers_count + 1):
            self.weights_by_layers.append(np.random.normal(
                size=(self.neurons_by_layers[layer], self.neurons_by_layers[layer + 1])) / 255)

    # ALGORITHM FUNCTIONS
    # def excitation(self, X, curr_layer):
    #     a = np.dot(X, self.weights_by_layers[curr_layer])
    #     a = a + self.biases[curr_layer]
    #     return a

    # def softmax(self, vector):
    #     return np.exp(vector) / np.exp(vector).sum(0)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def activation_sigm(self, exct_val):
        return 1.0 / (1.0 + np.exp(-exct_val))

    def sigmoid_derivative(self, val):
        return self.activation_sigm(val) * (1 - self.activation_sigm(val))

    def activation_tanh(self, exct_val):
        return (2 / (1 + np.exp(-2 * exct_val))) - 1

    def activation_relu(self, exct_val):
        a = 0 if exct_val < 0 else exct_val
        return a

    def relu_derivative(self, val):
        return val > 0

    def activation_softplus(self, exct_val):
        return np.log(1 + np.exp(exct_val))

    def sofplus_derivative(self, val):
        return 1 / (1 + np.exp(-val))

    # FORWARD_CHAINING
    def forward_chaining(self, input_data):
        a = input_data
        # print(a)
        for i, (b, w) in enumerate(zip(self.biases, self.weights_by_layers)):
            # a = self.activation_sigm(np.dot(a, w) + b)
            # a = self.activation_relu(np.dot(a, w) + b)
            # a = self.activation_softplus(np.dot(a, w) + b)
            a = np.dot(a, w) + b
            # if i != self.layers_count + 1:
            #     a = self.activation_sigm(np.dot(a, w) + b)
            # else:
            #     a = np.dot(a, w) + b
        # print(self.softmax(a))
        return self.softmax(a)

    # Stochastic Gradient Descent
    def SDG(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete...')

    def update_batch(self, batch, learning_rate):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights_by_layers]
        for x, y in batch:
            mini_delta_biases, mini_delta_weights = self.backprop(x, y)
            delta_biases = [db + mdb for db, mdb in zip(delta_biases, mini_delta_biases)]
            delta_weights = [dw + mdw for dw, mdw in zip(delta_weights, mini_delta_weights)]
            # print("DONE")
        self.weights_by_layers = [weights-(learning_rate / len(batch)) * dw
                                  for weights, dw in zip(self.weights_by_layers, delta_weights)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, delta_biases)]

    def backprop(self, x, y):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights_by_layers]
        activation = x
        activations_arr = [x]
        inputs_by_layers = [x]
        errors = []

        # FeedForward
        for i, (b, w) in enumerate(zip(self.biases, self.weights_by_layers)):
            input_layer = np.dot(activation, w) + b
            inputs_by_layers.append(input_layer)
            # if i == self.layers_count:  # Tutaj chyba bez +1
            #     activation = self.softmax(input_layer)
            # else:
            #     activation = self.activation_sigm(input_layer)
            activation = self.activation_sigm(input_layer)
            activations_arr.append(activation)

        # print(len(inputs_by_layers))
        # print(len(activations_arr))

        # BACKPROP
        # delta = np.multiply(self.cost_derivative(activations_arr[-1], y), self.sigmoid_derivative(inputs_by_layers[-1]))
        # errors = np.dot(np.expand_dims(cost_der, 1), np.expand_dims(activations_arr[-2], 1).T)
        # cost_der = np.expand_dims(cost_der, 0)
        cost_der = self.cost_derivative(activations_arr[-1], y)
        errors.append(cost_der)
        delta_w = np.dot(activations_arr[-2].reshape(1, activations_arr[-2].shape[0]).T, cost_der.reshape((1, cost_der.shape[0])))
        delta_weights[-1] = delta_w
        delta_biases[-1] = cost_der

        for layer in range(2, self.layers_count + 2):
            # errors_bef_act = np.dot(self.weights_by_layers[-layer + 1], delta_weights[-layer + 1])
            # for e in errors:
            #     print(e.shape)
            # print(self.weights_by_layers[-layer+1].shape)
            # print(errors[-layer + 1].shape)
            errors_bef_act = np.dot(self.weights_by_layers[-layer + 1], errors[-layer + 1].T)
            error = np.multiply(errors_bef_act.T, self.sigmoid_derivative(inputs_by_layers[-layer]))
            errors.insert(0, error)
            # error = np.dot(activations_arr[-layer-1].T, error)
            delta_w = np.dot(activations_arr[-layer-1].reshape((1, activations_arr[-layer-1].shape[0])).T, error.reshape(1, error.shape[0]))
            delta_weights[-layer] = delta_w
            delta_biases[-layer] = error
            # print(layer)

        return delta_biases, delta_weights

    def cost_derivative(self, output_activations, y):
        good_y = np.zeros(10)
        good_y[y] = 1
        return -(y - output_activations)

    def evaluate(self, test_data):
        # result = [self.forward_chaining(x) for x, y in test_data]
        # print(result[45])
        test_results = [(np.argmax(self.forward_chaining(x)), y) for (x, y) in test_data]
        # print(test_results)
        return sum(int(x == y) for (x, y) in test_results)

    # PRINTING
    def print(self):
        print("Number of neurons in each layer:", self.neurons_by_layers)
        print("Weights matrixes in each layer:")
        for i, matrix in enumerate(self.weights_by_layers):
            print("Matrix", i+1, "shape:", np.shape(matrix))
        print("Biases:", self.biases)

    # def forward_chaining(self, training_data, training_labels):
        # predicted_labels = []
        # for data, label in zip(training_data[:2], training_labels[:2]):
        #     layer_input = data.flatten()
        #     for curr_layer in range(self.layers_count + 1):
        #         exct_vector = self.excitation(layer_input, curr_layer)
        #         next_layer_input = list(map(self.activation_sigm, exct_vector))
        #         layer_input = next_layer_input

        #     predicted_labels = self.softmax(layer_input)

        # return predicted_labels
