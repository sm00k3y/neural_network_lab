import numpy as np
import math
import random
import copy


class MLP:
    # INIT FUNCTIONS
    def __init__(self, training_data, hidden_layers, labels_count, neurons_by_layers=[], weights_range=(0, 0)):
        self.training_data = training_data
        self.layers_count = hidden_layers
        self.first_layer_neuron_count = len(training_data[0][0].flatten())
        self.labels_count = labels_count

        self.neurons_by_layers = self.init_layers(neurons_by_layers)
        self.weights_by_layers = self.init_weights(weights_range)
        self.biases = np.zeros(hidden_layers + 1)

    def init_layers(self, neurons_by_layers):
        ret = []
        if (neurons_by_layers == [] or len(neurons_by_layers) != self.layers_count):
            ret.append(self.first_layer_neuron_count)
            for _ in range(self.layers_count):
                ret.append(math.floor(np.average([ret[-1], self.labels_count])))
            ret.append(self.labels_count)
        else:
            ret.append(self.first_layer_neuron_count)
            for layer in neurons_by_layers:
                ret.append(layer)
            ret.append(self.labels_count)
        return ret

    def init_weights(self, weights_range):
        ret = []
        if weights_range == (0, 0):
            for layer in range(self.layers_count + 1):
                ret.append(np.random.normal(
                           size=(self.neurons_by_layers[layer], self.neurons_by_layers[layer + 1])))
        else:
            for layer in range(self.layers_count + 1):
                ret.append(np.random.uniform(low=weights_range[0], high=weights_range[1],
                           size=(self.neurons_by_layers[layer], self.neurons_by_layers[layer + 1])))
        return ret

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # def softmax(self, x):
    #     """Compute softmax values for each sets of scores in x."""
    #     return np.exp(x) / np.sum(np.exp(x), axis=0)

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
            a = self.activation_sigm(np.dot(a, w) + b)
            # if i != self.layers_count:
            #     a = self.activation_sigm(np.dot(a, w) + b)
            # else:
            #     a = np.dot(a, w) + b
        # print(self.softmax(a))
        return self.softmax(a)

    # Stochastic Gradient Descent
    def SDG(self, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(self.training_data)
        for j in range(epochs):
            random.shuffle(self.training_data)
            mini_batches = [self.training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
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
        # self.default(learning_rate, len(batch), delta_weights, delta_biases)
        self.momentum(learning_rate, len(batch), delta_weights, delta_biases, 0.7)
        # self.weights_by_layers = [weights-(learning_rate / len(batch)) * dw
        #                           for weights, dw in zip(self.weights_by_layers, delta_weights)]
        # self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, delta_biases)]

    def backprop(self, x, y):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights_by_layers]
        activation = x
        activations_arr = [x]
        inputs_by_layers = []
        errors = []

        # FeedForward
        for i, (b, w) in enumerate(zip(self.biases, self.weights_by_layers)):
            input_layer = np.dot(activation, w) + b
            inputs_by_layers.append(input_layer)
            # if i == self.layers_count:
            #     activation = self.softmax(input_layer)
            # else:
            #     activation = self.activation_sigm(input_layer)
            activation = self.activation_sigm(input_layer)
            activations_arr.append(activation)

        # BACKPROP
        cost_der = self.cost_derivative(activations_arr[-1], y)
        errors.append(cost_der)
        delta_w = np.dot(activations_arr[-2].reshape(1, activations_arr[-2].shape[0]).T,
                         cost_der.reshape((1, cost_der.shape[0])))
        delta_weights[-1] = delta_w
        delta_biases[-1] = cost_der

        for layer in range(2, self.layers_count + 2):
            errors_bef_act = np.dot(self.weights_by_layers[-layer + 1], errors[-layer + 1].T)
            errors.insert(0, errors_bef_act.T)
            # error = np.multiply(errors_bef_act.T, self.sigmoid_derivative(inputs_by_layers[-layer]))
            error = errors_bef_act.T * self.sigmoid_derivative(inputs_by_layers[-layer])
            delta_w = np.dot(activations_arr[-layer-1].reshape((1, activations_arr[-layer-1].shape[0])).T,
                             error.reshape(1, error.shape[0]))
            delta_weights[-layer] = delta_w
            delta_biases[-layer] = error

        return delta_biases, delta_weights

    def cost_derivative(self, output_activations, y):
        good_y = np.zeros(10)
        good_y[y] = 1
        return output_activations - good_y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_chaining(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # PRINTING
    def print(self):
        print("Number of neurons in each layer:", self.neurons_by_layers)
        print("Weights matrixes in each layer:")
        for i, matrix in enumerate(self.weights_by_layers):
            print("Matrix", i+1, "shape:", np.shape(matrix))
        print("Biases:", self.biases)

    def serialize_model(self):
        for i, weights in enumerate(self.weights_by_layers):
            np.savetxt(f"serialization/weights_{i}.csv", np.asarray(weights), delimiter=",")
        # np.savetxt("serialization/biases.csv", self.biases, delimiter=',')

    # OPTIMIZERS
    def default(self, learning_rate, batch_len, delta_weights, delta_biases):
        self.weights_by_layers = [weights-(learning_rate / batch_len) * dw
                                  for weights, dw in zip(self.weights_by_layers, delta_weights)]
        self.biases = [b - (learning_rate / batch_len) * nb
                       for b, nb in zip(self.biases, delta_biases)]

    def momentum(self, learning_rate, batch_len, delta_weights, delta_biases, momentum_rate):
        self.weights_by_layers = [weights-(learning_rate / batch_len) * dw
                                  for weights, dw in zip(self.weights_by_layers, delta_weights)]
        self.biases = [b - (learning_rate / batch_len) * nb
                       for b, nb in zip(self.biases, delta_biases)]
        self.weights_by_layers += momentum_rate * self.previous_delta_weights
        self.biases += momentum_rate * self.previous_delta_biases
        self.previous_delta_weights = copy.deepcopy(delta_weights)
        self.previous_delta_biases = copy.deepcopy(delta_biases)
