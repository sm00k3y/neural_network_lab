from mlp import MLP
import tools
import tests


# For tests
LABELS = 10
LAYERS_COUNT = 1
SAMPLE = 1000


def testing():
    tests.neurons_count_in_hidden_layer()
    tests.learning_rate_test()
    tests.batch_sizes_test()
    tests.starting_weights_test()


if __name__ == "__main__":
    # testing()

    training_data, test_data = tools.get_data(SAMPLE)

    mlp = MLP(training_data=training_data, hidden_layers=LAYERS_COUNT, labels_count=LABELS)
    mlp.print()

    # SDG(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
    mlp.SDG(epochs=10, mini_batch_size=30, learning_rate=0.1, test_data=test_data)
    # mlp.serialize_model()
