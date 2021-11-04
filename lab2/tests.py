from mlp import MLP
import time
import tools


SAMPLE = 1000
LABELS_COUNT = 10

training_data, validation_data = tools.get_data(SAMPLE)


def neurons_count_in_hidden_layer():
    # Title
    print("TESTING DIFFERENT NEURONS NUMBER IN ONE HIDDEN LAYER")
    # Params
    HIDDEN_LAYERS = 1
    EPOCHS = 21
    BATCH_SIZE = 30
    LEARNING_RATE = 0.1
    # Tests
    for neurons in [10, 15, 100, 300, 784]:
        print("\n\nNeurons count:", neurons, end="\n\n")
        start = time.time()
        mlp = MLP(training_data, HIDDEN_LAYERS, LABELS_COUNT, [neurons])
        mlp.SDG(epochs=EPOCHS, mini_batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, test_data=validation_data)
        stop = time.time()
        print("\nTIME:", stop-start, "s")


def learning_rate_test():
    # Title
    print("TESTING DIFFERENT LEARNING RATES")
    # Params
    HIDDEN_LAYERS = 1
    EPOCHS = 21
    BATCH_SIZE = 30
    # Tests
    for learning_rate in [0.1, 0.05, 0.01, 0.001]:
        print("\n\nLearning rate:", learning_rate, end="\n\n")
        start = time.time()
        mlp = MLP(training_data, HIDDEN_LAYERS, LABELS_COUNT)
        mlp.SDG(epochs=EPOCHS, mini_batch_size=BATCH_SIZE, learning_rate=learning_rate, test_data=validation_data)
        stop = time.time()
        print("\nTIME:", stop-start, "s")


def batch_sizes_test():
    # Title
    print("TESTING DIFFERENT BATCH SIZES")
    # Params
    HIDDEN_LAYERS = 1
    EPOCHS = 21
    LEARNING_RATE = 0.1
    # Tests
    for batch_size in [1, 5, 10, 50, 100, 1000]:
        print("\n\nBatch size:", batch_size, end="\n\n")
        start = time.time()
        mlp = MLP(training_data, HIDDEN_LAYERS, LABELS_COUNT)
        mlp.SDG(epochs=EPOCHS, mini_batch_size=batch_size, learning_rate=LEARNING_RATE, test_data=validation_data)
        stop = time.time()
        print("\nTIME:", stop-start, "s")


def starting_weights_test():
    # Title
    print("TESTING DIFFERENT STARTING WEIGHTS")
    # Params
    HIDDEN_LAYERS = 1
    EPOCHS = 21
    LEARNING_RATE = 0.1
    BATCH_SIZE = 30
    # Tests
    for weight in [1, 5, 10, 100]:
        print("\n\nWeight range: (", weight, ",", -weight, ")", end="\n\n")
        start = time.time()
        mlp = MLP(training_data, HIDDEN_LAYERS, LABELS_COUNT, weights_range=(weight, -weight))
        mlp.SDG(epochs=EPOCHS, mini_batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, test_data=validation_data)
        stop = time.time()
        print("\nTIME:", stop-start, "s")
