from mlp import MLP
import tools


# For tests
FIRST_LAYER_NEURONS = 20
LABELS_COUNT = 3
LAYERS_COUNT = 1

SAMPLE = 200


def test_mlp():
    mlp = MLP(FIRST_LAYER_NEURONS, LAYERS_COUNT, LABELS_COUNT)
    mlp.print()


if __name__ == "__main__":
    training_data = tools.load_train_data()[:SAMPLE]
    training_labels = tools.load_train_labels()[:SAMPLE]
    test_data = tools.load_test_data()[:SAMPLE]
    test_labels = tools.load_test_labels()[:SAMPLE]
    data_size = len(training_data[0].flatten())
    labels = 10

    mlp = MLP(data_size, LAYERS_COUNT, labels)
    mlp.print()

    good_data = tools.prep_data(tools.normalize_data(training_data), training_labels)
    good_test_data = tools.prep_data(tools.normalize_data(test_data), test_labels)

    good_x = [x for x, y in good_test_data]
    print(len(good_x[0]))

    # preds = mlp.forward_chaining(good_x[13])
    # print("\nPREDICTION ARRAY:", preds)
    # print("PREDICTION ARRAY SUM", sum(preds))
    # print("ARGMAX", np.argmax(preds))

    #   SDG(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
    mlp.SDG(good_data, 100, 10, 0.1, good_test_data)
