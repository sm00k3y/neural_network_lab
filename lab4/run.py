import time

from consts import BATCH_SIZE, EPOCHS

from keras.preprocessing.image import ImageDataGenerator


def run(data, NNModel):
    X_train, X_test, y_train, y_test, num_pixels, num_classes = data

    # Build the model
    model = NNModel(num_pixels, num_classes)

    # Time the process
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

    stop_time = time.time()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("\nBaseline Error: %.2f%%" % (100-scores[1]*100))
    print("\nOVERALL TIME: {:.3f}s".format(stop_time - start_time))


# =================================
# PIPELINE WITH IMAGE PREPROCESSING
def run_pipeline(data, NNModel):
    # load data
    X_train, X_test, y_train, y_test, num_pixels, num_classes = data

    # Data augmentation
    datagen = ImageDataGenerator(featurewise_center=False,              # set input mean to 0 over the dataset
                                 samplewise_center=False,               # set each sample mean to 0
                                 featurewise_std_normalization=False,   # divide inputs by std of the dataset
                                 samplewise_std_normalization=False,    # divide each input by its std
                                 zca_whitening=False,                   # apply ZCA whitening
                                 rotation_range=10,                     # randomly rotate images in the range (degrees, 0 to 180)
                                 zoom_range=0.1,                        # Randomly zoom image
                                 width_shift_range=0.1,                 # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1,                # randomly shift images vertically (fraction of total height)
                                 shear_range=(0.1),                     # Shear range
                                 horizontal_flip=False,                 # randomly flip images
                                 vertical_flip=False)                   # randomly flip images

    train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    test_gen = datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

    # Build the model
    model = NNModel(num_pixels, num_classes)

    # Time the process
    start_time = time.time()

    # Fit the model
    model.fit_generator(train_gen,
                        epochs=EPOCHS,
                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                        validation_data=test_gen,
                        validation_steps=X_test.shape[0] // BATCH_SIZE)

    stop_time = time.time()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("\nBaseline Error: %.2f%%" % (100-scores[1]*100))
    print("\nOVERALL TIME: {:.3f}s".format(stop_time - start_time))
