from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import RandomRotation
from keras.layers import RandomTranslation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def CNN_full_model(num_pixels, num_classes):
    model = Sequential()

    # Preprocessing
    # model.add(RandomRotation(0.99))
    # model.add(RandomTranslation((-0.99, 0.99), (-0.99, 0.99)))

    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
