from keras.models import Sequential
from keras.layers import Dense


# Define model
def MLP_model(num_pixels, num_classes):
    # Create model
    model = Sequential()
    model.add(Dense(num_pixels / 2, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_pixels / 4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
