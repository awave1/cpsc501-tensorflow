import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
import os


class NotMNISTModel:
    def __init__(self):
        self.model = Sequential()

    def train(self):
        print("--Get data--")
        with np.load(f'{os.getcwd()}/notMNIST.npz', allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        print("--Process data--")
        print(len(y_train))
        x_train, x_test = x_train / 255.0, x_test / 255.0

        num_classes = 10

        self.model.add(Conv1D(32, 5, activation='relu',
                              kernel_regularizer=l2(0.001), input_shape=(28, 28)))
        self.model.add(MaxPooling1D(3))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("--Fit model--")
        self.model.fit(x_train, y_train, epochs=10, verbose=2)

        print("--Evaluate model--")
        model_loss, model_acc = self.model.evaluate(x_test,  y_test, verbose=2)
        print(f"Model Loss:    {model_loss:.2f}")
        print(f"Model Accuray: {model_acc*100:.1f}%")

        return self.model


def main():
    # Set the working directory to p2
    abs_dir_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_dir_path)
    os.chdir(dir_name)

    model = NotMNISTModel()
    model.train().save(f'{os.getcwd()}/notMNIST.h5')


if __name__ == "__main__":
    main()
