import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def train():
    print("--Get data--")
    with np.load(f'{os.getcwd()}/notMNIST.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print("--Process data--")
    print(len(y_train))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(
            0.005), input_shape=(28, 28)),
        tf.keras.layers.MaxPool1D(3),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("--Fit model--")
    model.fit(x_train, y_train, epochs=10, verbose=2)

    print("--Evaluate model--")
    model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f"Model Loss:    {model_loss:.2f}")
    print(f"Model Accuray: {model_acc*100:.1f}%")

    return model


def main():
    model = train()
    model.save(f'{os.getcwd()}/notMNIST.h5')


if __name__ == "__main__":
    main()
