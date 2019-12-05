import tensorflow as tf
import os


def train():
    print("--Get data--")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("--Process data--")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("--Make model--")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
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
    # Set the working directory to p1
    abs_dir_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_dir_path)
    os.chdir(dir_name)

    model = train()
    model.save('./MNIST.h5')


if __name__ == "__main__":
    main()
