from numpy.random import RandomState
import tensorflow as tf
import numpy as np
import pandas
import os
import functools


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


class CHDModel:
    def __init__(self):
        self.num_data_columns = ['sbp', 'tobacco', 'ldl',
                                 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
        self.categories = {
            'famhist': ['Present', 'Abscent']
        }
        self.train_file_path = f'{os.getcwd()}/heart_train.csv'
        self.test_file_path = f'{os.getcwd()}/heart_test.csv'

        train_data, test_data = self._split_dataset()

        self.train_data = train_data
        self.test_data = test_data

    def _split_dataset(self):
        """
        Splits the dataset using pandas and writes to corresponding csv files
        """
        dataset = pandas.read_csv(f'{os.getcwd()}/heart.csv')
        train_data = dataset.sample(frac=0.6, random_state=RandomState())
        test_data = dataset.loc[~dataset.index.isin(train_data.index)]

        train_data.to_csv(self.train_file_path, index=False)
        test_data.to_csv(self.test_file_path, index=False)

        return train_data, test_data

    def _get_dataset(self, is_train=True, **kwargs):
        if is_train:
            file = self.train_file_path
        else:
            file = self.test_file_path

        dataset = tf.data.experimental.make_csv_dataset(
            file,
            batch_size=50,
            label_name='chd',
            na_value='?',
            num_epochs=1,
            ignore_errors=True,
            **kwargs)

        return dataset

    def _get_packed_data(self):
        packed_train_data = self._get_dataset().map(
            PackNumericFeatures(self.num_data_columns))
        packed_test_data = self._get_dataset(is_train=False).map(
            PackNumericFeatures(self.num_data_columns))

        return packed_train_data, packed_test_data

    def build_model(self):
        packed_train_data, packed_test_data = self._get_packed_data()

        example_batch, labels_batch = next(iter(packed_train_data))

        desc = pandas.read_csv(self.train_file_path)[
            self.num_data_columns].describe()

        mean = np.array(desc.T['mean'])
        std = np.array(desc.T['std'])

        normalizer = functools.partial(
            normalize_numeric_data, mean=mean, std=std)
        numeric_column = tf.feature_column.numeric_column(
            'numeric', normalizer_fn=normalizer, shape=[len(self.num_data_columns)])
        numeric_columns = [numeric_column]

        numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
        numeric_layer(example_batch).numpy()

        categorical_columns = []
        for feature, vocab in self.categories.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
            categorical_columns.append(
                tf.feature_column.indicator_column(cat_col))

        categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
        preprocessing_layer = tf.keras.layers.DenseFeatures(
            categorical_columns + numeric_columns)

        model = tf.keras.Sequential([
            preprocessing_layer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

        train_data = packed_train_data.shuffle(500)
        test_data = packed_test_data

        model.fit(train_data, epochs=20)
        test_loss, test_accuracy = model.evaluate(test_data)
        print(f"Model Loss:    {test_loss:.2f}")
        print(f"Model Accuracy: {test_accuracy*100:.1f}%")

        predictions = model.predict(test_data)

        # Show some results
        for prediction, has_chd in zip(predictions[:10], list(test_data)[0][1][:10]):
            print("CHD prediction percentage: {:.2%}".format(
                prediction[0]), " | Actual outcome: ", ("CHD" if bool(has_chd) else "NO CHD"))

        return model

    def _show_batch(self, dataset):
        for batch, label in dataset.take(1):
            for key, value in batch.items():
                print("{:20s}: {}".format(key, value.numpy()))


def main():
    abs_dir_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_dir_path)
    os.chdir(dir_name)

    model = CHDModel()
    model.build_model()


if __name__ == "__main__":
    main()
