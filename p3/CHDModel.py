from numpy.random import RandomState
import pandas
import os


def split_dataset():
    dataset = pandas.read_csv(f'{os.getcwd()}/heart.csv')
    train_data = dataset.sample(frac=0.6, random_state=RandomState())
    test_data = dataset.loc[~dataset.index.isin(train_data.index)]

    train_data.to_csv('heart_train.csv')
    test_data.to_csv('heart_test.csv')

    return train_data, test_data


def main():
    abs_dir_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_dir_path)
    os.chdir(dir_name)

    train, test = split_dataset()


if __name__ == "__main__":
    main()
