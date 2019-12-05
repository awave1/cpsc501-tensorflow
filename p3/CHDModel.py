from numpy.random import RandomState
import pandas
import os


def split_dataset():
    dataset = pandas.read_csv(f'{os.getcwd()}/heart.csv')
    train_data = dataset.sample(frac=0.6, random_state=RandomState())
    test_data = dataset.loc[~dataset.index.isin(train_data.index)]
    return train_data, test_data


def main():
    split_dataset()


if __name__ == "__main__":
    main()
