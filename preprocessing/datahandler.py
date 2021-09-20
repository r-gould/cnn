import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataHandler:
    def load_dataset(self, root, shuffle=True):
        train_data = pd.read_csv(root + "train.csv", header=None).to_numpy()
        test_data = pd.read_csv(root + "test.csv", header=None).to_numpy()
        
        if shuffle:
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)

        (X_train, y_train) = self.get_features_and_target(train_data)
        (X_test, y_test) = self.get_features_and_target(test_data)
        return (X_train, y_train), (X_test, y_test)

    def get_features_and_target(self, dataset):
        X = dataset[:, 1:]
        y = dataset[:, :1]
        return X, y

    def split_data(self, data, split):
        m = data.shape[0]
        bound = int(m * split)
        data_train = data[:bound, :]
        data_valid = data[bound:, :]
        return (data_train, data_valid)

    def one_hot_encode(self, y, classes=10):
        m = y.shape[0]
        y_encoded = np.zeros((m, classes))
        y_encoded[np.arange(0, m), y.T] = 1
        return y_encoded