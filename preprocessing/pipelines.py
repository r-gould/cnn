import numpy as np

from .datahandler import DataHandler

def mnist_pipeline(root, train_split, shuffle=True):
    data = DataHandler()
    (X_train_valid, y_train_valid), (X_test, y_test) = data.load_dataset(root, shuffle)

    (X_train, X_valid) = data.split_data(X_train_valid, train_split)
    (y_train, y_valid) = data.split_data(y_train_valid, train_split)

    X_train = X_train / 255
    X_valid = X_valid / 255
    X_test = X_test / 255
    
    y_train = data.one_hot_encode(y_train)
    y_valid = data.one_hot_encode(y_valid)
    y_test = data.one_hot_encode(y_test)

    X_train = np.reshape(X_train, (-1, 28, 28, 1))
    X_valid = np.reshape(X_valid, (-1, 28, 28, 1))
    X_test = np.reshape(X_test, (-1, 28, 28, 1))

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)