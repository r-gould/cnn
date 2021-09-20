import numpy as np
import matplotlib.pyplot as plt
import neural.layers as layers
import neural.losses as losses

from preprocessing.pipelines import mnist_pipeline
from neural.network import NeuralNetwork

def mnist():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = mnist_pipeline("Convolutional Neural Network/datasets/mnist/", train_split=0.9)

    loss = losses.CrossEntropy()

    network = NeuralNetwork(input_dim=(28, 28, 1), loss=loss)
    network.add(layers.Conv(32, (5, 5), stride=1, pad="valid"))
    network.add(layers.Activation("relu"))
    network.add(layers.MaxPool((2, 2), stride=2))
    network.add(layers.Conv(64, (5, 5), stride=1, pad="valid"))
    network.add(layers.Activation("relu"))
    network.add(layers.MaxPool((2, 2), stride=2))
    network.add(layers.Flatten())
    network.add(layers.Dense(128))
    network.add(layers.Activation("relu"))
    network.add(layers.Dense(10))
    network.add(layers.Activation("softmax"))

    network.configure(learning_rate=0.01)
    costs = network.train(X_train, y_train, epochs=3, batch_size=32, valid_set=(X_valid, y_valid))

    plt.plot(costs)
    plt.show()

    yhat_test = network.predict(X_test)
    network.evaluate(y_test, yhat_test)

if __name__ == "__main__":
    mnist()