import numpy as np
import time
import neural.layers as layers

from .display import Display


class NeuralNetwork:
    def __init__(self, input_dim, loss):
        self.layers = []
        self.input_dim = input_dim
        self.loss = loss
        self.configured = False

    def add(self, layer_obj):
        if self.configured:
            raise RuntimeError("Adding layers after configuration is not allowed.")
        
        self.layers.append(layer_obj)
    
    def configure(self, learning_rate):
        self.learning_rate = learning_rate
        self.configured = True
        self.display = Display()

        self._initialize_layers()

    def _initialize_layers(self):
        input_dim = self.input_dim

        for layer in self.layers:
            layer._initialize(input_dim)

            input_dim = layer.output_dim

    def train(self, X_train, y_train, epochs, batch_size, valid_set=None, show_statistics=True):
        if not self.configured:
            raise RuntimeError("Network has not been configured.")
        
        m = X_train.shape[0]
        batch_count = m // batch_size
        costs = []

        print("Training started...\n")

        for epoch in range(1, epochs+1):
            start_time = time.time()

            for i in range(batch_count):
                X_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                
                yhat_batch = self.train_batch(X_batch, y_batch)

            end_time = time.time()
            train_time = end_time - start_time

            if show_statistics:
                start_time = time.time()

                yhat_train = self.predict(X_train, batch_size=batch_size)
                train_stats = self.get_cost_accuracy(y_train, yhat_train)
                
                (train_cost, train_acc) = train_stats
                costs.append(train_cost)

                valid_stats = None
                if valid_set:
                    (X_valid, y_valid) = valid_set
                    yhat_valid = self.predict(X_valid, batch_size=batch_size)
                    valid_stats = self.get_cost_accuracy(y_valid, yhat_valid)

                end_time = time.time()
                predict_time = end_time - start_time

                self.display.display_train_statistics(epoch, train_time, predict_time, train_stats, valid_stats)

        return costs
    
    def train_batch(self, X_batch, y_batch):
        yhat_batch = self._forward_propagate(X_batch, training=True)
        self._backward_propagate(y_batch, yhat_batch)
        self._update_parameters()
        return yhat_batch

    def get_cost_accuracy(self, y, yhat):
        cost = self.cost(y, yhat)
        acc = self.evaluate(y, yhat, display_stats=False)
        return (cost, acc)

    def predict(self, X, batch_size=256, y=None):
        m = X.shape[0]
        batch_count = m // batch_size

        output_shape = self.layers[-1].output_dim
        yhat = np.zeros((m, *output_shape))

        for i in range(batch_count):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            yhat_batch = self._forward_propagate(X_batch, training=False)
            yhat[i*batch_size:(i+1)*batch_size] = yhat_batch

        if y:
            self.evaluate(y, yhat, display_stats=True)
        
        return yhat

    def evaluate(self, y, yhat, display_stats=True):
        y_preds = np.argmax(y, axis=1)
        yhat_preds = np.argmax(yhat, axis=1)
        acc = np.mean(y_preds == yhat_preds)

        if display_stats:
            self.display.display_evaluation_statistics(acc)

        return acc
    
    def cost(self, y, yhat):
        batch_size = y.shape[0]
        cost = self.loss.calculate_loss(y, yhat) / batch_size
        return cost

    def _forward_propagate(self, X, training=True):
        for layer in self.layers:
            O = layer._forward(X, training)
            
            X = O

        return O

    def _backward_propagate(self, y, yhat):
        dO = self.loss.calculate_gradient(y, yhat)

        for layer in self.layers[::-1]:
            dX = layer._backward(dO)

            dO = dX

    def _update_parameters(self):
        for layer in self.layers:
            if not layer.trainable:
                continue
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
