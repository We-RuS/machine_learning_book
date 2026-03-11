import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=100):

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.losses = []

    def fit(self, X, y_true):
        self.weights = np.zeros(X.shape[1])
        self.bias = np.zeros(1)

        for i in range(self.n_iter):
            z = self.net_input(X)
            output = self.activate(z)
            d_loss__d_a = (y_true - output)
            self.weights -= self.learning_rate * np.dot(d_loss__d_a.T, X) / X.shape[0]
            self.bias -= self.learning_rate * d_loss__d_a.mean()
            loss = (- y_true.dot(np.log(output)) -
                    ((1 - y_true).dot(np.log(1 - output))) / X.shape[0])
            self.losses.append(loss)

    def net_input(self, X):
        return np.dot(X, self.weights.T) + self.bias

    def activate(self, net_input):
        return 1 / 1 + np.exp(-np.clip(net_input, -250, 250))

    def predict(self, X):
        return np.where(self.activate(self.net_input(X)) > 0.5, 1, 0)
