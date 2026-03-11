import numpy as np


class Perceptron:
    def __init__(self, random_state=1, n_iter=10, eta=0.1):
        self.random_state = random_state
        self.n_iter = n_iter
        self.eta = eta
        self.errors = []

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.b = np.float32(0.)

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
