import numpy as np

class Adaline:
    def __init__(self, eta=0.01, random_state=1, n_iter=10):
        self.eta = eta
        self.random_state = random_state
        self.n_iter = n_iter
        self.losses = []

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = np.float32(0.0)
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            loss = y - output
            self.w_ += self.eta * 2. * X.T.dot(loss) / X.shape[0]
            self.b_ += self.eta * 2. * loss.mean()
            self.losses.append((loss ** 2).mean())
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
