import numpy as np

class Perceptron:
    def __init__(self, n_features, eta=0.01, n_iter=10):
        self.n_features = n_features
        self.eta = eta
        self.n_iter = n_iter
        self.weights, self.bias = self.initialise_weights()

    def initialise_weights(self):
        weights = np.random.randn(self.n_features)
        bias = np.random.randn()
        return weights, bias
    
    def output(self, x):
        y = np.dot(x.T, self.weights) + self.bias
        return 1 if y >= 0 else 0
    
    def fit(self, X, y):
        for _ in range(self.n_iter):
            for xi, target in zip(X,y):
                prediction = self.output(xi)
                update = self.eta * (target - prediction)
                self.weights += update * xi
                self.bias += update


if __name__ == '__main__':
    X = np.array([[0,0], [0,1], [1,1]])
    y = np.array([0, 0, 1])
    perceptron = Perceptron(n_features=2, eta=0.1, n_iter=100)
    perceptron.fit(X,y)

    for xi, yi in zip(X,y):
        print(f"Input {xi}, Predicted: {perceptron.output(xi)}, Error: {yi - perceptron.output(xi)}")
                







