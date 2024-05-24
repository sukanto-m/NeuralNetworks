import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, X, y, layer_sizes, eta, n_iter, initialization_method='he', lambd=0.01):
        """
        Initializes the neural network.
        
        Parameters:
        - X: Input features (training data).
        - y: True labels (target values).
        - layer_sizes: list of integers, where the i-th element represents the number of neurons in the i-th layer.
        - eta: Learning rate for training the network.
        - n_iter: Number of iterations for training.
        - initialization_method: Method to initialize weights ('he', 'xavier', 'random_normal').
        - lambd: Regularization parameter.
        """
        self.layer_sizes = layer_sizes
        self.eta = eta
        self.n_iter = n_iter
        self.input_data = X
        self.true_labels = y
        self.initialization_method = initialization_method
        self.lambd = lambd
        self.weights, self.biases = self.initialise_weights()
        self.train_losses = []

    def initialise_weights(self):
        weights = []
        biases = []

        for i in range(1, len(self.layer_sizes)):
            if self.initialization_method == 'he':
                weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2. / self.layer_sizes[i-1])
            elif self.initialization_method == 'xavier':
                weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(1. / self.layer_sizes[i-1])
            elif self.initialization_method == 'random_normal':
                weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1])
            bias_vector = np.zeros((self.layer_sizes[i], 1))
            weights.append(weight_matrix)
            biases.append(bias_vector)
        
        return weights, biases

    def forward_pass(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i].T
            if i < len(self.layer_sizes) - 2:
                a = self.sigmoid(z)
            else:
                a = z  # Linear activation for the output layer
            activations.append(a)
        return activations

    def backward_pass(self, X, y, activations):
        m = X.shape[0]
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Output layer error
        delta = activations[-1] - y

        for i in reversed(range(len(dW))):
            dW[i] = (1/m) * np.dot(delta.T, activations[i])
            db[i] = (1/m) * np.sum(delta, axis=0, keepdims=True).T

            if i != 0:
                delta = np.dot(delta, self.weights[i]) * self.sigmoid_derivative(activations[i])
        
        return dW, db

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def update_params(self, dW, db, clip_value=5):
        for i in range(len(self.weights)):
            dW[i] = np.clip(dW[i], -clip_value, clip_value)
            db[i] = np.clip(db[i], -clip_value, clip_value)
            self.weights[i] -= self.eta * (dW[i] + self.lambd * self.weights[i])
            self.biases[i] -= self.eta * db[i]

    def train(self, batch_size=32, patience=10):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_iter):
            X_shuffled, y_shuffled = self.shuffle_data(self.input_data, self.true_labels)
            for i in range(0, X_shuffled.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                activations = self.forward_pass(X_batch)
                loss = self.loss(y_batch, activations[-1])
                dW, db = self.backward_pass(X_batch, y_batch, activations)
                self.update_params(dW, db)

                if (i // batch_size) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_iter}, Batch {i//batch_size+1}/{X_shuffled.shape[0]//batch_size}, Loss: {loss}")

            self.train_losses.append(loss)

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    return

    def shuffle_data(self, X, y):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def predict(self, X):
        activations = self.forward_pass(X)
        predictions = activations[-1]
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self.loss(y, predictions)
        return loss

    def loss(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred)**2)
        return loss

    def regularization_loss(self):
        reg_loss = 0
        for w in self.weights:
            reg_loss += 0.5 * self.lambd * np.sum(w**2)
        return reg_loss

    def total_loss(self, data_loss, reg_loss):
        total_loss = data_loss + reg_loss
        return total_loss

    def visualize_learning_curves(self, val_losses=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()