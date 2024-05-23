import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, X, layer_sizes, eta=0.1, n_iter=100, initialization_method='he'):
        """
        Initializes the neural network.
        
        Parameters:
        - layer_sizes: list of integers, where the i-th element represents the number of neurons in the i-th layer.
        - learning_rate: learning rate for training the network.
        - initialization_method: method to initialize weights ('he', 'xavier', 'random_normal').
        """

        self.layer_sizes = layer_sizes
        self.eta = eta
        self.n_iter = n_iter
        self.input_data = X
        self.initialization_method = initialization_method
        self.weights, self.biases = self.initialise_weights()

    def initialise_weights(self):
        """
        Initializes weights and biases for each layer in the network.
        
        Returns:
        - weights: list of numpy arrays, where the i-th element represents the weight matrix for the i-th layer.
        - biases: list of numpy arrays, where the i-th element represents the bias vector for the i-th layer.
        """

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
    
    @staticmethod
    def sigmoid(z):
        return 1./(1. + np.exp(-z))
    
    def forward_pass(self, X):
        """
        Perform a forward pass through the network.
        
        Parameters:
        - X: numpy array, input data.
        
        Returns:
        - output: numpy array, output predictions.
        """
        layer_output = X.T

        for i in range(len(self.layer_sizes) - 1):
            z = np.dot(self.weights[i],layer_output) + self.biases[i]

            if i < len(self.layer_sizes) - 2:
                layer_output = self.sigmoid(z)
            else:
                layer_output = z

        output = layer_output

        return output

    def backprop(self, X, y):
        """
        Perform a backward pass through the network to compute gradients.
        
        Parameters:
        - X: numpy array, input data.
        - y: numpy array, true labels.
        
        Returns:
        - dW: list of numpy arrays, gradients of weights.
        - db: list of numpy arrays, gradients of biases.
        """
        #Number of samples

        m = X.shape[0]

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Forward pass
        a = [X.T]
        z = []
        for i in range(len(self.layer_sizes) - 1):
            z_i = np.dot(self.weights[i], a[-1]) + self.biases[i]
            z.append(z_i)
            if i < len(self.layer_sizes) - 2:
                a_i = self.sigmoid(z_i)
            else:
                # Linear activation for output layer
                a_i = z_i
            a.append(a_i)

        delta = (a[-1] - y.T) / m

        # Backpropagate delta
        for i in range(len(self.layer_sizes) - 2, -1, -1):
            dW[i] = np.dot(delta, a[i].T)
            db[i] = np.sum(delta, axis=1, keepdims=True)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(z[i-1])
        
        return dW, db

    def sigmoid_derivative(self,z):
        """
        Compute the derivative of the sigmoid activation function.
        
        Parameters:
        - z: numpy array, weighted sum of inputs.
        
        Returns:
        - sigmoid_derivative: numpy array, derivative of sigmoid.
        """
        sigmoid_derivative = self.sigmoid(z) * (1 - self.sigmoid(z))
        return sigmoid_derivative

    def train(self, batch_size=32):
        for epoch in range(self.n_iter):
            #Shuffle training data 
            X_shuffled, y_shuffled = self.shuffle_data(self.input_data, self.true_labels)
            #Perform mini batch gradient descent
            for i in range(0, X_shuffled.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                activations = self.forward_pass(X_batch)

                loss = self.loss(y_batch, activations[-1])

                dW, db = self.backprop(X_batch, y_batch)

                self.update_params(dW, db)

                # Print training progress
                if (i // batch_size) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_iter}, Batch {i//batch_size+1}/{X_shuffled.shape[0]//batch_size}, Loss: {loss}")

    def shuffle_data(self, X, y):
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    def update_params(self, dW, db):
        """
        Update weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * dW[i]
            self.biases[i] -= self.eta * db[i]

    def predict(self, X):
        """
        Make predictions on new data.
        """
        activations = self.forward_pass(X)
        predictions = activations[-1]
        return predictions

    def evaluate(self, X, y):
        """
        Evaluate model performance on validation or test data.
        """
        predictions = self.predict(X)
        accuracy = self.compute_accuracy(predictions, y)
        return accuracy

    def loss(self, y_true, y_pred):
        """
        Compute the mean squared error loss.
        """
        loss = np.mean((y_true - y_pred)**2)
        return loss

    def regularization_loss(self, lambd):
        """
        Compute regularization loss (if applicable).
        """
        # Implement regularization loss calculation here
        reg_loss = 0  # Placeholder
        for w in self.weights:
            reg_loss += 0.5 * lambd * np.sum(w ** 2)
        return reg_loss

    def total_loss(self, data_loss, reg_loss):
        """
        Compute total loss by combining data and regularization losses.
        """
        total_loss = data_loss + reg_loss
        return total_loss

    def visualize_learning_curves(self, train_losses, val_losses=None):
        """
        Plot learning curves (training and validation losses).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()

    def compute_accuracy(self, predictions, true_labels):
        """
        Compute accuracy score.
        """
        accuracy = np.mean(predictions == true_labels)
        return accuracy                          





    



