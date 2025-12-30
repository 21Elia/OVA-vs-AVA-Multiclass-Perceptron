import numpy as np

class BinaryPerceptron:
    def __init__(self, max_epochs = 100):
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0

    def train(self, training_set:tuple):
        X = training_set[0]
        y = training_set[1]

        norms = np.linalg.norm(X, axis = 1)
        R = np.max(norms)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        k = 0
        n_errors = -1
        epoch = 0

        while(n_errors != 0 and epoch < self.max_epochs):
            n_errors = 0
            for i in range(n_samples):
                if (y[i] * self.decision_function(X[i]) <= 0):
                    self.weights = self.weights + y[i] * X[i]
                    self.bias = self.bias + y[i]*(R ** 2) 
                    k += 1
                    n_errors += 1
            
            epoch += 1
            # printing progress
            print(f"Epoch {epoch}: {n_errors} errors")

    def decision_function(self, x): # x is an example from the data set
        return np.dot(self.weights, x) + self.bias
    
    def predict(self, x):
        score = self.decision_function(x)
        if score >= 0:
            return 1
        else:
            return -1