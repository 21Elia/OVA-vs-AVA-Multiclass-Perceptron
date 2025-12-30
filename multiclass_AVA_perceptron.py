import numpy as np
from binary_perceptron import BinaryPerceptron

class MulticlassAVAPerceptron:
    def __init__(self, n_classes, max_epochs = 100):
        self.max_epochs = max_epochs
        self.perceptrons = {}
        self.classes = n_classes

    def train(self, training_set:tuple):
        training_X = training_set[0]
        training_y = training_set[1]

        n = self.classes * (self.classes - 1) / 2

        # initializing all possible pairs which are C(K, 2) = K*(K-1) / 2
        for i in range(self.classes):
            for j in range(i + 1, self.classes): # j = i + 1 to avoid duplicates pairs and pairs like (i, j) where i == j
                self.perceptrons[(i, j)] = None
        
        for (i, j) in self.perceptrons.keys():
            first_digit, second_digit = (i, j)
            
            # creating an index mask to filter out data
            mask = (training_y == first_digit) | (training_y == second_digit)

            new_X = training_X[mask, :] # selects only the rows of interest
            new_y = training_y[mask]
            # labels in new_y are still non binary so I convert it to binary
            new_y_binary = np.where(new_y == first_digit, 1, -1) 
            
            training_tuple = (new_X, new_y_binary)
    
            model = BinaryPerceptron(self.max_epochs)

            print(f"Training {i} vs {j}")
            model.train(training_tuple)
            self.perceptrons[(i, j)] = model

    def predict(self, x): # x is an example from the data set
        # The AVA technique consits on predicting the class that has most votes at the end of all K*(K-1)/2 challenges

        votes = [0 for _ in range(self.classes)] # votes is a list of 10 elements where for each digit I have its votes counter

        for (i, j) in self.perceptrons.keys():
            perceptron = self.perceptrons[(i, j)]
            prediction = perceptron.predict(x)
            if(prediction == 1):
                votes[i] += 1
            else:
                votes[j] += 1
        
        return np.argmax(votes)
    
    def test(self, test_set):
        test_X = test_set[0]
        test_y = test_set[1]
        
        confusion_matrix = np.zeros((10,10), dtype='int16')
        n_samples = test_X.shape[0]
        n_errors = 0

        print(f"Testing...\n")
        for i in range(n_samples):
            prediction = self.predict(test_X[i])
            label = test_y[i]

            if(prediction != label):
                n_errors += 1
            
            confusion_matrix[label, prediction] += 1
        
        accuracy = (n_samples - n_errors) / n_samples * 100
        return accuracy, confusion_matrix