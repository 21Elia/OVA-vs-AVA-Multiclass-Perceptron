import numpy as np
from binary_perceptron import BinaryPerceptron

class MulticlassOVAPerceptron:
    def __init__(self, n_classes, max_epochs = 100):
        self.max_epochs = max_epochs
        self.perceptrons = []
        self.classes = n_classes

    def train(self, training_set):
        training_X = training_set[0]
        training_y = training_set[1]

        # main training loop: for each binary classifier I change the labels based on the case some_digit VS ALL
        for digit in range(self.classes):
            
            # replace digit labels with +1 and all other labels with -1
            y_binary = np.where(training_y == digit, 1, -1) # np.where(condition, value_if_true, value_if_false)
            training_tuple = (training_X, y_binary) # new training set on which I will train the models

            model = BinaryPerceptron(self.max_epochs)
            
            print(f"Training binary perceptron for the digit: {digit} vs ALL")
            model.train(training_tuple)
            self.perceptrons.append(model)

    def predict(self, x):
        # The OVA technique consists on predicting the class that has the highest w^T * x + b score
        scores = []
        
        for perceptron in self.perceptrons: 
            score = perceptron.decision_function(x)
            scores.append(score)
        
        return np.argmax(scores) # returning the index of the maximum score (which will be the predicted digit)
    
    def test(self, test_set):
        test_X = test_set[0]
        test_y = test_set[1]

        # init
        confusion_matrix = np.zeros((10,10), dtype='int16')
        n_samples = test_X.shape[0]
        n_errors = 0

        print(f"\n Testing... \n")
        for i in range(n_samples):
            prediction = self.predict(test_X[i])
            label = test_y[i]

            if prediction != label:
                n_errors += 1
    
            confusion_matrix[label, prediction] += 1

        accuracy = (n_samples - n_errors) / n_samples * 100
        return accuracy, confusion_matrix