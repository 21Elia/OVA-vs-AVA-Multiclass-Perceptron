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

        for digit in range(self.classes):
            # I have to change the labels from the training set for each different digit I want to train the model on.
            # For the digit of interest I will replace the label with '+1' and for every other label which
            # is not of the corresponding digit of interest I will replace '-1'

            y_binary = np.where(training_y == digit, 1, -1) # np.where(condition, value_if_true, value_if_false)
            training_tuple = (training_X, y_binary) # new training set on which I will train the models

            model = BinaryPerceptron(self.max_epochs)
            
            print(f"Training binary perceptron for the digit: {digit} vs ALL")
            model.train(training_tuple)
            self.perceptrons.append(model)

    def predict(self, x): # x is an example from the data set
        # The OVA technique consists on predicting the class that has the highest w^T * x + b score
        scores = []
        for perceptron in self.perceptrons: 
            score = perceptron.decision_function(x)
            scores.append(score)
        
        return np.argmax(scores) # returning the index of the maximum score (which will be the predicted digit)
    
    def test(self, test_set):
        test_X = test_set[0]
        test_y = test_set[1]

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