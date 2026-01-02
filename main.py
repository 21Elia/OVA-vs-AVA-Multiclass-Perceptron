import pandas as pd
import numpy as np # vector manipulations
import seaborn as sns # for the confusion matrix heatmap
import matplotlib.pyplot as plt # for plotting
import time
from scipy.io import arff # to load data from the '.arff' format

from binary_perceptron import BinaryPerceptron
from multiclass_OVA_perceptron import MulticlassOVAPerceptron
from multiclass_AVA_perceptron import MulticlassAVAPerceptron

def load_data(path):
    # extracting real data values and metadata (data such as 'pixel1', 'pixel2' etc)
    data, meta = arff.loadarff(path)
    
    # converting to pandas dataframe
    df = pd.DataFrame(data)
    print(f"{df.head()} \n")
    print(f"Real size: {df.shape} \n\n")

    # extracting the labels from the dataframe into a numpy array: 
    # 'class' column is in raw bytes because scipy.io.arff reads and returns raw bytes to avoid decoding errors
    # (doesn't know if he needs to convert to ascii, unicode or other).
    # So I'm firstly converting it to string, then to integer, and then to numpy array
    y = df['class'].astype(str).astype(int).to_numpy()

    # extracting the x instances into a numpy array
    X = df.drop(columns = 'class').astype(float).to_numpy()
    X = X / 255.0 # normalizing pixels (from [0, 255] to [0, 1])
    return X, y


def shuffle_data(X, y):
    np.random.seed(42) # setting a seed so that every time I run the program I get the same permutation

    # creating an array storing a permutation of the indexes 
    indexes = np.random.permutation(X.shape[0])

    # shuffling examples and labels
    X_shuffled = X[indexes, :]
    y_shuffled = y[indexes]

    return X_shuffled, y_shuffled

def split_data(X, y):
    # defining amount of data I want the model to train on
    limit = 60000
    
    # extracing training examples and corresponding labels
    training_X = X[:limit, :]
    training_y = y[:limit]
    training_set = (training_X, training_y)

    # extracting test examples and corresponding labels
    test_X = X[limit:, :]
    test_y = y[limit:]
    test_set = (test_X, test_y)
    
    return training_set, test_set

def plot_confusion_matrix(confusion_matrix, strategy:str):
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix ' + strategy)
    plt.savefig('images/' + strategy + '-confusion-heatmap')


if __name__ == "__main__":

    # loading the MNIST dataset: converting it from the '.arff' format to numpy arrays
    X, y = load_data('mnist_784.arff')
    X, y = shuffle_data(X, y)

    # creating training_set and test_set
    training_set:tuple # (training_X, training_y)
    test_set:tuple # (test_X, test, y)
    training_set, test_set = split_data(X, y)

    # # checking if everything is working as expected (uncomment if needed)
    # print(f"training_y shape: {training_y.shape}\n") 
    # print(f"prima label y[0]: {training_y[0]}\n\n")

    # print(f"training_X shape: {training_X.shape}\n")
    # print(f"random pixel from the first image of the training set (normalised): {training_X[0][204]}\n\n")

    results = []
    training_X = training_set[0]
    training_y = training_set[1]

    # --- TEST OVA ---
    ova = MulticlassOVAPerceptron(10, max_epochs = 100)

    start = time.perf_counter()
    ova.train(training_set)
    end = time.perf_counter()
    ova_train_time = end - start

    start = time.perf_counter()
    ova_accuracy, ova_confusion_matrix = ova.test(test_set)
    end = time.perf_counter()
    ova_test_time = end - start

    results.append({
        "Modello" : "OVA",
        "Training Time (s)" : round(ova_train_time, 4),
        "Test Time (s)": round(ova_test_time, 4),
        "Average Prediction Time (s)" : round(ova_test_time / test_set[0].shape[0], 7),
        "Accuracy %": round(ova_accuracy, 2)
    })


    # --- TEST AVA ---
    ava = MulticlassAVAPerceptron(10, max_epochs = 100)

    start = time.perf_counter()
    ava.train(training_set)
    end = time.perf_counter()
    ava_train_time = end - start

    start = time.perf_counter()
    ava_accuracy, confusion_matrix_AVA = ava.test(test_set)
    end = time.perf_counter()
    ava_test_time = end - start

    results.append({
        "Modello" : "AVA",
        "Training Time (s)" : round(ava_train_time, 4),
        "Test Time (s)" : round(ava_test_time, 4),
        "Average Prediction Time (s)" : round(ava_test_time / test_set[0].shape[0], 8),
        "Accuracy %" : round(ava_accuracy, 2)

    })

    # plotting confusion matrices
    plot_confusion_matrix(ova_confusion_matrix, 'OVA')
    plot_confusion_matrix(confusion_matrix_AVA, 'AVA')
    plt.show()

    # printing time results
    df = pd.DataFrame(results)
    print(df.to_string(header = True, index = False, justify = 'left'))
    latex = df.to_latex(index = False, formatters = {"name" : str.upper})
    print(latex)