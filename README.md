# Multiclass Perceptron: OVA vs AVA
This project implements and compares two multiclass classification strategies: **One-vs-All (OVA)** and **All-vs-All (AVA)**.
The underlying model used to implement the two strategies is the standard Rosenblatt's Perceptron. The comparison is performed on the **MNIST dataset** (handwritten digits recognition),
measuring accuracy, confusion matrices and execution times.

# Before Execution
To ensure the program runs correctly, the following libraries are required:
- [Numpy](https://numpy.org/): Used for all matrix operations and vector manipulations.
- [pandas](https://pandas.pydata.org/) : Used to put the results into readable tables and to manage the dataframe operations during the data loading phase.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/): Used to visualize and plot confusion matrices as heatmaps allowing to visually analize the errors made by the model.
- [SciPy](https://scipy.org/): Used specifically for its io.arff module, which allows loading the MNIST dataset provided in .arff format.

All the listed modules can be installed with the [pip](https://pip.pypa.io/en/stable/) package installer by typing `pip install <package_name>` in the command line. 

# Dataset Setup
Since the dataset is not included in the repository (the file is too large), you can set it up manually by downloading the [MNIST](https://www.openml.org/search?type=data&sort=runs&id=554&status=active) dataset
in the `.arff` format and by placing the file into the main folder with all the other code files. 

# Code

Once all the necessary libraries and the dataset are installed, you can execute the testing code located in **main.py** file.
