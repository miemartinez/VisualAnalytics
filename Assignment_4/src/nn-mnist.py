#!/usr/bin/env python
"""
Training a logistic regression classifier for the MNIST data and evaluates the model. Saving the classification metrics from evaluation as csv file in the out folder as well as printing to the terminal.

Parameters:
    train_size: float <number-between-0-and-1>, default = 0.8
    filename: str <choose-filename-for-csv>, default = "nn_classification_metrics.csv"
Usage:
    nn_mnist.py -t <train-size> -f <chosen-filename>
Example:
    $ python3 nn_mnist.py -t 0.7 -f "nn_evaluation.csv"
    
## Task
- Making a neural network classifier as a command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal.
- Have the script save the classifier report in a folder called out, as well as printing it to screen. 
- The user should be able to define the filename as a command line argument
- Allow the user to define train size using command line arguments 
"""
# import libraries
import os
import sys
sys.path.append(os.path.join(".."))
import argparse

# import teaching utils
import numpy as np
import pandas as pd
# Neural networks with numpy
from utils.neuralnetwork import NeuralNetwork 

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# argparse 
ap = argparse.ArgumentParser()
# adding argument
ap.add_argument("-t", "--train_size", default = 0.8, help = "Percentage of data to use for training")
ap.add_argument("-f", "--filename", default = "nn_classification_metrics.csv", help = "Define filename for csv of classification metrics")
# parsing arguments
args = vars(ap.parse_args())




def main(args):
    filename = args["filename"]
    train_size = float(args["train_size"])
    
    # Create class object with train size and tolerance level
    nn_classifier = NeuralNetClassifier(filename = filename, train_size = train_size)
    # use method train_class
    # training model and saving it as clf (also returning the test data)
    nn, X_test_scaled, y_test = nn_classifier.train_class()
    
    
    # use method eval_classifier
    nn_classifier.eval_classifier(nn, X_test_scaled, y_test)

    
class NeuralNetClassifier:
    
    def __init__(self, train_size, filename):
        '''
        Constructing the Classification object
        '''
        # train size
        self.train_size = train_size
        # user chosen filename
        self.filename = filename
        
        # fetch the mnist data
        print("Fetching the data -- This may take a while so I suggest you take this time to go make a nice cup of coffee (or tea).")
        X, y = fetch_openml("mnist_784", version = 1, return_X_y=True)
        # make sure the data is type numpy array 
        self.X = np.array(X)          # if testing script, subset with: np.array(X[:1000])
        self.y = np.array(y)          # if testing script, subset with: np.array(y[:1000])
    
    
    
    def train_class(self):
        '''
        Creating training data and training the classification model
        '''
        # Create training and test data
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y,
                                                            random_state=9,
                                                            train_size=self.train_size) # argparse - make this the default
        # Min-Max scaling
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
        
        # convert labels from integers to vectors
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)

        # train network using the NeuralNetwork class from utils
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train_scaled.shape[1], 32, 16, 10])
        # printing progress
        print("[INFO] {}".format(nn))
        nn.fit(X_train_scaled, y_train, epochs=1000)
        
        return nn, X_test_scaled, y_test
    
    def eval_classifier(self, nn, X_test_scaled, y_test):
        # Create out directory if it doesn't exist in the data folder
        dirName = os.path.join("..", "data", "out")
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:   
            print("Directory " , dirName ,  " already exists")
            
        # evaluate network
        print(["[INFO] evaluating network..."])
        # testing the model on the test data
        predictions = nn.predict(X_test_scaled)
        predictions = predictions.argmax(axis=1)
        # calculating classification metrics
        classification_metrics = classification_report(y_test.argmax(axis=1), predictions) # return as dictionary
        print(classification_metrics) # Print in terminal
        
        # define as pandas dataframe and save as csv in the out folder
        path = os.path.join("..", "data", "out", self.filename)
        # transpose and make into a dataframe
        classification_metrics_df = pd.DataFrame(classification_report(y_test.argmax(axis=1), predictions, output_dict = True)).transpose()
        # saving as csv
        classification_metrics_df.to_csv(path)
        # print that the csv file has been saved
        print(f"Classification metrics are saved as {path}")

if __name__ == "__main__":
    main(args)

