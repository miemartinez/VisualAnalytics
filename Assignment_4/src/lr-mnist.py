#!/usr/bin/env python
"""
Training a logistic regression classifier for the MNIST data and evaluates the model. Saving the classification metrics and from evaluation as csv file in the out folder as well as printing to the terminal. It also creates and saves a confusion matrix as png file. Can also be used for classifying unseen image of a number between 0-9 by specifying path to the png image. 

Parameters:
    train_size: float <number-between-0-and-1>, default = 0.8
    classification_tolerance: float <number-between-0-and-1>, default = 0.1
    filename: str <choose-filename-for-csv>, default = "lr_classification_metrics.csv"
    path2image: str <path-to-unseen-image>
Usage:
    lr_mnist.py -t <train-size> -c <classification-tolerance> -f <chosen-filename> -p <path-to-image>
Example:
    $ python3 lr_mnist.py -t 0.7 -c 0.05 -p ../data/test.png
    
## Task
- Making a logistic regression classifier as a command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal.
- Have the script save the classifier report in a folder called out, as well as printing it to screen. 
- The user should be able to define the filename as a command line argument
- Allow the user to define Logistic Regression parameters using command line arguments 
- Allow the user to import some unseen image, process it, and use the trained model to predict it's value
"""
# import libraries
import os
import sys
sys.path.append(os.path.join(".."))
import argparse

# import teaching utils
import numpy as np
import pandas as pd
import utils.classifier_utils as clf_util
import matplotlib.pyplot as plt
import cv2

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# argparse 
ap = argparse.ArgumentParser()
# adding argument
ap.add_argument("-t", "--train_size", default = 0.8, help = "Percentage of data to use for training")
ap.add_argument("-c", "--classification_tolerance", default = 0.1, help = "Stopping criterion")
ap.add_argument("-f", "--filename", default = "lr_classification_metrics.csv", help = "Define filename for csv of classification metrics")
ap.add_argument("-p", "--path2image", required = False, help = "If you want to test the model on unseen data, add the file path of an unseen image")
# parsing arguments
args = vars(ap.parse_args())



def main(args):
    # defining arguments from command line
    # train size
    train_size = float(args["train_size"])
    # tolerance level
    tol = float(args["classification_tolerance"])
    # chosen filename for classification metrics csv
    filename = args["filename"]
    # path to unseen image if path is specified
    path2image = 0
    if args["path2image"] is not None:
        path2image = args["path2image"]
    # otherwise set image path to False
    else:
        path2image == False
    
    
    # Create class object with train size and tolerance level
    lr_classifier = LogRegClassifier(train_size = train_size, tolerance = tol, filename = filename, path2image = path2image)
    
    # use method train_class
    # training model and saving it as clf (also returning the test data)
    clf, X_test_scaled, y_test = lr_classifier.train_class()
    
    
    # use method eval_classifier
    lr_classifier.eval_classifier(clf, X_test_scaled, y_test)

    
    # test classifier if an unseen image is passed as argument. Otherwise pass.
    if path2image == False:
        pass
    else:
        lr_classifier.test_model(clf)

    
    
class LogRegClassifier:
    
    def __init__(self, train_size, tolerance, filename, path2image):
        '''
        Constructing the Classification object
        '''
        # defining the self attributes 
        self.train_size = train_size
        self.tol = tolerance
        self.filename = filename
        self.path2image = path2image
        
        # fetch the mnist data
        print("Fetching the data -- This may take a while so I suggest you take this time to go make a nice cup of coffee (or tea).")
        X, y = fetch_openml("mnist_784", version = 1, return_X_y=True)
       
        # make sure the data is type numpy array and saved as attributes to the class object
        self.X = np.array(X)
        self.y = np.array(y)
    
    
    
    def train_class(self):
        '''
        Creating training data and training the classification model
        '''
        # Create training and test data
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y,
                                                            random_state=9,
                                                            train_size=self.train_size) # argparse - make this the default
        # Min-Max scaling of X
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())

        # Train Logistic Regression Classifier
        clf = LogisticRegression(penalty='none', 
                         tol=self.tol, # using the default or user specified tolerance level
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
        
        # return the model and test data to use for the eval_classifier method
        return clf, X_test_scaled, y_test
    
    def eval_classifier(self, clf, X_test_scaled, y_test):
        '''
        Creates the output directory. Makes predictions for test data to evaluate the accuracy of the model. Saves classification metrics 
        as csv and a confusion matrix as png.
        '''
        # Create out directory if it doesn't exist in the data folder
        dirName = os.path.join("..", "data", "out")
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            # print that it has been created
            print("Directory " , dirName ,  " Created ")
        else:   
            # print that it exists
            print("Directory " , dirName ,  " already exists")
        
        
        # Predictions for scaled test data
        y_pred = clf.predict(X_test_scaled)

        # Determining the classification metrics
        classification_metrics = metrics.classification_report(y_test, y_pred) # return as dictionary
        print(classification_metrics) # Print in terminal

        # define as pandas dataframe and save as csv in the out folder
        path = os.path.join("..", "data", "out", self.filename)
        # transpose and make into a dataframe
        classification_metrics_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True)).transpose()
        # saving as csv
        classification_metrics_df.to_csv(path)
        # print that the csv file has been saved
        print(f"Classification metrics are saved as {path}")

        # Create confusion matrix and save in output folder
        path = os.path.join("..", "data", "out", "confusion_matrix.png")
        # plot the confusion matrix
        confusion_matrix = clf_util.plot_cm(y_test, y_pred, normalized=True)
        # save the matrix as png
        plt.savefig(path, dpi = 300, bbox_inches = "tight")
        # print that the matrix has been saved
        print(f"Confusion matrix is saved as {path}")
        
    def test_model(self, clf):
        '''
        Method for testing the clf model on an unseen image.
        '''
        print("**Testing the image you gave me**")
        # reading image from path
        test_image = cv2.imread(self.path2image)
        # gray scaling image
        gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
        # compressing image to match the classifier
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        # defining classes
        classes = sorted(set(self.y))
        # making prediction and printing results
        print("")
        print(clf_util.predict_unseen(compressed, clf, classes))


# behavior if run from command line
if __name__ == "__main__":
    main(args)



